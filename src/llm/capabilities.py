"""Per-model request capabilities.

Model generations differ in which request shapes they accept: how thinking is
configured, whether an assistant prefill is allowed, whether a forced
tool_choice is accepted, whether replayed thinking blocks are bound to the
conversation that produced them, and which field carries the output token cap. Backends and the tool loop read those
differences from here instead of matching model names inline.

A model in the `MODELS` catalog gets its profile from its entry, for every
request to that transport and model. Any other model's profile is inferred
from its name and transport.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

from src import config
from src.config import (
    ForcedToolChoice,
    MaxTokensParam,
    ModelCatalogEntry,
    ModelTransport,
    ThinkingMode,
)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What one model accepts in a request."""

    thinking: ThinkingMode
    """`budget`: `{"type": "enabled", "budget_tokens": N}`. `adaptive`:
    `{"type": "adaptive"}` plus `output_config.effort`; a budget is rejected."""
    supports_prefill: bool
    """Whether a trailing assistant message (e.g. a `{` JSON prefill) is accepted."""
    forced_tool_choice: ForcedToolChoice
    preserved_thinking: bool
    """Whether a replayed thinking block is checked against everything sent
    before it (system, tools, earlier messages). Any edit to that prefix makes
    the block a 400 on enforced accounts, so edited history must drop it."""
    user_after_tool: bool = True
    """Whether a user message may directly follow a tool result. Mistral
    models are trained only on tool results followed by an assistant turn or
    another result, and their servers reject a request that has a user turn
    there or that ends on an assistant turn."""
    max_tokens_param: MaxTokensParam = "max_tokens"
    """The request field the OpenAI transport puts the output token cap in.
    OpenAI's reasoning models reject `max_tokens` with a 400 and take
    `max_completion_tokens`."""


# Claude 3 and other models this table doesn't recognize.
_LEGACY = ModelCapabilities(
    thinking="budget",
    supports_prefill=True,
    forced_tool_choice="conflicts_with_thinking",
    preserved_thinking=False,
)
# Claude 4.0 through 4.5.
_CLAUDE_4 = ModelCapabilities(
    thinking="budget",
    supports_prefill=False,
    forced_tool_choice="conflicts_with_thinking",
    preserved_thinking=False,
)
# Claude 4.6 through 4.8, and Opus/Sonnet/Fable 5.0.
_CLAUDE_ADAPTIVE = ModelCapabilities(
    thinking="adaptive",
    supports_prefill=False,
    forced_tool_choice="conflicts_with_thinking",
    preserved_thinking=False,
)
# Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1, and any newer Claude this table
# doesn't list yet: each generation has only added restrictions, so an unknown
# new model gets the strictest known behavior.
_CLAUDE_NEWEST = ModelCapabilities(
    thinking="adaptive",
    supports_prefill=False,
    forced_tool_choice="rejected",
    preserved_thinking=True,
)
_CLAUDE_HAIKU_5_5 = ModelCapabilities(
    thinking="adaptive",
    supports_prefill=False,
    forced_tool_choice="skips_thinking",
    preserved_thinking=True,
)

# Exact (family, major, minor) entries that differ from their version band.
_CLAUDE_EXCEPTIONS: dict[tuple[str, int, int], ModelCapabilities] = {
    ("opus", 5, 0): _CLAUDE_ADAPTIVE,
    ("sonnet", 5, 0): _CLAUDE_ADAPTIVE,
    ("fable", 5, 0): _CLAUDE_ADAPTIVE,
    ("mythos", 5, 0): _CLAUDE_ADAPTIVE,
    ("haiku", 5, 5): _CLAUDE_HAIKU_5_5,
}

# claude-<family>-<major>[-<minor>], anywhere in the id so vendor prefixes
# ("anthropic/", "anthropic.") and date or "@" suffixes still match. Proxies
# such as OpenRouter spell the version with a dot (claude-opus-4.6). The minor
# is one or two digits, so a dated id like claude-opus-4-20250514 reads as 4.0.
_CLAUDE_MODEL = re.compile(
    r"claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:[-.](?P<minor>\d{1,2})(?!\d))?"
)


def anthropic_capabilities(model: str) -> ModelCapabilities:
    """Capabilities of a Claude model, by its model id."""
    match = _CLAUDE_MODEL.search(model)
    if match is None:
        return _LEGACY
    family = match["family"]
    major = int(match["major"])
    minor = int(match["minor"] or 0)
    exception = _CLAUDE_EXCEPTIONS.get((family, major, minor))
    if exception is not None:
        return exception
    if major < 4:
        return _LEGACY
    if (major, minor) < (4, 6):
        return _CLAUDE_4
    if major < 5:
        return _CLAUDE_ADAPTIVE
    return _CLAUDE_NEWEST


# Mistral's model families, as served directly or behind an OpenAI-compatible
# gateway ("mistralai/mistral-small-3.2-24b-instruct",
# "mistral-small-3.2-24b-instruct-2506").
_MISTRAL_MODEL = re.compile(
    r"(?:^|[/._-])(?:mistral|magistral|ministral|devstral|codestral|pixtral)",
    re.IGNORECASE,
)
_STRICT_TOOL_ROLE_ORDER = replace(_LEGACY, user_after_tool=False)


def _takes_max_completion_tokens(model: str) -> bool:
    """Whether the id names an OpenAI reasoning model.

    Matches gpt-5 and gpt-6 with any `-` or `.` suffix (gpt-5.4-mini,
    gpt-6-astra) and the o1, o3 and o4 families. Anything else (gpt-4.x,
    gpt-4o, a proxy alias) is not recognized.
    """
    m = model.lower()
    for base in ("gpt-5", "gpt-6"):
        if m == base or m.startswith(base + "-") or m.startswith(base + "."):
            return True
    for prefix in ("o1", "o3", "o4"):
        if m == prefix or m.startswith(prefix + "-"):
            return True
    return False


def _with_max_tokens_param(row: ModelCapabilities, model: str) -> ModelCapabilities:
    if _takes_max_completion_tokens(model):
        return replace(row, max_tokens_param="max_completion_tokens")
    return row


def _row_for_model_id(model: str) -> ModelCapabilities:
    if _CLAUDE_MODEL.search(model):
        return anthropic_capabilities(model)
    if _MISTRAL_MODEL.search(model):
        return _STRICT_TOOL_ROLE_ORDER
    return _LEGACY


def model_capabilities(model: str) -> ModelCapabilities:
    """Capabilities inferred from a model id alone, whatever the transport.

    A Claude model gets its generation's row whether it is reached directly or
    through an OpenAI-compatible proxy.
    """
    return _with_max_tokens_param(_row_for_model_id(model), model)


def catalog_entry_capabilities(entry: ModelCatalogEntry) -> ModelCapabilities:
    """The profile a `MODELS` entry declares: inferred, then overridden."""
    profile = entry.capabilities
    inferred = model_capabilities(profile.infer_from or entry.model)
    return replace(
        inferred, **profile.model_dump(exclude={"infer_from"}, exclude_none=True)
    )


_catalog_index_cache: (
    tuple[Mapping[str, ModelCatalogEntry], dict[tuple[str, str], ModelCapabilities]]
    | None
) = None


def _catalog_capabilities() -> dict[tuple[str, str], ModelCapabilities]:
    """Declared profiles by (transport, model), rebuilt when the catalog changes."""
    global _catalog_index_cache
    catalog = config.settings.MODELS
    cached = _catalog_index_cache
    if cached is not None and cached[0] is catalog:
        return cached[1]
    index = {
        (entry.transport, entry.model): catalog_entry_capabilities(entry)
        for entry in catalog.values()
    }
    _catalog_index_cache = (catalog, index)
    return index


def capabilities_for(transport: ModelTransport, model: str) -> ModelCapabilities:
    """Capabilities of `model` on `transport`.

    A `MODELS` catalog entry for this transport and model decides. Otherwise
    the Anthropic transport has a row per Claude generation, and other
    transports get the legacy row, except Mistral models, which need strict
    role order after tool results. On any transport, an OpenAI reasoning
    model id takes `max_completion_tokens`.
    """
    declared = _catalog_capabilities().get((transport, model))
    if declared is not None:
        return declared
    if transport == "anthropic":
        row = anthropic_capabilities(model)
    elif _MISTRAL_MODEL.search(model):
        row = _STRICT_TOOL_ROLE_ORDER
    else:
        row = _LEGACY
    return _with_max_tokens_param(row, model)


__all__ = [
    "ForcedToolChoice",
    "MaxTokensParam",
    "ModelCapabilities",
    "ThinkingMode",
    "anthropic_capabilities",
    "capabilities_for",
    "catalog_entry_capabilities",
    "model_capabilities",
]
