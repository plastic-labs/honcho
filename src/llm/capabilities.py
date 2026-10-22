"""Per-model request capabilities.

Model generations differ in which request shapes they accept: how thinking is
configured, whether an assistant prefill is allowed, whether a forced
tool_choice is accepted, and whether replayed thinking blocks are bound to the
conversation that produced them. Backends and the tool loop read those
differences from here instead of matching model names inline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.config import ModelTransport

ThinkingMode = Literal["budget", "adaptive"]

# How a model treats a forced tool_choice ({"type": "any"} / {"type": "tool"}):
# - "conflicts_with_thinking": accepted, but not alongside extended thinking
# - "skips_thinking": accepted with thinking on; that turn simply skips thinking
# - "rejected": a 400 whether or not thinking is on
ForcedToolChoice = Literal["conflicts_with_thinking", "skips_thinking", "rejected"]


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


def capabilities_for(transport: ModelTransport, model: str) -> ModelCapabilities:
    """Capabilities of `model` on `transport`.

    Only the Anthropic transport has model-dependent rows so far; the other
    transports get the legacy row, which none of their backends consult.
    """
    if transport == "anthropic":
        return anthropic_capabilities(model)
    return _LEGACY


__all__ = [
    "ForcedToolChoice",
    "ModelCapabilities",
    "ThinkingMode",
    "anthropic_capabilities",
    "capabilities_for",
]
