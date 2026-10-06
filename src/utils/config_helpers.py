"""Configuration resolution utilities for hierarchical settings."""

import logging
from typing import Any

from src import models
from src.config import settings
from src.schemas import (
    MessageConfiguration,
    ResolvedConfiguration,
)
from src.utils.json_coerce import as_dict

logger = logging.getLogger(__name__)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """
    Recursive update of a dictionary.
    Skips None values in the update dictionary.
    """
    for key, value in update.items():
        if value is None:
            continue

        nested = as_dict(value)
        existing = as_dict(base.get(key))
        if nested is not None and existing is not None:
            deep_update(existing, nested)
        else:
            base[key] = value


# Modules whose custom_instructions fall back to the top-level shared value.
# peer_card is resolved separately: its instructions are additive on top of the
# dream instructions, so falling back would inject the shared value twice.
_SHARED_INSTRUCTION_MODULES = ("reasoning", "summary", "dialectic", "dream")


def resolve_custom_instructions(
    levels: list[tuple[str, dict[str, Any]]],
    module: str,
    *,
    shared_fallback: bool = True,
) -> tuple[str | None, str | None]:
    """
    Resolve a module's custom instructions across configuration levels.

    The most specific level that sets a value wins. Within one level, the
    module's own `custom_instructions` beats the top-level shared value. An
    empty string is an explicit "none" and stops the fallback.

    Args:
        levels: (level name, normalized configuration dict), most specific first
        module: Configuration section name, e.g. "summary"
        shared_fallback: Whether to fall back to the top-level value

    Returns:
        (stripped instructions or None, source such as "session.shared" or
        "workspace.summary", or None when no level set a value)
    """
    for level_name, level in levels:
        value = (as_dict(level.get(module)) or {}).get("custom_instructions")
        source = f"{level_name}.{module}"
        if value is None and shared_fallback:
            value = level.get("custom_instructions")
            source = f"{level_name}.shared"
        if isinstance(value, str):
            return value.strip() or None, source
    return None, None


def normalize_configuration_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a workspace/session/message configuration dict to match the current
    `ResolvedConfiguration` schema.

    This function exists to preserve backwards compatibility with older clients/tests
    that used legacy configuration keys (e.g. `deriver.enabled` or `skip_deriver`).

    Behavior:
    - If `reasoning.enabled` is not explicitly set, but `deriver.enabled` is present,
      `reasoning.enabled` is derived from `deriver.enabled`.
    - If `skip_deriver` is explicitly `True`, `reasoning.enabled` is forced to `False`
      unless `reasoning.enabled` was already explicitly set.
    - Legacy keys are removed from the returned dict to avoid polluting the resolved
      configuration with unused fields.
    """
    normalized: dict[str, Any] = dict(raw)

    reasoning_present = "reasoning" in normalized
    reasoning: dict[str, Any] = dict(as_dict(normalized.get("reasoning")) or {})
    reasoning_enabled_explicit = reasoning.get("enabled") is not None

    if not reasoning_enabled_explicit:
        deriver: dict[str, Any] = dict(as_dict(normalized.get("deriver")) or {})
        if deriver.get("enabled") is not None:
            reasoning["enabled"] = bool(deriver["enabled"])

    if not reasoning_enabled_explicit and normalized.get("skip_deriver") is True:
        reasoning["enabled"] = False

    if reasoning_present or reasoning:
        normalized["reasoning"] = reasoning

    normalized.pop("deriver", None)
    normalized.pop("skip_deriver", None)

    return normalized


def get_configuration(
    message_configuration: MessageConfiguration | None,
    session: models.Session | None,
    workspace: models.Workspace | None = None,
) -> ResolvedConfiguration:
    """
    Resolve session configuration with hierarchical fallback.

    Resolution hierarchy:
    1. Message configuration
    2. Session configuration
    3. Workspace configuration
    4. Global defaults from settings

    Args:
        message_configuration: Optional message configuration
        session: Optional session model
        workspace: Optional workspace model

    Returns:
        ResolvedConfiguration
    """
    # Start with defaults
    config_dict: dict[str, Any] = {
        "reasoning": {
            "enabled": settings.DERIVER.ENABLED,
            "custom_instructions": None,
        },
        "peer_card": {
            "use": settings.PEER_CARD.ENABLED,
            "create": settings.PEER_CARD.ENABLED,
        },
        "summary": {
            "enabled": settings.SUMMARY.ENABLED,
            "messages_per_short_summary": settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY,
            "messages_per_long_summary": settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY,
        },
        "dream": {"enabled": settings.DREAM.ENABLED},
        "dialectic": {},
    }

    # Most specific first: Message -> Session -> Workspace
    levels = [
        (name, normalize_configuration_dict(raw))
        for name, raw in (
            (
                "message",
                message_configuration.model_dump(exclude_none=True)
                if message_configuration is not None
                else None,
            ),
            ("session", session.configuration if session is not None else None),
            ("workspace", workspace.configuration if workspace is not None else None),
        )
        if raw is not None
    ]

    # Apply overrides least specific first; deep_update modifies config_dict in place
    for _, level in reversed(levels):
        deep_update(config_dict, level)

    # custom_instructions don't merge field-by-field: a more specific level's
    # shared value must beat a less specific level's module value.
    for module in (*_SHARED_INSTRUCTION_MODULES, "peer_card"):
        value, source = resolve_custom_instructions(
            levels, module, shared_fallback=module != "peer_card"
        )
        config_dict[module]["custom_instructions"] = value
        config_dict[module]["custom_instructions_source"] = source

    return ResolvedConfiguration(**config_dict)
