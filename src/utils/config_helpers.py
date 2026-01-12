"""Configuration resolution utilities for hierarchical settings."""

import logging
from typing import Any, cast

from src import models
from src.config import settings
from src.schemas import (
    MessageConfiguration,
    ResolvedConfiguration,
)

logger = logging.getLogger(__name__)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """
    Recursive update of a dictionary.
    Skips None values in the update dictionary.
    """
    for key, value in update.items():
        if value is None:
            continue

        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(cast(dict[str, Any], base[key]), cast(dict[str, Any], value))
        else:
            base[key] = value


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

    reasoning_raw = normalized.get("reasoning")
    reasoning: dict[str, Any] = (
        cast(dict[str, Any], reasoning_raw) if isinstance(reasoning_raw, dict) else {}
    )
    reasoning_enabled_explicit = reasoning.get("enabled") is not None

    if not reasoning_enabled_explicit:
        deriver_raw = normalized.get("deriver")
        deriver = (
            cast(dict[str, Any], deriver_raw) if isinstance(deriver_raw, dict) else {}
        )
        if deriver.get("enabled") is not None:
            reasoning["enabled"] = bool(deriver["enabled"])

    if not reasoning_enabled_explicit and normalized.get("skip_deriver") is True:
        reasoning["enabled"] = False

    if reasoning:
        normalized["reasoning"] = reasoning

    normalized.pop("deriver", None)
    normalized.pop("skip_deriver", None)

    return normalized


def get_configuration(
    message_configuration: MessageConfiguration | None,
    session: models.Session,
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
        session: The session model
        workspace: Optional workspace model (if not provided, only session and global config are used)

    Returns:
        ResolvedConfiguration
    """
    # Start with defaults
    config_dict: dict[str, Any] = {
        "reasoning": {"enabled": settings.DERIVER.ENABLED},
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
    }

    # Apply overrides in order (Workspace -> Session -> Message)
    # Note: deep_update modifies config_dict in place

    if workspace is not None:
        deep_update(config_dict, normalize_configuration_dict(workspace.configuration))

    deep_update(config_dict, normalize_configuration_dict(session.configuration))

    if message_configuration is not None:
        deep_update(
            config_dict,
            normalize_configuration_dict(
                message_configuration.model_dump(exclude_none=True)
            ),
        )

    return ResolvedConfiguration(**config_dict)
