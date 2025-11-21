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
        "deriver": {"enabled": settings.DERIVER.ENABLED},
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
        "agentic_ingestion": {"enabled": settings.AGENTIC_INGESTION.ENABLED},
    }

    # Apply overrides in order (Workspace -> Session -> Message)
    # Note: deep_update modifies config_dict in place

    if workspace is not None:
        deep_update(config_dict, workspace.configuration)

    deep_update(config_dict, session.configuration)

    if message_configuration is not None:
        deep_update(config_dict, message_configuration.model_dump(exclude_none=True))

    return ResolvedConfiguration(**config_dict)
