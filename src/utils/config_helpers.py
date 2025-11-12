"""Configuration resolution utilities for hierarchical settings."""

import logging
from typing import Any

from src import models
from src.config import settings
from src.schemas import ResolvedSessionConfiguration

logger = logging.getLogger(__name__)


def get_configuration(
    session: models.Session, workspace: models.Workspace | None = None
) -> ResolvedSessionConfiguration:
    """
    Resolve session configuration with hierarchical fallback.

    Resolution hierarchy:
    1. Session configuration
    2. Workspace configuration
    3. Global defaults from settings

    Args:
        session: The session model
        workspace: Optional workspace model (if not provided, only session and global config are used)

    Returns:
        SessionConfiguration
    """
    updates: dict[str, Any] = {}

    # Add workspace configuration (only non-None values)
    if workspace is not None:
        updates.update(
            {k: v for k, v in workspace.configuration.items() if v is not None}
        )

    # Add session configuration (overwrites workspace, only non-None values)
    updates.update({k: v for k, v in session.configuration.items() if v is not None})

    return ResolvedSessionConfiguration(
        deriver_enabled=True,
        peer_cards_enabled=settings.PEER_CARD.ENABLED,
        summaries_enabled=settings.SUMMARY.ENABLED,
        dreams_enabled=settings.DREAM.ENABLED,
        messages_per_short_summary=settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY,
        messages_per_long_summary=settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY,
        **updates,
    )
