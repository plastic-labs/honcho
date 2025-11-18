"""Configuration resolution utilities for hierarchical settings."""

import logging
from typing import Any

from src import models
from src.config import settings
from src.schemas import (
    MessageConfiguration,
    ResolvedConfiguration,
    ResolvedDeriverConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedSummaryConfiguration,
)

logger = logging.getLogger(__name__)


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
    updates: dict[str, Any] = {}

    # Add workspace configuration (only non-None values)
    if workspace is not None:
        updates.update(
            {k: v for k, v in workspace.configuration.items() if v is not None}
        )

    # Add session configuration (overwrites workspace, only non-None values)
    updates.update({k: v for k, v in session.configuration.items() if v is not None})

    # Add message configuration (overwrites all previous configurations, only non-None values)
    if message_configuration is not None:
        updates.update(message_configuration.model_dump(exclude_none=True))

    return ResolvedConfiguration(
        ## DEFAULTS / GLOBAL CONFIG
        deriver=ResolvedDeriverConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(
            use=settings.PEER_CARD.ENABLED, create=settings.PEER_CARD.ENABLED
        ),
        summary=ResolvedSummaryConfiguration(
            enabled=settings.SUMMARY.ENABLED,
            messages_per_short_summary=settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY,
            messages_per_long_summary=settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY,
        ),
        dream=ResolvedDreamConfiguration(enabled=settings.DREAM.ENABLED),
        ## Others will override
        **updates,
    )
