"""Configuration resolution utilities for hierarchical settings."""

import logging

from src import models
from src.config import settings

logger = logging.getLogger(__name__)


def get_summary_config(
    session: models.Session, workspace: models.Workspace | None = None
) -> tuple[int, int]:
    """
    Resolve summary configuration with hierarchical fallback.

    Resolution hierarchy:
    1. Session configuration
    2. Workspace configuration
    3. Global defaults from settings

    Args:
        session: The session model
        workspace: Optional workspace model (if not provided, only session and global config are used)

    Returns:
        Tuple of (messages_per_short_summary, messages_per_long_summary)

    Raises:
        ValueError: If resolved values are invalid (negative, zero, or short > long)
    """
    # Default to global settings
    short_summary = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
    long_summary = settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY

    # Check workspace configuration
    if workspace is not None:
        workspace_short = workspace.configuration.get("messages_per_short_summary")
        workspace_long = workspace.configuration.get("messages_per_long_summary")

        if workspace_short is not None:
            short_summary = workspace_short
        if workspace_long is not None:
            long_summary = workspace_long

    # Check session configuration (takes precedence)
    session_short = session.configuration.get("messages_per_short_summary")
    session_long = session.configuration.get("messages_per_long_summary")

    if session_short is not None:
        short_summary = session_short
    if session_long is not None:
        long_summary = session_long

    # Validate resolved values
    if short_summary <= 0:
        raise ValueError(f"Invalid messages_per_short_summary: {short_summary}")
    if long_summary <= 0:
        raise ValueError(f"Invalid messages_per_long_summary: {long_summary}")

    return short_summary, long_summary
