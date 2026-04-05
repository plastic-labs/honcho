"""Honcho client initialization for Zo Computer skill."""

import os

from dotenv import load_dotenv
from honcho import Honcho

load_dotenv()


def get_client(workspace_id: str | None = None) -> Honcho:
    """Initialize and return a Honcho client.

    Reads HONCHO_API_KEY and HONCHO_WORKSPACE_ID from environment variables.
    The workspace_id parameter overrides the environment variable if provided.

    Args:
        workspace_id: Optional workspace ID override. Falls back to the
            HONCHO_WORKSPACE_ID env var, then to "default".

    Returns:
        Configured Honcho client instance.
    """
    resolved_workspace = workspace_id or os.getenv("HONCHO_WORKSPACE_ID", "default")
    return Honcho(workspace_id=resolved_workspace)
