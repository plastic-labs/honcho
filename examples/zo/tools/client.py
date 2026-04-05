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
    api_key = os.getenv("HONCHO_API_KEY")
    if not api_key:
        raise ValueError(
            "HONCHO_API_KEY is required. Set it in your environment or .env file."
        )

    env_workspace = os.getenv("HONCHO_WORKSPACE_ID")
    resolved_workspace = workspace_id or env_workspace or "default"
    return Honcho(api_key=api_key, workspace_id=resolved_workspace)
