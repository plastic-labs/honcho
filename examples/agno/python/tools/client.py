"""Honcho client initialization and context for Agno integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from honcho import Honcho

load_dotenv()


@dataclass
class HonchoContext:
    """Holds Honcho identity for a single conversation turn.

    Attributes:
        user_id: Unique identifier for the human peer.
        session_id: Identifier for the current conversation session.
        assistant_id: Peer ID for the assistant. Defaults to ``"assistant"``.
    """

    user_id: str
    session_id: str
    assistant_id: str = field(default="assistant")


def get_client(workspace_id: str | None = None) -> Honcho:
    """Initialize and return a Honcho client.

    Args:
        workspace_id: Optional workspace ID override. Falls back to
            ``HONCHO_WORKSPACE_ID`` env var, then to ``"default"``.

    Returns:
        Configured Honcho client instance.

    Raises:
        ValueError: If ``HONCHO_API_KEY`` is not set.
    """
    api_key = os.getenv("HONCHO_API_KEY")
    if not api_key:
        raise ValueError(
            "HONCHO_API_KEY is required. Set it in your environment or .env file."
        )

    env_workspace = os.getenv("HONCHO_WORKSPACE_ID")
    resolved_workspace = workspace_id or env_workspace or "default"
    return Honcho(api_key=api_key, workspace_id=resolved_workspace)
