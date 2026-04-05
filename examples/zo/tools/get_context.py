"""Retrieve conversation context from Honcho formatted for LLM use."""

from __future__ import annotations

from client import get_client


def get_context(
    user_id: str,
    session_id: str,
    assistant_id: str,
    tokens: int = 4000,
) -> list[dict[str, str]]:
    """Retrieve conversation context ready for injection into an LLM prompt.

    Fetches recent messages from a Honcho session within the given token
    budget and converts them to OpenAI-compatible message format. Use the
    returned list directly as the ``messages`` parameter in an LLM API call.

    Args:
        user_id: Unique identifier for the user peer. Used to ensure the
            peer is registered in the session before fetching context.
        session_id: Identifier for the conversation session.
        assistant_id: Peer ID representing the assistant. This determines
            which role is mapped to ``"assistant"`` in the output.
        tokens: Maximum number of tokens to include in the context window.
            Defaults to 4000.

    Returns:
        A list of message dicts in OpenAI format:
        ``[{"role": "user" | "assistant", "content": "..."}]``.
        Returns an empty list if the session has no messages.
    """
    honcho = get_client()
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer(assistant_id)
    session = honcho.session(session_id)

    session.add_peers([user_peer, assistant_peer])

    context = session.context(tokens=tokens)
    return context.to_openai(assistant=assistant_id)
