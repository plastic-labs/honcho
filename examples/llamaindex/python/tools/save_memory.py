"""Save a conversation message to Honcho memory."""

from .client import get_client

_VALID_ROLES = {"user", "assistant"}


def save_memory(
    user_id: str,
    content: str,
    role: str,
    session_id: str,
    assistant_id: str = "assistant",
) -> str:
    """Save a single conversation turn to Honcho memory.

    Args:
        user_id: Unique identifier for the user peer.
        content: Text content of the message to save.
        role: Either ``"user"`` or ``"assistant"``.
        session_id: Identifier for the conversation session.
        assistant_id: Peer ID for the assistant. Defaults to ``"assistant"``.

    Returns:
        A confirmation string describing what was saved.
    """
    if not content:
        raise ValueError("content must not be empty")
    if role not in _VALID_ROLES:
        raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

    honcho = get_client()
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer(assistant_id)
    session = honcho.session(session_id)

    session.add_peers([user_peer, assistant_peer])

    sender = assistant_peer if role == "assistant" else user_peer
    session.add_messages([sender.message(content)])

    return f"Saved {role} message to session '{session_id}' for user '{user_id}'."
