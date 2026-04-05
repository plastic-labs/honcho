"""Save a conversation message to Honcho memory."""

from client import get_client


def save_memory(user_id: str, content: str, role: str, session_id: str) -> str:
    """Save a single conversation turn to Honcho memory.

    Creates the peer and session if they do not already exist. Registers
    the peer in the session on first use, then persists the message.

    Args:
        user_id: Unique identifier for the user peer.
        content: Text content of the message to save.
        role: Either "user" or "assistant". Determines which peer sends
            the message. Any value other than "assistant" is treated as "user".
        session_id: Identifier for the conversation session.

    Returns:
        A confirmation string describing what was saved.

    Raises:
        ValueError: If content is empty.
    """
    if not content:
        raise ValueError("content must not be empty")

    honcho = get_client()
    user_peer = honcho.peer(user_id)
    assistant_peer = honcho.peer("assistant")
    session = honcho.session(session_id)

    session.add_peers([user_peer, assistant_peer])

    sender = assistant_peer if role == "assistant" else user_peer
    session.add_messages([sender.message(content)])

    return f"Saved {role} message to session '{session_id}' for user '{user_id}'."
