"""Query Honcho memory via the Dialectic API — exposed as a Google ADK tool."""

from .client import get_client


def query_memory(user_id: str, query: str) -> str:
    """Query what Honcho knows about a user using natural language.

    Sends a question to Honcho's Dialectic API and returns an answer grounded
    in the peer's long-term memory and stored observations.

    Args:
        user_id: The user peer ID to query memory for.
        query: Natural language question, e.g. ``"What are my hobbies?"``

    Returns:
        A natural language answer drawn from Honcho's memory, or a fallback
        message if no relevant information was found.

    Raises:
        ValueError: If query is empty or whitespace-only.
        RuntimeError: If the Dialectic API call fails.
    """
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("query must not be empty or whitespace")

    try:
        honcho = get_client()
        peer = honcho.peer(user_id)
        response = peer.chat(query=trimmed)
    except Exception as exc:
        raise RuntimeError("Failed to query Honcho memory") from exc

    if response:
        return str(response)
    return "No relevant information found in memory."
