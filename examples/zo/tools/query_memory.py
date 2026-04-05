"""Query a user's Honcho memory using the Dialectic API."""

from __future__ import annotations

from setup import get_client


def query_memory(user_id: str, query: str, session_id: str | None = None) -> str:
    """Query stored memory for a user using Honcho's Dialectic API.

    Sends a natural language question to Honcho and returns an answer
    grounded in the peer's long-term representation and stored observations.

    Args:
        user_id: Unique identifier for the user peer.
        query: Natural language question, e.g. "What are my hobbies?".
        session_id: Optional session ID to scope the query to a specific
            conversation. If omitted, the query draws from global memory.

    Returns:
        A natural language answer from Honcho's Dialectic API, or a
        default message if no relevant information was found.

    Raises:
        ValueError: If query is empty.
    """
    if not query:
        raise ValueError("query must not be empty")

    honcho = get_client()
    peer = honcho.peer(user_id)

    response = peer.chat(query=query, session=session_id)

    if response:
        return str(response)
    return "No relevant information found in memory."
