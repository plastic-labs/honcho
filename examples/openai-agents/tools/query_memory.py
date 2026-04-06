"""Query Honcho memory via the Dialectic API — exposed as an agent tool."""

from agents import RunContextWrapper, function_tool

from .client import HonchoContext, get_client


@function_tool
def query_memory(ctx: RunContextWrapper[HonchoContext], query: str) -> str:
    """Query what Honcho knows about the current user using natural language.

    Sends a question to Honcho's Dialectic API and returns an answer grounded
    in the peer's long-term memory and stored observations. Use this tool when
    the user asks about their own history, preferences, or past conversations.

    Args:
        ctx: Run context carrying the ``HonchoContext`` with user identity.
        query: Natural language question, e.g. ``"What are my hobbies?"`` or
            ``"What did we discuss last time?"``.

    Returns:
        A natural language answer drawn from Honcho's memory, or a fallback
        message if no relevant information was found.

    Raises:
        ValueError: If query is empty.
    """
    if not query:
        raise ValueError("query must not be empty")

    honcho = get_client()
    peer = honcho.peer(ctx.context.user_id)
    response = peer.chat(query=query)

    if response:
        return str(response)
    return "No relevant information found in memory."
