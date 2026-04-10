"""Query Honcho memory via the Dialectic API — exposed as a LlamaIndex FunctionTool."""

from llama_index.core.tools import FunctionTool

from .client import HonchoContext, get_client


def make_query_memory_tool(ctx: HonchoContext) -> FunctionTool:
    """Create a LlamaIndex FunctionTool bound to the current user context.

    Args:
        ctx: ``HonchoContext`` holding user and session identifiers.

    Returns:
        A ``FunctionTool`` the agent can call to query Honcho memory.
    """

    def query_memory(query: str) -> str:
        """Query Honcho's Dialectic API to recall facts about the current user.

        Use this when the user asks what you remember about them or their
        past conversations.

        Args:
            query: Natural language question about the user.

        Returns:
            A natural language answer from Honcho's memory.
        """
        if not query:
            raise ValueError("query must not be empty")

        honcho = get_client()
        peer = honcho.peer(ctx.user_id)
        response = peer.chat(query=query)
        return str(response) if response else "No relevant information found in memory."

    return FunctionTool.from_defaults(fn=query_memory)
