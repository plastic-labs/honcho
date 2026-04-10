"""QueryMemoryTool — Honcho Dialectic API exposed as a Smolagents Tool."""

from smolagents import Tool

from .client import get_client


class QueryMemoryTool(Tool):
    """Query Honcho's Dialectic API to recall facts about the current user.

    Subclasses ``smolagents.Tool`` so the agent can call it when the user
    asks about their history, preferences, or past conversations.
    """

    name = "query_memory"
    description = (
        "Query Honcho's Dialectic API to recall facts about the current user. "
        "Use this when the user asks what you remember about them or their past conversations."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Natural language question about the user, e.g. 'What are my hobbies?'",
        }
    }
    output_type = "string"

    def __init__(self, user_id: str) -> None:
        super().__init__()
        self.user_id = user_id

    def forward(self, query: str) -> str:
        """Execute the memory query.

        Args:
            query: Natural language question about the user.

        Returns:
            A natural language answer from Honcho's memory.
        """
        honcho = get_client()
        peer = honcho.peer(self.user_id)
        response = peer.chat(query=query)
        return str(response) if response else "No relevant information found in memory."
