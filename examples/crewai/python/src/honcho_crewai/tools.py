"""
Honcho Tools for CrewAI

This module provides tools that allow CrewAI agents to interact with Honcho's
session context, dialectic API, and semantic search capabilities.
"""

import logging
from typing import Any, Optional

from crewai.tools import BaseTool
from honcho import Honcho
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


# Input Schemas
class GetContextInput(BaseModel):
    """Input schema for get_context tool."""

    tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of tokens to include in the context"
    )
    peer_target: Optional[str] = Field(
        default=None, description="A peer ID to get context for (retrieves representation and peer card)"
    )
    summary: bool = Field(
        default=True, description="Whether to include session summary in the context"
    )
    peer_perspective: Optional[str] = Field(
        default=None, description="Peer ID to use as the perspective for context retrieval"
    )


class DialecticInput(BaseModel):
    """Input schema for dialectic (chat) tool."""

    query: str = Field(..., min_length=1, description="Natural language question to ask")
    target: Optional[str] = Field(
        default=None, description="Optional target peer for local representation query"
    )
    session_id: Optional[str] = Field(
        default=None, description="Optional session ID to scope query to specific session"
    )


class SearchInput(BaseModel):
    """Input schema for search tool."""

    query: str = Field(..., min_length=1, description="Search query for semantic matching")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results to return (1-100)")
    filters: Optional[dict[str, Any]] = Field(
        default=None, description="Optional filters to apply to search results"
    )


# Tool Implementations
class HonchoGetContextTool(BaseTool):
    """
    Tool to retrieve session context with token limits.

    This tool fetches the conversation history and session summary within
    a specified token budget, optimized for LLM context windows.
    """

    name: str = "get_session_context"
    description: str = (
        "Retrieve recent conversation context within token limits. "
        "Returns formatted messages with optional summary and peer information. "
        "Useful for getting optimized context that fits within token budgets."
    )
    args_schema: type[BaseModel] = GetContextInput

    _honcho: Honcho = PrivateAttr()
    _session_id: str = PrivateAttr()
    _peer_id: str = PrivateAttr()

    def __init__(self, honcho: Honcho, session_id: str, peer_id: str):
        """
        Initialize the get_context tool.

        Args:
            honcho: Honcho client instance
            session_id: ID of the session to get context from
            peer_id: ID of the peer requesting context
        """
        super().__init__()
        self._honcho = honcho
        self._session_id = session_id
        self._peer_id = peer_id

    def _run(
        self,
        tokens: Optional[int] = None,
        peer_target: Optional[str] = None,
        summary: bool = True,
        peer_perspective: Optional[str] = None,
    ) -> str:
        """
        Execute get_context and format results.

        Args:
            tokens: Maximum tokens to include
            peer_target: Target peer ID for representation
            summary: Whether to include summary
            peer_perspective: Peer ID to use as perspective

        Returns:
            Formatted string containing context information
        """
        try:
            session = self._honcho.session(self._session_id)
            context = session.get_context(
                summary=summary,
                tokens=tokens,
                peer_target=peer_target,
                peer_perspective=peer_perspective,
            )

            # Format for agent consumption
            result = []

            # Add summary if present
            if context.summary:
                result.append(f"=== Session Summary ===")
                result.append(context.summary.content)
                result.append("")

            # Add peer representation if present
            if context.peer_representation:
                result.append(f"=== Peer Representation ===")
                result.append(context.peer_representation)
                result.append("")

            # Add peer card if present
            if context.peer_card:
                result.append(f"=== Peer Card ===")
                result.extend(context.peer_card)
                result.append("")

            # Add messages
            if context.messages:
                result.append(f"=== Messages ({len(context.messages)}) ===")
                for msg in context.messages:
                    result.append(f"{msg.peer_id}: {msg.content}")

            return "\n".join(result) if result else "No context available"

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"Error retrieving context: {str(e)}"


class HonchoDialecticTool(BaseTool):
    """
    Tool to query Honcho's dialectic API (peer representations).

    This tool allows agents to ask questions about what the system knows
    about users or other peers, leveraging Honcho's reasoning capabilities.
    """

    name: str = "query_peer_knowledge"
    description: str = (
        "Query the system's representation about peers. "
        "Ask questions like 'What does the user like?' or 'What are their preferences?' "
        "to retrieve information from the peer's long-term representation. "
        "Can optionally query what one peer knows about another (local representation)."
    )
    args_schema: type[BaseModel] = DialecticInput

    _honcho: Honcho = PrivateAttr()
    _session_id: str = PrivateAttr()
    _peer_id: str = PrivateAttr()

    def __init__(self, honcho: Honcho, session_id: str, peer_id: str):
        """
        Initialize the dialectic tool.

        Args:
            honcho: Honcho client instance
            session_id: Default session ID for scoped queries
            peer_id: ID of the peer to query about
        """
        super().__init__()
        self._honcho = honcho
        self._session_id = session_id
        self._peer_id = peer_id

    def _run(
        self,
        query: str,
        target: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Execute dialectic query.

        Args:
            query: Natural language question to ask
            target: Optional target peer for local representation
            session_id: Optional session ID to scope the query

        Returns:
            String response from the dialectic API
        """
        try:
            peer = self._honcho.peer(self._peer_id)

            # Use provided session_id or fall back to default
            scope_session_id = session_id or self._session_id

            # Query the dialectic API (non-streaming)
            response = peer.chat(
                query=query,
                stream=False,
                target=target,
                session_id=scope_session_id,
            )

            # Return the response or a default message
            if response:
                return str(response)
            else:
                return "No relevant information found."

        except Exception as e:
            logger.error(f"Error querying dialectic API: {e}")
            return f"Error querying peer knowledge: {str(e)}"


class HonchoSearchTool(BaseTool):
    """
    Tool to perform semantic search across session messages.

    This tool enables agents to find relevant past messages using
    semantic similarity search, useful for retrieving specific information
    from conversation history.
    """

    name: str = "search_session_messages"
    description: str = (
        "Search through session messages using semantic similarity. "
        "Finds messages that are semantically related to the query, "
        "useful for retrieving specific information from past conversations."
    )
    args_schema: type[BaseModel] = SearchInput

    _honcho: Honcho = PrivateAttr()
    _session_id: str = PrivateAttr()

    def __init__(self, honcho: Honcho, session_id: str):
        """
        Initialize the search tool.

        Args:
            honcho: Honcho client instance
            session_id: ID of the session to search in
        """
        super().__init__()
        self._honcho = honcho
        self._session_id = session_id

    def _run(self, query: str, limit: int = 10, filters: Optional[dict[str, Any]] = None) -> str:
        """
        Execute semantic search.

        Args:
            query: Search query for semantic matching
            limit: Number of results to return (1-100)
            filters: Optional filters to apply to search results

        Returns:
            Formatted string with search results
        """
        try:
            session = self._honcho.session(self._session_id)

            # Perform semantic search
            messages = session.search(query=query, limit=limit, filters=filters)

            if not messages:
                return f"No messages found matching '{query}'"

            # Format results for agent consumption
            result = [f"=== Search Results for '{query}' ({len(messages)} found) ==="]
            for i, msg in enumerate(messages, 1):
                result.append(f"\n{i}. [{msg.peer_id}] {msg.content}")
                if hasattr(msg, "created_at") and msg.created_at:
                    result.append(f"   Created: {msg.created_at}")

            return "\n".join(result)

        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return f"Error searching messages: {str(e)}"


# Factory Functions
def create_get_context_tool(
    honcho: Honcho,
    session_id: str,
    peer_id: str,
) -> HonchoGetContextTool:
    """
    Create a get_context tool instance.

    Args:
        honcho: Honcho client instance
        session_id: ID of the session to get context from
        peer_id: ID of the peer requesting context

    Returns:
        Configured HonchoGetContextTool instance

    Example:
        ```python
        from honcho import Honcho
        from honcho_crewai import create_get_context_tool

        honcho = Honcho()
        tool = create_get_context_tool(
            honcho=honcho,
            session_id="session_123",
            peer_id="user_456"
        )
        ```
    """
    return HonchoGetContextTool(
        honcho=honcho,
        session_id=session_id,
        peer_id=peer_id,
    )


def create_dialectic_tool(
    honcho: Honcho,
    session_id: str,
    peer_id: str,
) -> HonchoDialecticTool:
    """
    Create a dialectic tool instance.

    Args:
        honcho: Honcho client instance
        session_id: Default session ID for scoped queries
        peer_id: ID of the peer to query about

    Returns:
        Configured HonchoDialecticTool instance

    Example:
        ```python
        from honcho import Honcho
        from honcho_crewai import create_dialectic_tool

        honcho = Honcho()
        tool = create_dialectic_tool(
            honcho=honcho,
            session_id="session_123",
            peer_id="user_456"
        )
        ```
    """
    return HonchoDialecticTool(
        honcho=honcho,
        session_id=session_id,
        peer_id=peer_id,
    )


def create_search_tool(
    honcho: Honcho,
    session_id: str,
) -> HonchoSearchTool:
    """
    Create a search tool instance.

    Args:
        honcho: Honcho client instance
        session_id: ID of the session to search in

    Returns:
        Configured HonchoSearchTool instance

    Example:
        ```python
        from honcho import Honcho
        from honcho_crewai import create_search_tool

        honcho = Honcho()
        tool = create_search_tool(
            honcho=honcho,
            session_id="session_123"
        )
        ```
    """
    return HonchoSearchTool(
        honcho=honcho,
        session_id=session_id,
    )
