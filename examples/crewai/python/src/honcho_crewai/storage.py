"""
Honcho Storage for CrewAI External Memory

This module provides a Honcho-backed storage provider for CrewAI's external memory
system, enabling AI agents to maintain persistent conversation memory across sessions.
"""

import logging
import uuid
from typing import Any, Optional

from crewai.memory.storage.interface import Storage
from honcho import Honcho

logger = logging.getLogger(__name__)


class HonchoStorage(Storage):
    """
    Honcho-backed storage provider for CrewAI external memory.

    Implements CrewAI's Storage interface using Honcho's session-based memory,
    allowing agents to maintain context across conversations.

    Attributes:
        honcho: The Honcho client instance
        user: Peer representing the user
        assistant: Peer representing the AI assistant
        session: The conversation session
        session_id: Unique identifier for the session

    Example:
        ```python
        from honcho_crewai import HonchoStorage
        from crewai.memory.external.external_memory import ExternalMemory

        # Initialize storage
        storage = HonchoStorage(user_id="user123")

        # Use with CrewAI's external memory
        external_memory = ExternalMemory(storage=storage)
        ```
    """

    def __init__(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        honcho_client: Optional[Honcho] = None,
    ) -> None:
        """
        Initialize Honcho storage for a specific user and session.

        Args:
            user_id: Unique identifier for the user
            session_id: Optional session ID. If not provided, a UUID will be generated
            honcho_client: Optional Honcho client instance. If not provided, creates one
                using the demo environment (https://demo.honcho.dev)
        """
        self.honcho = honcho_client or Honcho()

        # Initialize user and assistant peers
        self.user = self.honcho.peer(user_id)
        self.assistant = self.honcho.peer("assistant")

        # Create or use existing session
        if not session_id:
            session_id = str(uuid.uuid4())
        self.session = self.honcho.session(session_id)
        self.session_id = session_id

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """
        Save a message to Honcho session.

        This method is called by CrewAI to store messages and context. Messages
        are associated with the appropriate peer (user or assistant) based on
        the metadata.

        Args:
            value: Message content to save
            metadata: Metadata dict that may contain 'role', 'agent', or 'type' info
                     Common keys: 'role', 'agent', 'type'
        """
        try:
            # Determine if this is from user or assistant based on metadata
            # Check various metadata keys that might indicate the role
            role = metadata.get("role", metadata.get("agent", "assistant"))
            is_user = role == "user"
            peer = self.user if is_user else self.assistant

            content_str = str(value)

            # Add message to session
            self.session.add_messages([peer.message(content_str, metadata=metadata)])

            logger.debug(
                "Saved message from %s: %s...",
                metadata.get("name", role),
                content_str[:100],
            )

        except Exception:
            logger.exception("Error saving to Honcho")
            raise

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant messages using semantic search.

        This method uses Honcho's semantic vector search to find messages most
        relevant to the query.

        Args:
            query: Search query used for semantic matching
            limit: Maximum number of messages to retrieve
            score_threshold: Minimum relevance score (not currently used by Honcho API)
            filters: Optional filters to scope the search. Supports Honcho's filter syntax
                including logical operators (AND, OR, NOT), comparison operators
                (gt, gte, lt, lte, eq, ne), and metadata filtering.
                Example: {"peer_id": "user123"} or {"metadata": {"type": "important"}}
                See: https://docs.honcho.dev/v3/documentation/core-concepts/features/using-filters

        Returns:
            List of message dictionaries in CrewAI expected format.
            Each dict contains:
                - content: The message content
                - memory: The message content (required by CrewAI)
                - context: The message content (for compatibility)
                - metadata: Message metadata including peer_id, created_at, and custom metadata
        """
        try:
            results = []
            # Use semantic search to find relevant messages
            # This performs vector similarity search on message content
            messages = self.session.search(query=query, filters=filters, limit=limit)

            # Convert to CrewAI expected format
            for msg in messages:
                # Build base metadata with peer_id and created_at
                metadata = {
                    "peer_id": msg.peer_id,
                    "created_at": str(msg.created_at)
                    if hasattr(msg, "created_at")
                    else None,
                }

                # Merge custom metadata if present
                if hasattr(msg, "metadata") and msg.metadata:
                    metadata.update(msg.metadata)

                results.append(
                    {
                        "content": msg.content,
                        "memory": msg.content,
                        "context": msg.content,
                        "metadata": metadata,
                    }
                )

            logger.debug("Search for '%s' returned %d results", query, len(results))
            return results

        except Exception:
            logger.exception("Error searching Honcho")
            raise

    def reset(self) -> None:
        """
        Create a new session, effectively resetting memory.

        This creates a new Honcho session with a fresh UUID, allowing the agent
        to start a new conversation without the previous context.
        """
        try:
            new_session_id = str(uuid.uuid4())
            self.session = self.honcho.session(new_session_id)
            self.session_id = new_session_id

            logger.debug("Reset session. New session ID: %s", new_session_id)

        except Exception:
            logger.exception("Error resetting Honcho session")
            raise
