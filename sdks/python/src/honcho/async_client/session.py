from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    Configuration,
    DeriverStatus,
    Message,
    MessageCreateParam,
    SessionCore,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHttpClient, AsyncPage
from ..session_context import SessionContext, SessionSummaries, Summary
from ..utils import prepare_file_for_upload

if TYPE_CHECKING:
    from ..types import Representation
    from .peer import AsyncPeer

logger = logging.getLogger(__name__)


class SessionPeerConfig(BaseModel):
    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )


class AsyncSession(SessionBase):
    """
    Represents a session in Honcho with async operations.

    Sessions are scoped to a set of peers and contain messages/content.
    They create bidirectional relationships between peers and provide
    a context for multi-party conversations and interactions.

    Attributes:
        id: Unique identifier for this session
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this session. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this session. May be stale if not
            recently fetched. Call get_config() for fresh data.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHttpClient = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this session. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this session. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        session_id: str = Field(
            ..., min_length=1, description="Unique identifier for this session"
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        http: AsyncHttpClient = Field(
            ..., description="Reference to the HTTP client instance"
        ),
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
        """
        Initialize a new AsyncSession.

        Args:
            session_id: Unique identifier for this session within the workspace
            workspace_id: Workspace ID for scoping operations
            http: Reference to the HTTP client instance
            metadata: Optional metadata to initialize the cached value
            config: Optional configuration to initialize the cached value
        """
        super().__init__(
            id=session_id,
            workspace_id=workspace_id,
        )
        self._http = http
        self._metadata = metadata
        self._configuration = config

    @classmethod
    async def create(
        cls,
        session_id: str,
        workspace_id: str,
        http: AsyncHttpClient,
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> AsyncSession:
        """
        Create a new AsyncSession with optional configuration.

        Provided metadata and configuration will overwrite any existing data in those
        locations if given.

        Args:
            session_id: Unique identifier for this session within the workspace
            workspace_id: Workspace ID for scoping operations
            http: Reference to the HTTP client instance
            metadata: Optional metadata dictionary to associate with this session.
            If set, will get/create session immediately with metadata.
            config: Optional configuration to set for this session.
            If set, will get/create session immediately with flags.

        Returns:
            A new AsyncSession instance
        """
        instance = cls(session_id, workspace_id, http, metadata=metadata, config=config)

        if config is not None or metadata is not None:
            body: dict[str, Any] = {"id": session_id}
            if config is not None:
                body["configuration"] = config
            if metadata is not None:
                body["metadata"] = metadata

            response = await http.request(
                "POST",
                f"/v2/workspaces/{workspace_id}/sessions",
                json=body,
            )
            session_data = SessionCore.model_validate(response)
            instance._metadata = session_data.metadata
            instance._configuration = session_data.configuration

        return instance

    async def add_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]] = Field(
            ..., description="Peers to add to the session"
        ),
    ) -> None:
        """
        Add peers to this session.

        Makes an async API call to add one or more peers to this session. Adding peers
        creates bidirectional relationships and allows them to participate in
        the session's conversations.

        Args:
            peers: Peers to add to the session. Can be:
                - str: Single peer ID
                - AsyncPeer: Single AsyncPeer object
                - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
                - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                - tuple[AsyncPeer, SessionPeerConfig]: Single AsyncPeer object and SessionPeerConfig
                - List[tuple[Union[AsyncPeer, str], SessionPeerConfig]]: List of AsyncPeer objects and/or peer IDs and SessionPeerConfig
                - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/AsyncPeer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or AsyncPeer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/add",
            json=peer_dict,
        )

    async def set_peers(
        self,
        peers: str
        | PeerBase
        | tuple[str, SessionPeerConfig]
        | tuple[PeerBase, SessionPeerConfig]
        | list[PeerBase | str]
        | list[tuple[PeerBase | str, SessionPeerConfig]]
        | list[PeerBase | str | tuple[PeerBase | str, SessionPeerConfig]] = Field(
            ..., description="Peers to set for the session"
        ),
    ) -> None:
        """
        Set the complete peer list for this session.

        Makes an API call to replace the current peer list with the provided peers.
        This will remove any peers not in the new list and add any that are missing.

        Args:
            peers: Peers to set for the session. Can be:
                - str: Single peer ID
                - AsyncPeer: Single AsyncPeer object
                - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
                - tuple[str, SessionPeerConfig]: Single peer ID and SessionPeerConfig
                - tuple[AsyncPeer, SessionPeerConfig]: Single AsyncPeer object and SessionPeerConfig
                - List[tuple[Union[AsyncPeer, str], SessionPeerConfig]]: List of AsyncPeer objects and/or peer IDs and SessionPeerConfig
                - Mixed lists with peers and tuples/lists containing peer+config combinations
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                # Handle tuple[str/AsyncPeer, SessionPeerConfig]
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                # Handle direct str or AsyncPeer
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/set",
            json=peer_dict,
        )

    async def remove_peers(
        self,
        peers: str | PeerBase | list[PeerBase | str] = Field(
            ..., description="Peers to remove from the session"
        ),
    ) -> None:
        """
        Remove peers from this session.

        Makes an async API call to remove one or more peers from this session.
        Removed peers will no longer be able to participate in the session
        unless added back.

        Args:
            peers: Peers to remove from the session. Can be:
                   - str: Single peer ID
                   - AsyncPeer: Single AsyncPeer object
                   - List[Union[AsyncPeer, str]]: List of AsyncPeer objects and/or peer IDs
        """
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/remove",
            json=peer_ids,
        )

    async def get_peers(self) -> list[AsyncPeer]:
        """
        Get all peers in this session.

        Makes an async API call to retrieve the list of peer IDs that are currently
        members of this session. Automatically converts the paginated response
        into a list for us -- the max number of peers in a session is usually 10.

        Returns:
            A list of AsyncPeer objects that are members of this session
        """
        from .peer import AsyncPeer

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/list",
            json={},
        )
        items = response.get("items", []) if response else []
        return [AsyncPeer(peer["id"], self.workspace_id, self._http) for peer in items]

    async def get_peer_config(self, peer: str | PeerBase) -> SessionPeerConfig:
        """
        Get the configuration for a peer in this session.
        """
        peer_id = peer if isinstance(peer, str) else peer.id
        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/{peer_id}/config",
        )
        return SessionPeerConfig(
            observe_others=response.get("observe_others") if response else None,
            observe_me=response.get("observe_me") if response else None,
        )

    async def set_peer_config(
        self, peer: str | PeerBase, config: SessionPeerConfig
    ) -> None:
        """
        Set the configuration for a peer in this session.
        """
        peer_id = peer if isinstance(peer, str) else peer.id
        body: dict[str, Any] = {}
        if config.observe_others is not None:
            body["observe_others"] = config.observe_others
        if config.observe_me is not None:
            body["observe_me"] = config.observe_me

        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/peers/{peer_id}/config",
            json=body,
        )

    @validate_call
    async def add_messages(
        self,
        messages: MessageCreateParam
        | dict[str, Any]
        | list[MessageCreateParam | dict[str, Any]] = Field(
            ..., description="Messages to add to the session"
        ),
    ) -> list[Message]:
        """
        Add one or more messages to this session.

        Makes an API call to store messages in this session. Any message added
        to a session will automatically add the creating peer to the session
        if they are not already a member.

        Args:
            messages: Messages to add to the session. Can be:
                      - MessageCreateParam: Single MessageCreateParam object
                      - List[MessageCreateParam]: List of MessageCreateParam objects
        """
        if not isinstance(messages, list):
            messages = [messages]

        # Convert to dicts for the API
        message_dicts = []
        for msg in messages:
            if isinstance(msg, MessageCreateParam):
                message_dicts.append(msg.model_dump(exclude_none=True))
            else:
                message_dicts.append(msg)

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/messages",
            json={"messages": message_dicts},
        )
        return [Message.model_validate(m) for m in (response or [])]

    @validate_call
    async def get_messages(
        self,
        *,
        filters: dict[str, object] | None = Field(
            None, description="Dictionary of filter criteria"
        ),
    ) -> AsyncPage[dict[str, Any], Message]:
        """
        Get messages from this session with optional filtering.

        Makes an async API call to retrieve messages from this session. Results can be
        filtered based on various criteria.

        Args:
            filters: Dictionary of filter criteria. Supported filters include:
                    - peer_id: Filter messages by the peer who created them
                    - metadata: Filter messages by metadata key-value pairs
                    - timestamp_start: Filter messages after a specific timestamp
                    - timestamp_end: Filter messages before a specific timestamp

        Returns:
            An async paginated list of Message objects matching the specified criteria, ordered by
            creation time (most recent first)
        """

        async def fetch_page(
            page: int = 1, size: int = 50
        ) -> AsyncPage[dict[str, Any], Message]:
            response = await self._http.request(
                "POST",
                f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/messages/list",
                json={"filters": filters, "page": page, "size": size},
            )
            return AsyncPage(
                items=response.get("items", []),
                total=response.get("total"),
                page=response.get("page", page),
                size=response.get("size", size),
                pages=response.get("pages"),
                transform_func=lambda m: Message.model_validate(m),
                fetch_next=lambda: fetch_page(page + 1, size),
            )

        return await fetch_page()

    async def delete(self) -> None:
        """
        Delete this session and all associated data.

        Makes an async API call to permanently delete this session and all related data including:
        - Messages
        - Message embeddings
        - Observations
        - Session-Peer associations
        - Background processing queue items

        This action cannot be undone.
        """
        await self._http.request(
            "DELETE",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}",
        )

    async def clone(
        self,
        *,
        message_id: str | None = None,
    ) -> "AsyncSession":
        """
        Clone this session, optionally up to a specific message.

        Makes an async API call to create a copy of this session with a new ID.
        All messages and peers from the original session are copied to the new session.
        If a message_id is provided, only messages up to and including that message
        are copied.

        Args:
            message_id: Optional message ID to cut off the clone at. If provided,
                       the cloned session will only contain messages up to and
                       including this message.

        Returns:
            A new AsyncSession object representing the cloned session

        Example:
            ```python
            # Clone entire session
            cloned = await session.clone()

            # Clone session up to a specific message
            cloned = await session.clone(message_id="msg_abc123")
            ```
        """
        body: dict[str, Any] = {}
        if message_id is not None:
            body["message_id"] = message_id

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/clone",
            json=body,
        )
        cloned_session_data = SessionCore.model_validate(response)

        # Return a new AsyncSession object with the cloned session's data
        return AsyncSession(
            cloned_session_data.id,
            self.workspace_id,
            self._http,
            metadata=cloned_session_data.metadata,
            config=cloned_session_data.configuration,
        )

    async def get_metadata(self) -> dict[str, object]:
        """
        Get metadata for this session.

        Makes an async API call to retrieve the current metadata associated with this session.
        Metadata can include custom attributes, settings, or any other key-value data.
        This method also updates the cached metadata attribute.

        Returns:
            A dictionary containing the session's metadata. Returns an empty dictionary
            if no metadata is set
        """
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions",
            json={"id": self.id},
        )
        session_data = SessionCore.model_validate(response)
        self._metadata = session_data.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this session"
        ),
    ) -> None:
        """
        Set metadata for this session.

        Makes an async API call to update the metadata associated with this session.
        This will overwrite any existing metadata with the provided values.
        This method also updates the cached metadata attribute.

        Args:
            metadata: A dictionary of metadata to associate with this session.
                     Keys must be strings, values can be any JSON-serializable type
        """
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}",
            json={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """
        Get configuration for this session.

        Makes an async API call to retrieve the current configuration associated with this session.
        Configuration includes settings that control session behavior.
        This method also updates the cached configuration attribute.

        Returns:
            A dictionary containing the session's configuration. Returns an empty dictionary
            if no configuration is set
        """
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions",
            json={"id": self.id},
        )
        session_data = SessionCore.model_validate(response)
        self._configuration = session_data.configuration or {}
        return self._configuration

    @validate_call
    async def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary to associate with this session"
        ),
    ) -> None:
        """
        Set configuration for this session.

        Makes an async API call to update the configuration associated with this session.
        This will overwrite any existing configuration with the provided values.
        This method also updates the cached configuration attribute.

        Args:
            configuration: A dictionary of configuration to associate with this session.
                          Keys must be strings, values can be any JSON-serializable type
        """
        await self._http.request(
            "PUT",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}",
            json={"configuration": configuration},
        )
        self._configuration = configuration

    async def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for this session.

        Makes a single async API call to retrieve the latest metadata and configuration
        associated with this session and updates the cached attributes.
        """
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions",
            json={"id": self.id},
        )
        session_data = SessionCore.model_validate(response)
        self._metadata = session_data.metadata or {}
        self._configuration = session_data.configuration or {}

    @validate_call
    async def get_context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(
            None, gt=0, description="Maximum number of tokens to include in the context"
        ),
        peer_target: str | None = Field(
            None,
            description="A peer ID to get context for. If given *without* `peer_perspective`, a representation and peer card will be included from the omniscient Honcho-level view of `peer_target`. If given *with* `peer_perspective`, will get the representation and card for `peer_target` *from the perspective of `peer_perspective`*.",
        ),
        last_user_message: str | Message | None = Field(
            None,
            description="The most recent message (string or Message object), used to fetch semantically relevant observations and returned as part of the context object. Use this alongside `peer_target` to get a more focused context -- does nothing if `peer_target` is not provided.",
        ),
        peer_perspective: str | None = Field(
            None,
            description="A peer ID to get context *from the perspective of*. If given, response will attempt to include representation and card from the perspective of `peer_perspective`. Must be provided with `peer_target`.",
        ),
        limit_to_session: bool = Field(
            False,
            description="Whether to limit the representation to this session only. If True, only observations from this session will be included.",
        ),
        search_top_k: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Number of semantically relevant facts to return when searching with `last_user_message`.",
        ),
        search_max_distance: float | None = Field(
            None,
            ge=0.0,
            le=1.0,
            description="Maximum semantic distance for search results (0.0-1.0) when searching with `last_user_message`.",
        ),
        include_most_derived: bool | None = Field(
            None,
            description="Whether to include the most derived observations in the representation.",
        ),
        max_observations: int | None = Field(
            None,
            ge=1,
            le=100,
            description="Maximum number of observations to include in the representation.",
        ),
    ) -> SessionContext:
        """
        Get optimized context for this session within a token limit.

        Makes an API call to retrieve a curated list of messages that provides
        optimal context for the conversation while staying within the specified
        token limit. Uses tiktoken for token counting, so results should be
        compatible with OpenAI models.

        Args:
            summary: Whether to include summary information
            tokens: Maximum number of tokens to include in the context. Will default
            to Honcho server configuration if not provided.
            peer_target: A peer ID to get context for. If given *without* `peer_perspective`, a representation and peer card will be included from the omniscient Honcho-level view of `peer_target`. If given *with* `peer_perspective`, will get the representation and card for `peer_target` *from the perspective of `peer_perspective`*.
            last_user_message: The most recent message (string or Message object), used to fetch semantically relevant observations and returned as part of the context object. Use this alongside `peer_target` to get a more focused context -- does nothing if `peer_target` is not provided.
            peer_perspective: A peer ID to get context *from the perspective of*. If given, response will attempt to include representation and card from the perspective of `peer_perspective`. Must be provided with `peer_target`.
            limit_to_session: Whether to limit the representation to this session only. If True, only observations from this session will be included.
            search_top_k: Number of semantically relevant facts to return when searching with `last_user_message`.
            search_max_distance: Maximum semantic distance for search results (0.0-1.0) when searching with `last_user_message`.
            include_most_derived: Whether to include the most derived observations in the representation.
            max_observations: Maximum number of observations to include in the representation.

        Returns:
            A SessionContext object containing the optimized message history and
            summary, if available, that maximizes conversational context while
            respecting the token limit

        Note:
            Token counting is performed using tiktoken. For models using different
            tokenizers, you may need to adjust the token limit accordingly.
        """

        if peer_target is None and peer_perspective is not None:
            raise ValueError(
                "You must provide a `peer_target` when `peer_perspective` is provided"
            )

        if peer_target is None and last_user_message is not None:
            raise ValueError(
                "You must provide a `peer_target` when `last_user_message` is provided"
            )

        last_user_message_id = (
            last_user_message.id
            if isinstance(last_user_message, Message)
            else last_user_message
        )

        params: dict[str, Any] = {
            "summary": summary,
            "limit_to_session": limit_to_session,
        }
        if tokens is not None:
            params["tokens"] = tokens
        if last_user_message_id is not None:
            params["last_message"] = last_user_message_id
        if peer_target is not None:
            params["peer_target"] = peer_target
        if peer_perspective is not None:
            params["peer_perspective"] = peer_perspective
        if search_top_k is not None:
            params["search_top_k"] = search_top_k
        if search_max_distance is not None:
            params["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            params["include_most_derived"] = include_most_derived
        if max_observations is not None:
            params["max_observations"] = max_observations

        context = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/context",
            params=params,
        )

        # Convert the summary to our Summary if it exists
        session_summary = None
        if context and context.get("summary"):
            summary_data = context["summary"]
            session_summary = Summary(
                content=summary_data.get("content", ""),
                message_id=summary_data.get("message_id", ""),
                summary_type=summary_data.get("summary_type", ""),
                created_at=summary_data.get("created_at", ""),
                token_count=summary_data.get("token_count", 0),
            )

        messages = [
            Message.model_validate(m)
            for m in (context.get("messages", []) if context else [])
        ]

        return SessionContext(
            session_id=self.id,
            messages=messages,
            summary=session_summary,
            peer_representation=str(context.get("peer_representation"))
            if context and context.get("peer_representation")
            else None,
            peer_card=context.get("peer_card") if context else None,
        )

    async def get_summaries(self) -> SessionSummaries:
        """
        Get available summaries for this session.

        Makes an async API call to retrieve both short and long summaries for this session,
        if they are available. Summaries are created asynchronously by the backend
        as messages are added to the session.

        Returns:
            A SessionSummaries object containing:
            - id: The session ID
            - short_summary: The short summary if available, including metadata
            - long_summary: The long summary if available, including metadata

        Note:
            Summaries may be None if:
            - Not enough messages have been added to trigger summary generation
            - The summary generation is still in progress
            - Summary generation is disabled for this session
        """
        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/summaries",
        )

        # Create Summary objects from the response data
        short_summary = None
        if response and response.get("short_summary"):
            summary_data = response["short_summary"]
            short_summary = Summary(
                content=summary_data.get("content", ""),
                message_id=summary_data.get("message_id", ""),
                summary_type=summary_data.get("summary_type", ""),
                created_at=summary_data.get("created_at", ""),
                token_count=summary_data.get("token_count", 0),
            )

        long_summary = None
        if response and response.get("long_summary"):
            summary_data = response["long_summary"]
            long_summary = Summary(
                content=summary_data.get("content", ""),
                message_id=summary_data.get("message_id", ""),
                summary_type=summary_data.get("summary_type", ""),
                created_at=summary_data.get("created_at", ""),
                token_count=summary_data.get("token_count", 0),
            )

        return SessionSummaries(
            id=response.get("id", self.id) if response else self.id,
            short_summary=short_summary,
            long_summary=long_summary,
        )

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1, description="The search query to use"),
        filters: dict[str, object] | None = Field(
            None, description="Filters to scope the search"
        ),
        limit: int = Field(
            default=10, ge=1, le=100, description="Number of results to return"
        ),
    ) -> list[Message]:
        """
        Search for messages in this session.

        Makes an async API call to search for messages in this session.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/search",
            json={"query": query, "filters": filters, "limit": limit},
        )
        return [Message.model_validate(m) for m in (response or [])]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def upload_file(
        self,
        file: tuple[str, bytes, str] | tuple[str, Any, str] | Any = Field(
            ...,
            description="File to upload. Can be a file object, (filename, bytes, content_type) tuple, or (filename, fileobj, content_type) tuple.",
        ),
        peer: str | PeerBase = Field(
            ..., description="The peer creating the messages (ID string or Peer object)"
        ),
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with the messages",
        ),
        configuration: Configuration | dict[str, Any] | None = Field(
            None,
            description="Optional configuration dictionary to associate with the messages",
        ),
        created_at: str | datetime | None = Field(
            None,
            description="Optional created-at timestamp for the messages. Should be an ISO 8601 formatted string.",
        ),
    ) -> list[Message]:
        """
        Upload file to create message(s) in this session.

        Accepts a flexible payload:
        - File objects (opened in binary mode)
        - (filename, bytes, content_type) tuples
        - (filename, fileobj, content_type) tuples

        Files are normalized to (filename, fileobj, content_type) tuples for the HTTP client.

        Args:
            file: File to upload. Can be:
                - a file object (must have .name and .read())
                - a tuple (filename, bytes, content_type)
                - a tuple (filename, fileobj, content_type)
            peer: The peer who will be attributed as the creator of the messages.
                Can be a peer ID string or an AsyncPeer object.
            metadata: Optional metadata dictionary to associate with the messages
            configuration: Optional configuration dictionary to associate with the messages
            created_at: Optional created-at timestamp for the messages. Should be an ISO 8601 formatted string.

        Returns:
            A list of Message objects representing the created messages

        Note:
            Supported file types include PDFs, text files, and JSON documents.
            Large files will be automatically split into multiple messages to fit
            within message size limits.
        """

        # Prepare file for upload using shared utility
        filename, content_bytes, content_type = prepare_file_for_upload(file)

        # Extract peer ID from AsyncPeer object if needed
        resolved_peer_id = peer if isinstance(peer, str) else peer.id

        # Build form data
        files = {"file": (filename, content_bytes, content_type)}

        # Build extra data dict with optional fields
        data: dict[str, str] = {"peer_id": resolved_peer_id}
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if configuration is not None:
            config_dict = (
                configuration.model_dump()
                if isinstance(configuration, Configuration)
                else configuration
            )
            data["configuration"] = json.dumps(config_dict)
        if created_at is not None:
            # Ensure created_at is a string (ISO format)
            if isinstance(created_at, datetime):
                data["created_at"] = created_at.isoformat()
            else:
                data["created_at"] = created_at

        response = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/sessions/{self.id}/messages/upload",
            files=files,
            json=data,
        )

        return [Message.model_validate(msg) for msg in (response or [])]

    async def working_rep(
        self,
        peer: str | PeerBase,
        *,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """
        Get the current working representation of the peer in this session.

        Args:
            peer: Peer to get the working representation of.
            target: Optional target peer to get the representation of. If provided,
            queries what `peer` knows about the `target`.
            search_query: Semantic search query to filter relevant observations
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A Representation object containing explicit and deductive observations

        Example:
            ```python
            # Get peer's representation in this session
            rep = await session.working_rep('user123')
            print(rep)

            # Get what user123 knows about assistant in this session
            local_rep = await session.working_rep('user123', target='assistant')

            # Get representation with semantic search
            searched_rep = await session.working_rep(
                'user123',
                search_query='preferences',
                search_top_k=10
            )
            ```
        """
        from ..types import Representation as _Representation

        peer_id = peer if isinstance(peer, str) else peer.id
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        body: dict[str, Any] = {"session_id": self.id}
        if target_id is not None:
            body["target"] = target_id
        if search_query is not None:
            body["search_query"] = search_query
        if search_top_k is not None:
            body["search_top_k"] = search_top_k
        if search_max_distance is not None:
            body["search_max_distance"] = search_max_distance
        if include_most_derived is not None:
            body["include_most_derived"] = include_most_derived
        if max_observations is not None:
            body["max_observations"] = max_observations

        data = await self._http.request(
            "POST",
            f"/v2/workspaces/{self.workspace_id}/peers/{peer_id}/representation",
            json=body,
        )
        return _Representation.from_dict(data or {})

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def get_deriver_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
    ) -> DeriverStatus:
        """
        Get the deriver processing status, optionally scoped to an observer, sender, and/or session.

        Args:
            observer: Optional observer (ID string or AsyncPeer object) to scope the status check
            sender: Optional sender (ID string or AsyncPeer object) to scope the status check
        """
        resolved_observer_id = (
            None
            if observer is None
            else (observer if isinstance(observer, str) else observer.id)
        )
        resolved_sender_id = (
            None
            if sender is None
            else (sender if isinstance(sender, str) else sender.id)
        )

        params: dict[str, Any] = {"session_id": self.id}
        if resolved_observer_id:
            params["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            params["sender_id"] = resolved_sender_id

        response = await self._http.request(
            "GET",
            f"/v2/workspaces/{self.workspace_id}/queue/status",
            params=params,
        )
        return DeriverStatus.model_validate(response)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def poll_deriver_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        timeout: float = Field(
            300.0,
            gt=0,
            description="Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).",
        ),
    ) -> DeriverStatus:
        """
        Poll get_deriver_status until pending_work_units and in_progress_work_units are both 0.
        This allows you to guarantee that all messages have been processed by the deriver for
        use with the dialectic endpoint.

        The polling estimates sleep time by assuming each work unit takes 1 second.

        Args:
            observer: Optional observer (ID string or AsyncPeer object) to scope the status check
            sender: Optional sender (ID string or AsyncPeer object) to scope the status check
            timeout: Maximum time to poll in seconds. Defaults to 5 minutes (300 seconds).

        Returns:
            DeriverStatus when all work units are complete

        Raises:
            TimeoutError: If timeout is exceeded before work units complete
            Exception: If get_deriver_status fails repeatedly
        """
        start_time = time.time()

        while True:
            try:
                status = await self.get_deriver_status(observer, sender)
            except Exception as e:
                logger.warning(f"Failed to get deriver status: {e}")
                # Sleep briefly before retrying
                await asyncio.sleep(1)

                # Check timeout after error
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutError(
                        f"Polling timeout exceeded after {timeout}s. "
                        + f"Error during status check: {e}"
                    ) from e
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return status

            # Check timeout before sleeping
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            # Sleep for the expected time to complete all current work units
            # Assuming each pending and in-progress work unit takes 1 second
            total_work_units = status.pending_work_units + status.in_progress_work_units
            sleep_time = max(1, total_work_units)

            # Don't sleep past the timeout
            remaining_time = timeout - elapsed_time
            sleep_time = min(sleep_time, remaining_time)
            if sleep_time <= 0:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. "
                    + f"Current status: {status.pending_work_units} pending, "
                    + f"{status.in_progress_work_units} in progress work units."
                )

            await asyncio.sleep(sleep_time)

    def __repr__(self) -> str:
        """
        Return a string representation of the AsyncSession.

        Returns:
            A string representation suitable for debugging
        """
        return f"AsyncSession(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the AsyncSession.

        Returns:
            The session's ID
        """
        return self.id
