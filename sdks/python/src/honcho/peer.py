from __future__ import annotations

import datetime
from collections.abc import Generator
from typing import TYPE_CHECKING, cast

from honcho_core import Honcho as HonchoCore
from honcho_core._types import omit
from honcho_core.types.workspaces import PeerCardResponse
from honcho_core.types.workspaces.session import Session as SessionCore
from honcho_core.types.workspaces.sessions import MessageCreateParam
from honcho_core.types.workspaces.sessions.message import Message
from honcho_core.types.workspaces.sessions.message_create_param import Configuration
from pydantic import ConfigDict, Field, PrivateAttr, validate_call

from .base import PeerBase, SessionBase
from .pagination import SyncPage
from .types import DialecticStreamResponse

if TYPE_CHECKING:
    from .observations import ObservationScope
    from .types import PeerContext, Representation

from .session import Session


class Peer(PeerBase):
    """
    Represents a peer in the Honcho system.

    Peers can send messages, participate in sessions, and maintain both global
    and local representations for contextual interactions. A peer represents
    an entity (user, assistant, etc.) that can communicate within the system.

    Attributes:
        id: Unique identifier for this peer
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this peer. May be stale if not recently
            fetched. Call get_metadata() for fresh data.
        configuration: Cached configuration for this peer. May be stale if not
            recently fetched. Call get_config() for fresh data.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _client: HonchoCore = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this peer. May be stale. Use get_metadata() for fresh data."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this peer. May be stale. Use get_config() for fresh data."""
        return self._configuration

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        peer_id: str = Field(
            ...,
            min_length=1,
            description="Unique identifier for this peer within the workspace",
        ),
        workspace_id: str = Field(
            ..., min_length=1, description="Workspace ID for scoping operations"
        ),
        client: HonchoCore = Field(
            ..., description="Reference to the parent Honcho client instance"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None,
            description="Optional metadata dictionary to associate with this peer. If set, will get/create peer immediately with metadata.",
        ),
        config: dict[str, object] | None = Field(
            None,
            description="Optional configuration to set for this peer. If set, will get/create peer immediately with flags.",
        ),
    ) -> None:
        """
        Initialize a new Peer.

        Provided metadata and configuration will overwrite any existing data in those
        locations if given.

        Args:
            peer_id: Unique identifier for this peer within the workspace
            workspace_id: Workspace ID for scoping operations
            client: Reference to the parent Honcho client instance
            metadata: Optional metadata dictionary to associate with this peer.
            If set, will get/create peer immediately with metadata.
            config: Optional configuration to set for this peer.
            If set, will get/create peer immediately with flags.
        """
        super().__init__(
            id=peer_id,
            workspace_id=workspace_id,
        )
        self._client = client
        self._metadata = metadata
        self._configuration = config

        if config is not None or metadata is not None:
            peer_data = self._client.workspaces.peers.get_or_create(
                workspace_id=workspace_id,
                id=peer_id,
                configuration=config if config is not None else omit,
                metadata=metadata if metadata is not None else omit,
            )
            # Update cached values with API response
            self._metadata = peer_data.metadata
            self._configuration = peer_data.configuration

    def chat(
        self,
        query: str,
        *,
        stream: bool = False,
        target: str | PeerBase | None = None,
        session: str | SessionBase | None = None,
    ) -> str | DialecticStreamResponse | None:
        """
        Query the peer's representation with a natural language question.

        Makes an API call to the Honcho dialectic endpoint to query either the peer's
        global representation (all content associated with this peer) or their local
        representation of another peer (what this peer knows about the target peer).

        Args:
            query: The natural language question to ask.
            stream: Whether to stream the response
            target: Optional target peer for local representation query. If provided,
                    queries what this peer knows about the target peer rather than
                    querying the peer's global representation. Can be a peer ID string
                    or a Peer object.
            session: Optional session to scope the query to. If provided, only
                     information from that session is considered. Can be a session
                     ID string or a Session object.

        Returns:
            For non-streaming: Response string containing the answer, or None if no relevant information
            For streaming: DialecticStreamResponse object that can be iterated over and provides final response
        """
        # Extract IDs from objects if needed
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )
        resolved_session_id = (
            None
            if session is None
            else (session if isinstance(session, str) else session.id)
        )

        if stream:

            def stream_response() -> Generator[str, None, None]:
                import json

                # Use core SDK with_streaming_response
                with self._client.workspaces.peers.with_streaming_response.chat(
                    peer_id=self.id,
                    workspace_id=self.workspace_id,
                    query=query,
                    stream=True,
                    target=target_id,
                    session_id=resolved_session_id,
                ) as response:
                    response.http_response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            json_str = line[6:]  # Remove "data: " prefix
                            try:
                                chunk_data = json.loads(json_str)
                                if chunk_data.get("done"):
                                    break
                                delta_obj = chunk_data.get("delta", {})
                                content = delta_obj.get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

            return DialecticStreamResponse(stream_response())

        response = self._client.workspaces.peers.chat(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            query=query,
            stream=stream,
            target=target_id,
            session_id=resolved_session_id,
        )
        if response.content in ("", None, "None"):
            return None
        return response.content

    def get_sessions(
        self, filters: dict[str, object] | None = None
    ) -> SyncPage[SessionCore, Session]:
        """
        Get all sessions this peer is a member of.

        Makes an API call to retrieve all sessions where this peer is an active participant.
        Sessions are created when peers are added to them or send messages to them.

        Returns:
            A paginated list of Session objects this peer belongs to. Returns an empty
            list if the peer is not a member of any sessions
        """
        from .session import Session

        sessions_page = self._client.workspaces.peers.sessions.list(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            filters=filters,
        )
        return SyncPage(
            sessions_page,
            lambda session: Session(session.id, self.workspace_id, self._client),
        )

    @validate_call
    def message(
        self,
        content: str = Field(
            ..., min_length=1, description="The text content for the message"
        ),
        *,
        metadata: dict[str, object] | None = Field(
            None, description="Optional metadata dictionary"
        ),
        config: Configuration | None = Field(
            None,
            description="Optional configuration dictionary to associate with the message",
        ),
        created_at: datetime.datetime | str | None = Field(
            None,
            description="Optional created-at timestamp for the message. Accepts a datetime which will be converted to an ISO 8601 string, or a preformatted string.",
        ),
    ) -> MessageCreateParam:
        """
        Create a MessageCreateParam object attributed to this peer.

        This is a convenience method for creating MessageCreateParam objects with this peer's ID.
        The created MessageCreateParam can then be added to sessions or used in other operations.

        Args:
            content: The text content for the message
            metadata: Optional metadata dictionary to associate with the message

        Returns:
            A new MessageCreateParam object with this peer's ID and the provided content
        """
        created_at_str: str | None
        if isinstance(created_at, datetime.datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = created_at

        return MessageCreateParam(
            peer_id=self.id,
            content=content,
            configuration=config,
            metadata=metadata,
            created_at=created_at_str,
        )

    def get_metadata(self) -> dict[str, object]:
        """
        Get the current metadata for this peer.

        Makes an API call to retrieve metadata associated with this peer. Metadata
        can include custom attributes, settings, or any other key-value data
        associated with the peer. This method also updates the cached metadata attribute.

        Returns:
            A dictionary containing the peer's metadata. Returns an empty dictionary
            if no metadata is set
        """
        peer = self._client.workspaces.peers.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        self._metadata = peer.metadata or {}
        return self._metadata

    @validate_call
    def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the metadata for this peer.

        Makes an API call to update the metadata associated with this peer.
        This will overwrite any existing metadata with the provided values.
        This method also updates the cached metadata attribute.

        Args:
            metadata: A dictionary of metadata to associate with this peer.
            Keys must be strings, values can be any JSON-serializable type
        """
        self._client.workspaces.peers.update(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            metadata=metadata,
        )
        self._metadata = metadata

    def get_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        Makes an API call to retrieve configuration associated with this peer.
        Configuration currently includes one optional flag, `observe_me`.
        This method also updates the cached configuration attribute.

        Returns:
            A dictionary containing the peer's configuration
        """
        peer = self._client.workspaces.peers.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        self._configuration = peer.configuration or {}
        return self._configuration

    @validate_call
    def set_config(
        self,
        config: dict[str, object] = Field(
            ..., description="Configuration dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the configuration for this peer. Currently the only supported config
        value is the `observe_me` flag, which controls whether derivation tasks
        should be created for this peer's global representation. Default is True.

        Makes an API call to update the configuration associated with this peer.
        This will overwrite any existing configuration with the provided values.
        This method also updates the cached configuration attribute.

        Args:
            config: A dictionary of configuration to associate with this peer.
            Keys must be strings, values can be any JSON-serializable type
        """
        self._client.workspaces.peers.update(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            configuration=config,
        )
        self._configuration = config

    def get_peer_config(self) -> dict[str, object]:
        """
        Get the current workspace-level configuration for this peer.

        .. deprecated::
            Use :meth:`get_config` instead.

        Returns:
            A dictionary containing the peer's configuration
        """
        return self.get_config()

    @validate_call
    def set_peer_config(
        self,
        config: dict[str, object] = Field(
            ..., description="Configuration dictionary to associate with this peer"
        ),
    ) -> None:
        """
        Set the configuration for this peer.

        .. deprecated::
            Use :meth:`set_config` instead.

        Args:
            config: A dictionary of configuration to associate with this peer
        """
        return self.set_config(config)

    def refresh(self) -> None:
        """
        Refresh cached metadata and configuration for this peer.

        Makes a single API call to retrieve the latest metadata and configuration
        associated with this peer and updates the cached attributes.
        """
        peer = self._client.workspaces.peers.get_or_create(
            workspace_id=self.workspace_id,
            id=self.id,
        )
        self._metadata = peer.metadata or {}
        self._configuration = peer.configuration or {}

    @validate_call
    def search(
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
        Search across all messages in the workspace with this peer as author.

        Makes an API call to search endpoint.

        Args:
            query: The search query to use
            filters: Filters to scope the search. See [search filters documentation](https://docs.honcho.dev/v2/guides/using-filters).
            limit: Number of results to return (1-100, default: 10)

        Returns:
            A list of Message objects representing the search results.
            Returns an empty list if no messages are found.
        """
        return self._client.workspaces.peers.search(
            self.id,
            workspace_id=self.workspace_id,
            query=query,
            filters=filters,
            limit=limit,
        )

    def card(
        self,
        target: str | PeerBase | None = None,
    ) -> str:
        """
        Get the peer card for this peer.

        Makes an API call to retrieve the peer card, which contains a representation
        of what this peer knows. If a target is provided, returns this peer's local
        representation of the target peer.

        Args:
            target: Optional target peer for local card. If provided, returns this
                    peer's card of the target peer. Can be a Peer object or peer ID string.

        Returns:
            A string containing the peer card joined with newlines, or an empty string if none is available
        """
        # Validate target parameter
        if isinstance(target, str) and len(target.strip()) == 0:
            raise ValueError("target string cannot be empty")

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )
        response: PeerCardResponse = self._client.workspaces.peers.card(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            target=target_id,
        )
        if response.peer_card is None:
            return ""

        items: list[str] = response.peer_card

        return "\n".join(items)

    def working_rep(
        self,
        session: str | SessionBase | None = None,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "Representation":
        """
        Get a working representation for this peer.

        Args:
            session: Optional session to scope the representation to.
            target: Optional target peer to get the representation of. If provided,
            returns the representation of the target from the perspective of this peer.
            search_query: Semantic search query to filter relevant observations
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A Representation object containing explicit and deductive observations

        Example:
            ```python
            # Get global representation
            rep = peer.working_rep()
            print(rep)

            # Get representation scoped to a session
            session_rep = peer.working_rep(session='session-123')

            # Get representation with semantic search
            searched_rep = peer.working_rep(
                search_query='preferences',
                search_top_k=10,
                max_observations=50
            )
            ```
        """
        from .types import Representation as _Representation

        session_id = (
            None
            if session is None
            else session
            if isinstance(session, str)
            else session.id
        )

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )
        data = self._client.workspaces.peers.working_representation(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            session_id=session_id,
            target=target_id,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )
        representation = data.get("representation")
        if representation is not None:
            return _Representation.from_dict(cast(dict[str, object], representation))
        else:
            return _Representation.from_dict(data)

    def get_context(
        self,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_derived: bool | None = None,
        max_observations: int | None = None,
    ) -> "PeerContext":
        """
        Get context for this peer, including representation and peer card.

        This is a convenience method that retrieves both the working representation
        and peer card in a single API call.

        Args:
            target: Optional target peer to get context for. If provided, returns
                   the context for the target from this peer's perspective.
                   Can be a Peer object or peer ID string.
            search_query: Semantic search query to filter relevant observations
            search_top_k: Number of semantically relevant facts to return
            search_max_distance: Maximum semantic distance for search results (0.0-1.0)
            include_most_derived: Whether to include the most derived observations
            max_observations: Maximum number of observations to include

        Returns:
            A PeerContext object containing the representation and peer card

        Example:
            ```python
            # Get own context
            context = peer.get_context()
            print(context.representation)
            print(context.peer_card)

            # Get context for another peer
            context = peer.get_context(target='other-peer-id')

            # Get context with semantic search
            context = peer.get_context(
                search_query='preferences',
                search_top_k=10
            )
            ```
        """
        from .types import PeerContext as _PeerContext

        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        response = self._client.workspaces.peers.get_context(
            peer_id=self.id,
            workspace_id=self.workspace_id,
            target=target_id,
            search_query=search_query if search_query is not None else omit,
            search_top_k=search_top_k if search_top_k is not None else omit,
            search_max_distance=search_max_distance
            if search_max_distance is not None
            else omit,
            include_most_derived=include_most_derived
            if include_most_derived is not None
            else omit,
            max_observations=max_observations if max_observations is not None else omit,
        )

        return _PeerContext.from_api_response(response)

    @property
    def observations(self) -> "ObservationScope":
        """
        Access this peer's self-observations (where observer == observed == self).

        This property provides a convenient way to access observations that this peer
        has made about themselves. Use this for self-observation scenarios.

        Returns:
            An ObservationScope scoped to this peer's self-observations

        Example:
            ```python
            # List self-observations
            obs_list = peer.observations.list()

            # Search self-observations
            results = peer.observations.query("preferences")

            # Delete a self-observation
            peer.observations.delete("obs-123")
            ```
        """
        from .observations import ObservationScope as _ObservationScope

        return _ObservationScope(self._client, self.workspace_id, self.id, self.id)

    def observations_of(self, target: str | PeerBase) -> "ObservationScope":
        """
        Access observations this peer has made about another peer.

        This method provides scoped access to observations where this peer is the
        observer and the target is the observed peer.

        Args:
            target: The target peer (either a Peer object or peer ID string)

        Returns:
            An ObservationScope scoped to this peer's observations of the target

        Example:
            ```python
            # Get observations about another peer
            bob_observations = peer.observations_of("bob")

            # List observations
            obs_list = bob_observations.list()

            # Search observations
            results = bob_observations.query("work history")

            # Get the representation from these observations
            rep = bob_observations.get_representation()
            ```
        """
        from .observations import ObservationScope as _ObservationScope

        target_id = target.id if isinstance(target, PeerBase) else target
        return _ObservationScope(self._client, self.workspace_id, self.id, target_id)

    def __repr__(self) -> str:
        """
        Return a string representation of the Peer.

        Returns:
            A string representation suitable for debugging
        """
        return f"Peer(id='{self.id}')"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Peer.

        Returns:
            The peer's ID
        """
        return self.id
