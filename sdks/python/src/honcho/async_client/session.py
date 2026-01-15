"""Async Session class for Honcho SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, validate_call

from ..api_types import (
    MessageCreateParams,
    MessageResponse,
    PeerResponse,
    QueueStatusResponse,
    RepresentationResponse,
    SessionResponse,
)
from ..base import PeerBase, SessionBase
from ..http import AsyncHonchoHTTPClient, routes
from ..session_context import SessionContext, SessionSummaries, Summary
from ..utils import prepare_file_for_upload
from .pagination import AsyncPage

if TYPE_CHECKING:
    from .peer import AsyncPeer

logger = logging.getLogger(__name__)


class SessionPeerConfig(BaseModel):
    """Configuration for a peer within a session."""

    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form session-level representations of other peers",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers should form representations of this peer",
    )


class AsyncSession(SessionBase):
    """
    Represents a session in Honcho with async operations.

    Sessions are scoped to a set of peers and contain messages/content.

    Attributes:
        id: Unique identifier for this session
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this session.
        configuration: Cached configuration for this session.
    """

    _metadata: dict[str, object] | None = PrivateAttr(default=None)
    _configuration: dict[str, object] | None = PrivateAttr(default=None)
    _http: AsyncHonchoHTTPClient = PrivateAttr()

    @property
    def metadata(self) -> dict[str, object] | None:
        """Cached metadata for this session."""
        return self._metadata

    @property
    def configuration(self) -> dict[str, object] | None:
        """Cached configuration for this session."""
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
        http: AsyncHonchoHTTPClient = Field(..., description="HTTP client instance"),
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
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
        http: AsyncHonchoHTTPClient,
        *,
        metadata: dict[str, object] | None = None,
        config: dict[str, object] | None = None,
    ) -> AsyncSession:
        """Create a new AsyncSession with optional configuration."""
        if config is not None or metadata is not None:
            body: dict[str, Any] = {"id": session_id}
            if metadata is not None:
                body["metadata"] = metadata
            if config is not None:
                body["configuration"] = config

            data = await http.post(routes.sessions(workspace_id), body=body)
            session_data = SessionResponse.model_validate(data)
            return cls(
                session_id,
                workspace_id,
                http,
                metadata=session_data.metadata,
                config=session_data.configuration,
            )

        return cls(session_id, workspace_id, http)

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
        """Add peers to this session."""
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._http.post(
            routes.session_peers_add(self.workspace_id, self.id),
            body=peer_dict,
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
        """Set the complete peer list for this session."""
        if not isinstance(peers, list):
            peers = [peers]

        peer_dict: dict[str, Any] = {}
        for peer in peers:
            if isinstance(peer, tuple):
                peer_id = peer[0] if isinstance(peer[0], str) else peer[0].id
                peer_config = peer[1]
                peer_dict[peer_id] = peer_config.model_dump(exclude_none=True)
            else:
                peer_id = peer if isinstance(peer, str) else peer.id
                peer_dict[peer_id] = {}

        await self._http.put(
            routes.session_peers_set(self.workspace_id, self.id),
            body=peer_dict,
        )

    async def remove_peers(
        self,
        peers: str | PeerBase | list[PeerBase | str] = Field(
            ..., description="Peers to remove from the session"
        ),
    ) -> None:
        """Remove peers from this session."""
        if not isinstance(peers, list):
            peers = [peers]

        peer_ids = [peer if isinstance(peer, str) else peer.id for peer in peers]

        await self._http.delete(
            routes.session_peers_remove(self.workspace_id, self.id),
            body=peer_ids,
        )

    async def get_peers(self) -> list["AsyncPeer"]:
        """Get all peers in this session."""
        from .peer import AsyncPeer

        data: dict[str, Any] = await self._http.get(
            routes.session_peers(self.workspace_id, self.id)
        )

        peers_data: list[Any] = data.get("items", [])
        return [
            AsyncPeer(
                PeerResponse.model_validate(peer).id, self.workspace_id, self._http
            )
            for peer in peers_data
        ]

    async def get_peer_config(self, peer: str | PeerBase) -> SessionPeerConfig:
        """Get the configuration for a peer in this session."""
        peer_id = peer if isinstance(peer, str) else peer.id
        data = await self._http.get(
            routes.session_peer_config(self.workspace_id, self.id, peer_id)
        )
        return SessionPeerConfig(
            observe_others=data.get("observe_others"),
            observe_me=data.get("observe_me"),
        )

    async def set_peer_config(
        self, peer: str | PeerBase, config: SessionPeerConfig
    ) -> None:
        """Set the configuration for a peer in this session."""
        peer_id = peer if isinstance(peer, str) else peer.id
        body: dict[str, Any] = {}
        if config.observe_others is not None:
            body["observe_others"] = config.observe_others
        if config.observe_me is not None:
            body["observe_me"] = config.observe_me

        await self._http.put(
            routes.session_peer_config(self.workspace_id, self.id, peer_id),
            body=body,
        )

    @validate_call
    async def add_messages(
        self,
        messages: MessageCreateParams | list[MessageCreateParams] = Field(
            ..., description="Messages to add to the session"
        ),
    ) -> list[MessageResponse]:
        """Add one or more messages to this session."""
        if not isinstance(messages, list):
            messages = [messages]

        # Convert MessageCreateParams to dict
        messages_data: list[dict[str, Any]] = []
        for msg in messages:
            msg_dict: dict[str, Any] = {
                "content": msg.content,
                "peer_id": msg.peer_id,
            }
            if msg.metadata is not None:
                msg_dict["metadata"] = msg.metadata
            if msg.configuration is not None:
                msg_dict["configuration"] = msg.configuration.model_dump(
                    exclude_none=True
                )
            if msg.created_at is not None:
                created_at_val = msg.created_at
                msg_dict["created_at"] = (
                    created_at_val.isoformat()
                    if hasattr(created_at_val, "isoformat")
                    else str(created_at_val)
                )
            messages_data.append(msg_dict)

        data = await self._http.post(
            routes.messages(self.workspace_id, self.id),
            body={"messages": messages_data},
        )
        return [MessageResponse.model_validate(msg) for msg in data]

    @validate_call
    async def get_messages(
        self,
        *,
        filters: dict[str, object] | None = Field(
            None, description="Dictionary of filter criteria"
        ),
    ) -> AsyncPage[MessageResponse, MessageResponse]:
        """Get messages from this session with optional filtering."""
        data = await self._http.post(
            routes.messages_list(self.workspace_id, self.id),
            body={"filters": filters} if filters else None,
        )

        async def fetch_next(page: int) -> AsyncPage[MessageResponse, MessageResponse]:
            next_data = await self._http.post(
                routes.messages_list(self.workspace_id, self.id),
                body={"filters": filters} if filters else None,
                query={"page": page},
            )
            return AsyncPage(next_data, MessageResponse, None, fetch_next)

        return AsyncPage(data, MessageResponse, None, fetch_next)

    async def delete(self) -> None:
        """Delete this session and all associated data."""
        await self._http.delete(routes.session(self.workspace_id, self.id))

    async def clone(
        self,
        *,
        message_id: str | None = None,
    ) -> "AsyncSession":
        """Clone this session, optionally up to a specific message."""
        query: dict[str, Any] = {}
        if message_id is not None:
            query["message_id"] = message_id

        data = await self._http.post(
            routes.session_clone(self.workspace_id, self.id),
            query=query if query else None,
        )
        cloned = SessionResponse.model_validate(data)
        return AsyncSession(
            cloned.id,
            self.workspace_id,
            self._http,
            metadata=cloned.metadata,
            config=cloned.configuration,
        )

    async def get_metadata(self) -> dict[str, object]:
        """Get metadata for this session."""
        data = await self._http.post(
            routes.sessions(self.workspace_id),
            body={"id": self.id},
        )
        session = SessionResponse.model_validate(data)
        self._metadata = session.metadata or {}
        return self._metadata

    @validate_call
    async def set_metadata(
        self,
        metadata: dict[str, object] = Field(..., description="Metadata dictionary"),
    ) -> None:
        """Set metadata for this session."""
        await self._http.put(
            routes.session(self.workspace_id, self.id),
            body={"metadata": metadata},
        )
        self._metadata = metadata

    async def get_config(self) -> dict[str, object]:
        """Get configuration for this session."""
        data = await self._http.post(
            routes.sessions(self.workspace_id),
            body={"id": self.id},
        )
        session = SessionResponse.model_validate(data)
        self._configuration = session.configuration or {}
        return self._configuration

    @validate_call
    async def set_config(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary"
        ),
    ) -> None:
        """Set configuration for this session."""
        await self._http.put(
            routes.session(self.workspace_id, self.id),
            body={"configuration": configuration},
        )
        self._configuration = configuration

    async def refresh(self) -> None:
        """Refresh cached metadata and configuration for this session."""
        data = await self._http.post(
            routes.sessions(self.workspace_id),
            body={"id": self.id},
        )
        session = SessionResponse.model_validate(data)
        self._metadata = session.metadata or {}
        self._configuration = session.configuration or {}

    @validate_call
    async def get_context(
        self,
        *,
        summary: bool = True,
        tokens: int | None = Field(None, gt=0),
        peer_target: str | None = None,
        last_user_message: str | MessageResponse | None = None,
        peer_perspective: str | None = None,
        limit_to_session: bool = False,
        search_top_k: int | None = Field(None, ge=1, le=100),
        search_max_distance: float | None = Field(None, ge=0.0, le=1.0),
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = Field(None, ge=1, le=100),
    ) -> SessionContext:
        """Get optimized context for this session within a token limit."""
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
            if isinstance(last_user_message, MessageResponse)
            else last_user_message
        )

        query: dict[str, Any] = {
            "summary": summary,
            "limit_to_session": limit_to_session,
        }
        if tokens is not None:
            query["tokens"] = tokens
        if last_user_message_id is not None:
            query["last_message"] = last_user_message_id
        if peer_target is not None:
            query["peer_target"] = peer_target
        if peer_perspective is not None:
            query["peer_perspective"] = peer_perspective
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._http.get(
            routes.session_context(self.workspace_id, self.id),
            query=query,
        )

        # Convert summary if present
        session_summary = None
        if data.get("summary"):
            s = data["summary"]
            session_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        # Parse messages
        messages = [
            MessageResponse.model_validate(msg) for msg in data.get("messages", [])
        ]

        return SessionContext(
            session_id=self.id,
            messages=messages,
            summary=session_summary,
            peer_representation=str(data.get("peer_representation"))
            if data.get("peer_representation")
            else None,
            peer_card=data.get("peer_card"),
        )

    async def get_summaries(self) -> SessionSummaries:
        """Get available summaries for this session."""
        data = await self._http.get(
            routes.session_summaries(self.workspace_id, self.id)
        )

        short_summary = None
        if data.get("short_summary"):
            s = data["short_summary"]
            short_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        long_summary = None
        if data.get("long_summary"):
            s = data["long_summary"]
            long_summary = Summary(
                content=s["content"],
                message_id=s["message_id"],
                summary_type=s["summary_type"],
                created_at=s["created_at"],
                token_count=s["token_count"],
            )

        return SessionSummaries(
            id=data.get("id") or self.id,
            short_summary=short_summary,
            long_summary=long_summary,
        )

    @validate_call
    async def search(
        self,
        query: str = Field(..., min_length=1),
        filters: dict[str, object] | None = None,
        limit: int = Field(default=10, ge=1, le=100),
    ) -> list[MessageResponse]:
        """Search for messages in this session."""
        data = await self._http.post(
            routes.session_search(self.workspace_id, self.id),
            body={"query": query, "filters": filters, "limit": limit},
        )
        return [MessageResponse.model_validate(msg) for msg in data]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def upload_file(
        self,
        file: tuple[str, bytes, str] | tuple[str, Any, str] | Any = Field(
            ..., description="File to upload"
        ),
        peer: str | PeerBase = Field(...),
        metadata: dict[str, object] | None = None,
        configuration: dict[str, Any] | None = None,
        created_at: str | datetime | None = None,
    ) -> list[MessageResponse]:
        """Upload file to create message(s) in this session."""
        filename, content_bytes, content_type = prepare_file_for_upload(file)
        resolved_peer_id = peer if isinstance(peer, str) else peer.id

        # Build form data
        data_dict: dict[str, str] = {"peer_id": resolved_peer_id}
        if metadata is not None:
            data_dict["metadata"] = json.dumps(metadata)
        if configuration is not None:
            data_dict["configuration"] = json.dumps(configuration)
        if created_at is not None:
            if isinstance(created_at, datetime):
                data_dict["created_at"] = created_at.isoformat()
            else:
                data_dict["created_at"] = created_at

        response = await self._http.upload(
            routes.messages_upload(self.workspace_id, self.id),
            files={"file": (filename, content_bytes, content_type)},
            data=data_dict,
        )

        return [MessageResponse.model_validate(msg) for msg in response]

    async def get_representation(
        self,
        peer: str | PeerBase,
        *,
        target: str | PeerBase | None = None,
        search_query: str | None = None,
        search_top_k: int | None = None,
        search_max_distance: float | None = None,
        include_most_frequent: bool | None = None,
        max_conclusions: int | None = None,
    ) -> str:
        """Get a subset of the representation of the peer in this session."""
        peer_id = peer if isinstance(peer, str) else peer.id
        target_id = (
            None
            if target is None
            else (target if isinstance(target, str) else target.id)
        )

        query: dict[str, Any] = {"session_id": self.id}
        if target_id:
            query["target"] = target_id
        if search_query is not None:
            query["search_query"] = search_query
        if search_top_k is not None:
            query["search_top_k"] = search_top_k
        if search_max_distance is not None:
            query["search_max_distance"] = search_max_distance
        if include_most_frequent is not None:
            query["include_most_frequent"] = include_most_frequent
        if max_conclusions is not None:
            query["max_conclusions"] = max_conclusions

        data = await self._http.post(
            routes.peer_representation(self.workspace_id, peer_id),
            body=query,
        )
        response = RepresentationResponse.model_validate(data)
        return response.representation

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def get_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
    ) -> QueueStatusResponse:
        """Get the queue processing status for this session."""
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

        query: dict[str, Any] = {"session_id": self.id}
        if resolved_observer_id:
            query["observer_id"] = resolved_observer_id
        if resolved_sender_id:
            query["sender_id"] = resolved_sender_id

        data = await self._http.get(
            routes.workspace_queue_status(self.workspace_id),
            query=query,
        )
        return QueueStatusResponse.model_validate(data)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    async def poll_queue_status(
        self,
        observer: str | PeerBase | None = None,
        sender: str | PeerBase | None = None,
        timeout: float = Field(300.0, gt=0),
    ) -> QueueStatusResponse:
        """Poll get_queue_status until all work units are complete."""
        start_time = time.time()

        while True:
            try:
                status = await self.get_queue_status(observer, sender)
            except Exception as e:
                logger.warning(f"Failed to get queue status: {e}")
                await asyncio.sleep(1)

                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    raise TimeoutError(
                        f"Polling timeout exceeded after {timeout}s. Error: {e}"
                    ) from e
                continue

            if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                return status

            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress."
                )

            total_work_units = status.pending_work_units + status.in_progress_work_units
            sleep_time = max(1, total_work_units)
            remaining_time = timeout - elapsed_time
            sleep_time = min(sleep_time, remaining_time)

            if sleep_time <= 0:
                raise TimeoutError(
                    f"Polling timeout exceeded after {timeout}s. Status: {status.pending_work_units} pending, {status.in_progress_work_units} in progress."
                )

            await asyncio.sleep(sleep_time)

    def __repr__(self) -> str:
        return f"AsyncSession(id='{self.id}')"

    def __str__(self) -> str:
        return self.id
