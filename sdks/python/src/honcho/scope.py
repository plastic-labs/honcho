# pyright: reportPrivateUsage=false
"""Sync Scope class for Honcho SDK."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, PrivateAttr, validate_call

from .api_types import ScopeBackfillJob, ScopeStatusResponse, SessionResponse
from .base import ScopeBase, SessionBase
from .http import routes
from .pagination import SyncPage
from .session import Session
from .utils import resolve_scope_membership, resolve_scope_session

if TYPE_CHECKING:
    from .aio import ScopeAio
    from .client import Honcho

logger = logging.getLogger(__name__)

__all__ = ["Scope"]


class Scope(ScopeBase):
    """
    Represents a scope in Honcho.

    A scope is a named set of sessions that acts as a visibility boundary. Recall
    performed through a scope sees only what happened in that scope's sessions,
    while the underlying peer keeps its single unified representation across
    everything it has ever participated in.

    Membership changes are applied asynchronously: adding a session that already
    has messages copies its existing conclusions into the scope, and removing one
    reconciles them back out. Poll :meth:`status` to watch that settle.

    Attributes:
        id: Unprefixed scope name, unique within the workspace
        workspace_id: Workspace ID for scoping operations
        metadata: Cached metadata for this scope. May be stale if not recently
            fetched.
        created_at: When this scope was created, if known

    Example:
        ```python
        therapy = honcho.scope("therapy")
        therapy.add_sessions([session_1, session_2])

        # Ask a question answered only from the therapy sessions
        answer = user.chat("What is stressing them out?", scope="therapy")
        ```
    """

    _metadata: dict[str, Any] | None = PrivateAttr(default=None)
    _created_at: datetime | None = PrivateAttr(default=None)
    _honcho: "Honcho" = PrivateAttr()

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Cached metadata for this scope. May be stale if not recently fetched."""
        return self._metadata

    @property
    def created_at(self) -> datetime | None:
        """When this scope was created. Only available if fetched from the API."""
        return self._created_at

    def __init__(
        self,
        scope_id: str,
        honcho: "Honcho",
        *,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """
        Initialize a new Scope.

        **Do not call this directly — use** ``honcho.scope()``.

        Args:
            scope_id: Unprefixed scope name, unique within the workspace
            honcho: Honcho client instance
            metadata: Cached metadata, if already fetched
            created_at: Creation timestamp, if already fetched
        """
        super().__init__(
            id=scope_id,
            workspace_id=honcho.workspace_id,
        )
        self._honcho = honcho
        self._metadata = metadata
        self._created_at = created_at

    @property
    def aio(self) -> "ScopeAio":
        """
        Access async versions of all Scope methods.

        Returns a ScopeAio view that provides async versions of all methods while
        sharing state with this Scope instance.

        Example:
            ```python
            await scope.aio.add_sessions(["session-1"])
            status = await scope.aio.status()
            ```
        """
        # Import here to avoid circular import (aio.py imports this module)
        from .aio import ScopeAio

        return ScopeAio(self)

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def add_sessions(self, sessions: Sequence[str | SessionBase]) -> None:
        """
        Add sessions to this scope.

        Every named session must already exist. Adding a session that is already
        a member is a no-op.

        Sessions that already hold messages are backfilled into the scope
        asynchronously, so recall through this scope may not reflect their history
        immediately — poll :meth:`status` to watch that complete.

        Args:
            sessions: Sessions to add, as ID strings or Session objects. At most
                100 per call, matching the server's limit; split larger membership
                changes into separate calls so a failure names the batch that
                failed.

        Raises:
            ValueError: If no sessions are given, or more than 100.
        """
        session_ids = resolve_scope_membership(sessions)
        self._honcho._ensure_workspace()
        self._honcho._http.post(
            routes.scope_sessions(self.workspace_id, self.id),
            body={"session_ids": session_ids},
        )

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def remove_session(self, session: str | SessionBase) -> None:
        """
        Remove a session from this scope.

        Conclusions copied or derived while the session was a member are
        reconciled out asynchronously, and the scope's peer card is rebuilt from
        whatever evidence remains. Poll :meth:`status` to watch that settle.

        Args:
            session: Session to remove, as an ID string or a Session object
        """
        self._honcho._ensure_workspace()
        self._honcho._http.delete(
            routes.scope_session(
                self.workspace_id, self.id, resolve_scope_session(session)
            )
        )

    def sessions(
        self,
        page: int = 1,
        size: int = 50,
        *,
        reverse: bool = False,
    ) -> SyncPage[SessionResponse, Session]:
        """
        Get the sessions that are members of this scope.

        Ordered by how long each session has been a member — longest-standing
        first, or most recently added first when ``reverse`` is True.

        Args:
            page: Page number (1-indexed)
            size: Number of results per page
            reverse: If True, reverses the default ordering. Default: False.

        Returns:
            Paginated response containing Session objects
        """
        self._honcho._ensure_workspace()

        def fetch(next_page: int) -> dict[str, Any]:
            query: dict[str, Any] = {"page": next_page, "size": size}
            if reverse:
                query["reverse"] = "true"
            return self._honcho._http.post(
                routes.scope_sessions_list(self.workspace_id, self.id),
                query=query,
            )

        def transform(response: SessionResponse) -> Session:
            return Session(
                response.id,
                self._honcho,
                metadata=response.metadata,
                configuration=response.configuration,
                created_at=response.created_at,
                is_active=response.is_active,
            )

        def fetch_next(next_page: int) -> SyncPage[SessionResponse, Session]:
            return SyncPage(fetch(next_page), SessionResponse, transform, fetch_next)

        return SyncPage(fetch(page), SessionResponse, transform, fetch_next)

    def status(self) -> dict[str, ScopeBackfillJob]:
        """
        Get the backfill/reconciliation progress for this scope.

        Use this after a membership change to tell "the scope knows nothing about
        that session yet" apart from "the scope has caught up and there is
        genuinely nothing to recall".

        Returns:
            Per-session backfill state, keyed by session ID. Only sessions that
            have had a backfill enqueued appear; an empty dict means none have.
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.get(routes.scope_status(self.workspace_id, self.id))
        return ScopeStatusResponse.model_validate(data).backfill_status

    def __repr__(self) -> str:
        return f"Scope(id={self.id!r}, workspace_id={self.workspace_id!r})"

    def __str__(self) -> str:
        return self.id
