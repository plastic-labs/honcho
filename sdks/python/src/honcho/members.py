# pyright: reportPrivateUsage=false
"""Peer membership access for the Honcho SDK."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .api_types import PeerMembership
from .base import PeerBase
from .http import routes
from .pagination import SyncPage
from .utils import resolve_id
from .utils.scopes import _RESOURCE_NAME_PATTERN

if TYPE_CHECKING:
    from .aio import PeerMembersAio
    from .client import Honcho

__all__ = ["PeerMembers"]


def resolve_member_id(peer: str | PeerBase) -> str:
    """Resolve and validate the member ID interpolated into a membership path.

    Validated because an unvalidated value silently changes which resource the
    request addresses: ``alice?typo`` would remove ``alice``.

    Raises:
        ValueError: If the ID is empty or malformed.
    """
    member_id = resolve_id(peer)
    if not re.fullmatch(_RESOURCE_NAME_PATTERN, member_id):
        raise ValueError(f"Peer ID must match pattern {_RESOURCE_NAME_PATTERN}")
    return member_id


def membership_body(
    peer: str | PeerBase, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the request body for adding a member."""
    body: dict[str, Any] = {"peer_id": resolve_id(peer)}
    if metadata is not None:
        body["metadata"] = metadata
    return body


def list_memberships(
    honcho: "Honcho",
    route: str,
    filters: dict[str, Any] | None,
    *,
    page: int,
    size: int,
) -> SyncPage[PeerMembership]:
    """Fetch one page of a members or parents listing, wired to fetch the rest."""
    honcho._ensure_workspace()

    def fetch_next(next_page: int) -> SyncPage[PeerMembership]:
        data = honcho._http.post(
            route,
            body={"filters": filters} if filters else None,
            query={"page": next_page, "size": size},
        )
        return SyncPage(data, PeerMembership, None, fetch_next)

    return fetch_next(page)


class PeerMembers:
    """
    The members of a peer.

    A peer's members are peers whose knowledge reads of that peer aggregate.
    Chat, representation, context, card, and search that name the peer as their
    target include each member's own knowledge, attributed to the member, down
    to the requested ``depth`` (1 by default). No conclusions are copied:
    removing a member simply stops reads from including it.

    Typically accessed via `peer.members`.

    Example:
        ```python
        team = honcho.peer("engineering-team")
        team.members.add("alice", metadata={"role": "lead"})
        team.members.add(bob)
        team.members.list()

        # Async operations via .aio accessor
        await team.members.aio.add("carol")
        ```
    """

    _honcho: "Honcho"
    workspace_id: str
    peer_id: str

    def __init__(self, honcho: "Honcho", workspace_id: str, peer_id: str) -> None:
        """
        Initialize a PeerMembers view.

        Args:
            honcho: The Honcho client instance
            workspace_id: The workspace ID
            peer_id: The ID of the peer whose members this view manages
        """
        self._honcho = honcho
        self.workspace_id = workspace_id
        self.peer_id = peer_id

    @property
    def aio(self) -> "PeerMembersAio":
        """
        Access async versions of all PeerMembers methods.

        Example:
            ```python
            await team.members.aio.add("alice")
            members = await team.members.aio.list()
            ```
        """
        # Import here to avoid circular import (aio.py imports from this module)
        from .aio import PeerMembersAio

        return PeerMembersAio(self)

    def add(
        self,
        peer: str | PeerBase,
        metadata: dict[str, Any] | None = None,
    ) -> PeerMembership:
        """
        Add a peer as a member of this peer.

        Both peers are created if they do not exist. Adding an existing member
        again returns the existing membership, with its metadata replaced when
        ``metadata`` is given. Requires a workspace- or admin-level key: a
        member's knowledge becomes readable by anyone who can read this peer.

        Args:
            peer: The peer to add, as an ID string or a Peer object
            metadata: Optional application-owned metadata for the membership,
                such as a role

        Returns:
            The membership

        Raises:
            ConflictError: If the membership would create a cycle.
            UnprocessableEntityError: If either peer is a scope, the peer names
                itself, or this peer already has the maximum number of members.
        """
        self._honcho._ensure_workspace()
        data = self._honcho._http.post(
            routes.peer_members(self.workspace_id, self.peer_id),
            body=membership_body(peer, metadata),
        )
        return PeerMembership.model_validate(data)

    def list(
        self,
        filters: dict[str, Any] | None = None,
        *,
        page: int = 1,
        size: int = 50,
    ) -> SyncPage[PeerMembership]:
        """
        List this peer's members, oldest membership first.

        Args:
            filters: Optional filter criteria on the member peers, in the same
                shape as ``honcho.peers()``.
            page: Page number (1-indexed). Default: 1.
            size: Number of items per page. Default: 50.

        Returns:
            A paginated list of memberships
        """
        return list_memberships(
            self._honcho,
            routes.peer_members_list(self.workspace_id, self.peer_id),
            filters,
            page=page,
            size=size,
        )

    def remove(self, peer: str | PeerBase) -> None:
        """
        Remove a member from this peer.

        Removing a peer that is not a member is a no-op. No conclusions change:
        the member keeps everything it knows, and reads of this peer stop
        including it.

        Args:
            peer: The member to remove, as an ID string or a Peer object

        Raises:
            ValueError: If the member ID is malformed.
        """
        self._honcho._ensure_workspace()
        self._honcho._http.delete(
            routes.peer_member(self.workspace_id, self.peer_id, resolve_member_id(peer))
        )

    def __repr__(self) -> str:
        return (
            f"PeerMembers(workspace_id={self.workspace_id!r}, peer_id={self.peer_id!r})"
        )
