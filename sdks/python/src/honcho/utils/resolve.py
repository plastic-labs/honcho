"""ID resolution utilities for the Honcho Python SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from ..base import PeerBase, SessionBase


@overload
def resolve_id(obj: None) -> None: ...


@overload
def resolve_id(obj: str) -> str: ...


@overload
def resolve_id(obj: "PeerBase | SessionBase") -> str: ...


def resolve_id(obj: "str | PeerBase | SessionBase | None") -> str | None:
    """
    Resolve an ID from a string, PeerBase, SessionBase, or None.

    This utility function extracts the ID from an object that may be:
    - A string (returned as-is)
    - An object with an `id` attribute (the id is extracted)
    - None (returns None)

    Args:
        obj: A string ID, an object with an `id` attribute (like Peer or Session), or None

    Returns:
        The resolved string ID, or None if input is None

    Example:
        >>> resolve_id("user-123")
        'user-123'
        >>> resolve_id(peer)  # where peer.id == "user-123"
        'user-123'
        >>> resolve_id(None)
        None
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return obj.id
