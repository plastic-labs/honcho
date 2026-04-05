"""Base classes for Honcho SDK entities.

This module provides base classes that contain only the essential data fields
shared by both sync and async variants of Peer and Session. These base classes
can be imported anywhere without causing circular import issues, enabling
type-safe method signatures like `str | PeerBase`.
"""

from pydantic import BaseModel, Field


class PeerBase(BaseModel):
    """Base class for Peer objects (sync and async variants).

    This class contains only the essential data fields shared by both
    Peer and AsyncPeer. Use this type in method signatures to accept
    either a peer ID string or any Peer object.

    Attributes:
        id: Unique identifier for this peer
        workspace_id: Workspace ID for scoping operations
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this peer")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )


class SessionBase(BaseModel):
    """Base class for Session objects (sync and async variants).

    This class contains only the essential data fields shared by both
    Session and AsyncSession. Use this type in method signatures to accept
    either a session ID string or any Session object.

    Attributes:
        id: Unique identifier for this session
        workspace_id: Workspace ID for scoping operations
    """

    id: str = Field(..., min_length=1, description="Unique identifier for this session")
    workspace_id: str = Field(
        ..., min_length=1, description="Workspace ID for scoping operations"
    )
