"""
Resource creation events for Honcho telemetry.

Emitted from the crud get-or-create functions after commit, only when a row
was actually inserted. Emitting there (rather than from the routers) covers
implicit creation too: posting messages creates the session and its peers,
and creating a session creates its workspace and scopes.
"""

from typing import ClassVar

from pydantic import Field

from src.telemetry.events.base import BaseEvent


class WorkspaceCreatedEvent(BaseEvent):
    """Emitted when a workspace is created."""

    _event_type: ClassVar[str] = "workspace.created"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")

    def get_resource_id(self) -> str:
        return self.workspace_name


class PeerCreatedEvent(BaseEvent):
    """Emitted once per peer created. Scope peers emit ScopeCreatedEvent instead."""

    _event_type: ClassVar[str] = "peer.created"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    peer_name: str = Field(..., description="Peer name")

    def get_resource_id(self) -> str:
        return f"{self.workspace_name}:{self.peer_name}"


class SessionCreatedEvent(BaseEvent):
    """Emitted when a session is created."""

    _event_type: ClassVar[str] = "session.created"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    session_name: str = Field(..., description="Session name")

    def get_resource_id(self) -> str:
        return f"{self.workspace_name}:{self.session_name}"


class ScopeCreatedEvent(BaseEvent):
    """Emitted once per scope created."""

    _event_type: ClassVar[str] = "scope.created"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "api"

    workspace_name: str = Field(..., description="Workspace name")
    scope_name: str = Field(..., description="Scope name (unprefixed)")

    def get_resource_id(self) -> str:
        return f"{self.workspace_name}:{self.scope_name}"


__all__ = [
    "PeerCreatedEvent",
    "ScopeCreatedEvent",
    "SessionCreatedEvent",
    "WorkspaceCreatedEvent",
]
