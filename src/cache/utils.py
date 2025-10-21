"""Cache key constants and utilities."""

from pydantic import BaseModel

from src.config import settings


class CacheKey(BaseModel):
    namespace: str
    workspace_name: str
    session_name: str | None
    peer_name: str | None

    def to_string(self) -> str:
        """Generate the appropriate cache key based on the resource type."""
        if self.peer_name is not None:
            return self.get_peer_cache_key()
        elif self.session_name is not None:
            return self.get_session_cache_key()
        else:
            return self.get_workspace_cache_key()

    @classmethod
    def from_cache_key(cls, cache_key: str) -> "CacheKey":
        """Parse a cache key and extract the workspace name and resource type."""
        parts = cache_key.split(":")
        if (
            len(parts) < 3
            or parts[0] != settings.CACHE.NAMESPACE
            or parts[1] != "workspace"
        ):
            raise ValueError(f"Invalid cache key format: {cache_key}")

        workspace_name = parts[2]
        session_name = None
        peer_name = None

        if len(parts) >= 5:
            if parts[3] == "session":
                session_name = parts[4]
            elif parts[3] == "peer":
                peer_name = parts[4]

        return cls(
            namespace=settings.CACHE.NAMESPACE,
            workspace_name=workspace_name,
            session_name=session_name,
            peer_name=peer_name,
        )

    def get_workspace_cache_key(self) -> str:
        """Get the workspace cache key."""
        return f"{self.namespace}:workspace:{self.workspace_name}"

    def get_session_cache_key(self) -> str:
        """Get the session cache key."""
        return f"{self.namespace}:workspace:{self.workspace_name}:session:{self.session_name}"

    def get_peer_cache_key(self) -> str:
        """Get the peer cache key."""
        return f"{self.namespace}:workspace:{self.workspace_name}:peer:{self.peer_name}"


def get_cache_namespace() -> str:
    """Get the cache namespace from settings."""
    return settings.CACHE.NAMESPACE
