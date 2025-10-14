"""Cache key constants and utilities."""

from src.config import settings


def get_cache_namespace() -> str:
    """Get the cache namespace from settings."""
    return settings.CACHE.NAMESPACE


def get_workspace_prefix(workspace_name: str) -> str:
    """Get cache prefix for a workspace."""
    namespace = get_cache_namespace()
    return f"{namespace}:workspace:{workspace_name}"


def get_session_prefix(workspace_name: str) -> str:
    """Get cache prefix for sessions in a workspace."""
    namespace = get_cache_namespace()
    return f"{namespace}:session:{workspace_name}"


def get_peer_prefix(workspace_name: str) -> str:
    """Get cache prefix for peers in a workspace."""
    namespace = get_cache_namespace()
    return f"{namespace}:peer:{workspace_name}"


def get_workspace_cache_prefixes(workspace_name: str) -> dict[str, str]:
    """Get all cache prefixes for a workspace."""
    return {
        "workspace": get_workspace_prefix(workspace_name),
        "session": get_session_prefix(workspace_name),
        "peer": get_peer_prefix(workspace_name),
    }
