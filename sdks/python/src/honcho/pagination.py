# Re-export pagination classes from http module for backwards compatibility
from .http.pagination import AsyncPage, SyncPage

__all__ = ["SyncPage", "AsyncPage"]
