"""Cache module for Honcho."""

from .client import (
    close_cache,
    init_cache,
    is_enabled,
)
from .utils import (
    get_cache_namespace,
)

__all__ = [
    "close_cache",
    "get_cache_namespace",
    "init_cache",
    "is_enabled",
]
