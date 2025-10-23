"""Cache module for Honcho."""

from .client import (
    close_cache,
    init_cache,
    is_enabled,
)

__all__ = [
    "close_cache",
    "init_cache",
    "is_enabled",
]
