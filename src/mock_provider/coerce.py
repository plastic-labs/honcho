"""Typed narrowing for values decoded from JSON.

``isinstance(value, dict)`` on an ``Any`` narrows to ``dict[Unknown, Unknown]``,
which spreads unknown types through everything downstream. These helpers narrow
and pin the element types in one step.
"""

from __future__ import annotations

from typing import Any, cast


def as_dict(value: object) -> dict[str, Any] | None:
    """The value as a JSON object, or None if it is not one."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def as_list(value: object) -> list[Any] | None:
    """The value as a JSON array, or None if it is not one."""
    return cast("list[Any]", value) if isinstance(value, list) else None


def as_str(value: object) -> str | None:
    """The value as a JSON string, or None if it is not one."""
    return value if isinstance(value, str) else None


def as_int(value: object) -> int | None:
    """The value as a JSON integer, or None if it is not one.

    ``bool`` is excluded: it is an ``int`` subclass, and a JSON ``true`` reaching
    a size or dimension field is a malformed request, not the number one.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else None
