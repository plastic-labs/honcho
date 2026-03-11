"""Input hardening: validate resource IDs and workspace names.

Agents hallucinate bad IDs. Catch them early with clear errors.
"""

from __future__ import annotations

import re
import sys

UNSAFE_CHARS = re.compile(r'[?#%\x00-\x1f\x7f/\\]')
WORKSPACE_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def validate_resource_id(value: str, resource_type: str = "resource") -> str:
    """Validate a resource ID. Returns the value if valid, raises SystemExit on invalid."""
    if not value:
        _fail(
            "EMPTY_ID",
            f"Empty {resource_type} ID provided",
            {resource_type: ""},
        )

    if UNSAFE_CHARS.search(value):
        _fail(
            "INVALID_ID",
            f"Invalid {resource_type} ID: contains unsafe characters (?, #, %, control chars, path separators)",
            {resource_type: value},
        )

    if ".." in value:
        _fail(
            "INVALID_ID",
            f"Invalid {resource_type} ID: contains path traversal",
            {resource_type: value},
        )

    return value


def validate_workspace_name(value: str) -> str:
    """Validate workspace name: alphanumeric, hyphens, underscores."""
    if not value:
        _fail("EMPTY_WORKSPACE", "Empty workspace name", {"workspace": ""})

    if not WORKSPACE_NAME_RE.match(value):
        _fail(
            "INVALID_WORKSPACE",
            "Workspace name must be alphanumeric with hyphens/underscores only",
            {"workspace": value},
        )

    return value


def _fail(code: str, message: str, details: dict) -> None:
    """Print structured error and exit."""
    from honcho_cli.output import print_error

    print_error(code, message, details)
    raise SystemExit(1)
