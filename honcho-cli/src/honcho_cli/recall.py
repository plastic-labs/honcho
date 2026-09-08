"""Recall-boundary flags shared by CLI chat (and later representation / search).

Flag shape (DEV-2690 / DEV-2472):

- ``--scope <name>``: repeatable, or comma-separated. One name is that scope's
  reasoned view; several names are an explicit-only allowlist of their sessions.
- ``--sessions <id,...>``: repeatable or comma-separated session-ID allowlist
  (explicit conclusions only). Peer chat only — workspace chat has no SDK
  ``sessions`` option.
- ``-s`` / ``--session``, ``--scope``, and ``--sessions`` are mutually exclusive.
"""

from __future__ import annotations

import typer

from honcho_cli.output import print_error
from honcho_cli.validation import validate_resource_id


def parse_csv_repeatable(values: list[str] | None, *, kind: str) -> list[str] | None:
    """Flatten repeatable and/or comma-separated flag values, de-duped in order."""
    if not values:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        for part in raw.split(","):
            name = part.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(validate_resource_id(name, kind))
    return out or None


def scope_for_sdk(names: list[str] | None) -> str | list[str] | None:
    """One name stays a string (full scope view); several become a list (allowlist)."""
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return names


def reject_incompatible_recall(
    *,
    session_id: str | None,
    scope: list[str] | None,
    sessions: list[str] | None = None,
) -> None:
    """Exit if more than one of -s / --scope / --sessions is set."""
    present: list[str] = []
    if session_id:
        present.append("-s/--session")
    if scope:
        present.append("--scope")
    if sessions:
        present.append("--sessions")
    if len(present) > 1:
        print_error(
            "INCOMPATIBLE_FLAGS",
            f"{' and '.join(present)} are mutually exclusive",
        )
        raise typer.Exit(1)
