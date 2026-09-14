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


def parse_csv_repeatable(
    values: list[str] | None, *, kind: str, flag: str
) -> list[str] | None:
    """Flatten repeatable and/or comma-separated flag values, de-duped in order.

    A flag that was given but holds no names (``--scope ""``, ``--sessions ,``)
    is an error rather than "no bound": the server rejects an empty allowlist
    for the same reason, and silently widening recall to the whole workspace
    is the wrong default for a caller that built the value programmatically.
    """
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
    if not out:
        print_error(
            "EMPTY_RECALL_BOUND",
            f"{flag} was given but contains no {kind} names",
        )
        raise typer.Exit(1)
    return out


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
    session_from_env: bool = False,
) -> None:
    """Exit if more than one of -s / --scope / --sessions is set.

    ``session_from_env`` marks a session that arrived through ``HONCHO_SESSION_ID``
    rather than a typed ``-s``, so the error can name the variable the caller
    never typed and how to clear it for one command.
    """
    bounds: list[str] = []
    if scope:
        bounds.append("--scope")
    if sessions:
        bounds.append("--sessions")
    if len(bounds) > 1:
        _exit_incompatible(f"{' and '.join(bounds)} are mutually exclusive")
    if session_id and bounds:
        flag = bounds[0]
        if session_from_env:
            _exit_incompatible(
                f"{flag} was not applied: recall is already confined to session "
                f"'{session_id}' by HONCHO_SESSION_ID in your environment, and a "
                f"session cannot be combined with {flag}. If that session is not what "
                f"you meant, unset it for this command: `env -u HONCHO_SESSION_ID honcho ...`"
            )
        _exit_incompatible(f"-s/--session and {flag} are mutually exclusive")


def _exit_incompatible(message: str) -> None:
    print_error("INCOMPATIBLE_FLAGS", message)
    raise typer.Exit(1)
