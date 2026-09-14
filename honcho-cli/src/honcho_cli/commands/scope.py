"""Scope commands: list, create, inspect, sessions, add-sessions, remove-session, status.

A scope is a named set of sessions that acts as a recall boundary. Reads that
accept ``--scope`` (``honcho peer chat``, ``honcho workspace chat``) answer from
what happened inside the scope's member sessions only.

Every command except ``create`` fails closed on an unknown scope name: the SDK's
``honcho.scope()`` is get-or-create, so it is never used to *look up* a scope
here — a typo in ``add-sessions`` must not silently create a new scope and
start a backfill into it.
"""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from honcho import Honcho, NotFoundError, Scope

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.commands.workspace import _handle_error, _raw_list
from honcho_cli.common import add_common_options, get_client, handle_cmd_flags
from honcho_cli.output import print_error, print_result
from honcho_cli.recall import parse_csv_repeatable
from honcho_cli.validation import validate_resource_id

app = typer.Typer(
    cls=HonchoTyperGroup,
    help="List, create, inspect, and manage scopes — named session sets that bound recall.",
)
add_common_options(app)


def _scope_dict(scope: Scope) -> dict:
    return {
        "id": scope.id,
        "metadata": scope.metadata,
        "created_at": str(scope.created_at),
    }


def _find_scope(client: Honcho, name: str) -> Scope:
    """Resolve an existing scope by name, or exit with SCOPE_NOT_FOUND.

    Walks ``client.scopes()`` rather than calling ``client.scope(name)`` because
    the latter creates the scope when it is missing.
    """
    for scope in client.scopes():
        if scope.id == name:
            return scope
    print_error(
        "SCOPE_NOT_FOUND",
        f"Scope '{name}' not found. Run `honcho scope list` to see existing scopes, "
        f"or `honcho scope create {name}` to create it.",
        {"scope": name},
    )
    raise typer.Exit(1)


def _handle_scope_error(e: Exception, name: str) -> None:
    """Like ``_handle_error`` but keeps the server message on 404s.

    Membership calls 404 on a missing *session*, not a missing scope, so
    rewriting the message as ``Scope 'x' not found`` would point at the wrong
    resource.
    """
    if isinstance(e, typer.Exit):
        # Already reported (e.g. SCOPE_NOT_FOUND from _find_scope); don't double-print.
        raise e
    if isinstance(e, NotFoundError):
        print_error("NOT_FOUND", str(e), {"scope": name})
        raise typer.Exit(1)
    if isinstance(e, ValueError):
        print_error("INVALID_ARGUMENT", str(e), {"scope": name})
        raise typer.Exit(1)
    _handle_error(e, "scope", name)


def _backfill_summary(status: dict) -> dict:
    counts = {"pending": 0, "completed": 0, "failed": 0}
    for job in status.values():
        counts[job.state] = counts.get(job.state, 0) + 1
    return counts


@app.command("list")
def list_scopes(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List scopes in the workspace."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()

    try:
        items = [_scope_dict(s) for s in client.scopes()]
        print_result(items, columns=["id", "metadata", "created_at"], title="Scopes")
    except Exception as e:
        _handle_error(e, "scope", "list")


@app.command("create")
def create_scope(
    name: str = typer.Argument(help="Scope name to create or get (unprefixed, unique in the workspace)"),
    sessions: Optional[List[str]] = typer.Option(
        None,
        "--sessions",
        help="Sessions to add (repeat or comma-separate). History backfills asynchronously; see `honcho scope status`.",
    ),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the scope"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create or get a scope, optionally adding sessions to it."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    session_ids = parse_csv_repeatable(sessions, kind="session", flag="--sessions")

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    client, config = get_client()

    try:
        scope = client.scope(name, metadata=parsed_metadata)
        result = _scope_dict(scope)
        if session_ids:
            scope.add_sessions(session_ids)
            result["added_sessions"] = session_ids
        print_result(result)
    except Exception as e:
        _handle_scope_error(e, name)


@app.command()
def inspect(
    name: str = typer.Argument(help="Scope name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a scope: metadata, member sessions, and backfill state."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    client, config = get_client()

    try:
        scope = _find_scope(client, name)
        raw_sessions = _raw_list(scope.sessions())
        backfill = scope.status()
        result = _scope_dict(scope)
        result["session_count"] = len(raw_sessions)
        result["sessions"] = [s.id for s in raw_sessions]
        result["backfill"] = _backfill_summary(backfill)
        print_result(result)
    except Exception as e:
        _handle_scope_error(e, name)


@app.command("sessions")
def scope_sessions(
    name: str = typer.Argument(help="Scope name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List the sessions that are members of a scope (longest-standing first)."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    client, config = get_client()

    try:
        scope = _find_scope(client, name)
        items = [
            {
                "id": s.id,
                "is_active": s.is_active,
                "metadata": s.metadata,
                "created_at": str(s.created_at),
            }
            for s in _raw_list(scope.sessions())
        ]
        print_result(items, columns=["id", "is_active", "metadata", "created_at"], title=f"Scope sessions ({name})")
    except Exception as e:
        _handle_scope_error(e, name)


@app.command("add-sessions")
def add_sessions(
    name: str = typer.Argument(help="Scope name"),
    session_ids: List[str] = typer.Argument(help="Session IDs (space- or comma-separated, max 100)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Add existing sessions to a scope. History is backfilled asynchronously — poll `honcho scope status`."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    ids = parse_csv_repeatable(session_ids, kind="session", flag="session_ids")
    client, config = get_client()

    try:
        scope = _find_scope(client, name)
        scope.add_sessions(ids)
        print_result({"scope": name, "added_sessions": ids})
    except Exception as e:
        _handle_scope_error(e, name)


@app.command("remove-session")
def remove_session(
    name: str = typer.Argument(help="Scope name"),
    session_id: str = typer.Argument(help="Session ID to remove"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Remove a session from a scope. Its conclusions are reconciled out asynchronously."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    sid = validate_resource_id(session_id, "session")
    client, config = get_client()

    try:
        scope = _find_scope(client, name)
        scope.remove_session(sid)
        print_result({"scope": name, "removed_session": sid})
    except Exception as e:
        _handle_scope_error(e, name)


@app.command()
def status(
    name: str = typer.Argument(help="Scope name"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Per-session backfill state. Recall through the scope is complete once nothing is pending."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    name = validate_resource_id(name, "scope")
    client, config = get_client()

    try:
        scope = _find_scope(client, name)
        backfill = scope.status()
        items = [
            {
                "session_id": sid,
                "state": job.state,
                "docs_copied": job.docs_copied,
                "updated_at": str(job.updated_at),
            }
            for sid, job in backfill.items()
        ]
        print_result(
            {"scope": name, "summary": _backfill_summary(backfill), "sessions": items},
        )
    except Exception as e:
        _handle_scope_error(e, name)
