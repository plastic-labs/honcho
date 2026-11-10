"""Workspace commands: list, inspect, create, delete, search, chat, queue-status, replay."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import typer

from honcho import (
    APIError,
    AuthenticationError,
    Honcho,
    NotFoundError,
    PermissionDeniedError,
    ServerError,
)

from honcho_cli import replay as replay_lib
from honcho_cli import replay_view
from honcho_cli.config import identity_headers
from honcho_cli.output import print_error, print_json, print_result, status, use_json
from honcho_cli.recall import parse_csv_repeatable, reject_incompatible_recall, scope_for_sdk
from honcho_cli.validation import validate_resource_id

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.common import add_common_options, format_evidence, get_client, get_endpoint_client, get_flag_overrides, get_resolved_config, handle_cmd_flags, same_instance

app = typer.Typer(cls=HonchoTyperGroup, help="List, create, inspect, chat, delete, and search workspaces.")
add_common_options(app)


def _get_workspace_id(workspace_id: str | None) -> str:

    config = get_resolved_config()
    wid = workspace_id or config.workspace_id
    if not wid:
        print_error("NO_WORKSPACE", "No workspace ID provided. Pass --workspace/-w or set HONCHO_WORKSPACE_ID.")
        raise typer.Exit(1)
    return validate_resource_id(wid, "workspace")


def _raw_list(page) -> list:
    """Collect all raw API response items across all pages of a SyncPage."""
    items = list(page._raw_items)
    while page.has_next_page():
        page = page.get_next_page()
        if page is None:
            break
        items.extend(page._raw_items)
    return items


@app.command("list")
def list_workspaces(
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List all accessible workspaces."""

    handle_cmd_flags(json_output=json_output)
    client, config = get_client(require_workspace=False)

    try:
        workspaces = list(client.workspaces())
        items = [{"id": w} for w in workspaces]
        print_result(items, columns=["id"], title="Workspaces")
    except Exception as e:
        _handle_error(e, "workspace", "list")


@app.command("create")
def create_workspace(
    workspace_id: str = typer.Argument(help="Workspace ID to create or get"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the workspace"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create or get a workspace."""
    handle_cmd_flags(json_output=json_output)
    wid = validate_resource_id(workspace_id, "workspace")
    client, config = get_client(require_workspace=False)
    ws_client = _with_workspace(client, wid)

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    try:
        # Trigger get-or-create via the workspace ensure mechanism
        ws_client.get_configuration()
        result: dict[str, object] = {"workspace_id": wid}
        if parsed_metadata is not None:
            ws_client.set_metadata(parsed_metadata)
            result["metadata"] = parsed_metadata
        print_result(result)
    except Exception as e:
        _handle_error(e, "workspace", wid)


@app.command()
def inspect(
    workspace_id: Optional[str] = typer.Argument(None, help="Workspace ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a workspace: peers, sessions, config."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)

    wid = _get_workspace_id(workspace_id)
    client, config = get_client(require_workspace=False)

    # Override workspace if positional arg given
    if workspace_id:
        client = _with_workspace(client, workspace_id)

    try:
        ws_config = client.get_configuration()
        ws_metadata = client.get_metadata()
        peer_page = client.peers()
        session_page = client.sessions()

        raw_peers = peer_page._raw_items
        raw_sessions = session_page._raw_items

        result = {
            "workspace_id": wid,
            "metadata": ws_metadata,
            "configuration": _config_to_dict(ws_config) if ws_config else None,
            "peer_count": peer_page.total,
            "session_count": session_page.total,
            "peers": [
                {"id": p.id, "metadata": p.metadata, "created_at": str(p.created_at)}
                for p in raw_peers[:20]
            ],
            "sessions": [
                {"id": s.id, "is_active": s.is_active, "metadata": s.metadata, "created_at": str(s.created_at)}
                for s in raw_sessions[:20]
            ],
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "workspace", wid)


@app.command()
def delete(
    workspace_id: str = typer.Argument(help="Workspace ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt (for scripted/agent use)"),
    cascade: bool = typer.Option(False, "--cascade", help="Delete all sessions before deleting the workspace"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete a workspace. Use --dry-run first to see what will be deleted.

    Requires --yes to skip confirmation, or will prompt interactively.
    If sessions exist, requires --cascade to delete them first.
    """

    handle_cmd_flags(json_output=json_output)

    validate_resource_id(workspace_id, "workspace")
    # workspace_id is a required positional and we rebuild the client with it
    # immediately, so the default-workspace guard isn't needed here.
    client, config = get_client(require_workspace=False)
    ws_client = _with_workspace(client, workspace_id)

    # Verify workspace exists before prompting for confirmation
    try:
        ws_client.get_metadata()
    except Exception as e:
        _handle_error(e, "workspace", workspace_id)
        return

    # Always fetch sessions for dry-run or cascade
    raw_sessions = _raw_list(ws_client.sessions()) if (dry_run or cascade) else []

    if dry_run:
        print_result({
            "dry_run": True,
            "workspace_id": workspace_id,
            "sessions_to_delete": len(raw_sessions),
            "session_ids": [s.id for s in raw_sessions],
            "warning": "This action cannot be undone.",
        })
        return

    if not yes:
        if cascade and raw_sessions:
            typer.confirm(
                f"Delete workspace '{workspace_id}' and {len(raw_sessions)} session(s)? This cannot be undone.",
                abort=True,
            )
        else:
            typer.confirm(f"Delete workspace '{workspace_id}'? This cannot be undone.", abort=True)

    try:
        deleted_sessions = []
        if cascade and raw_sessions:
            for s in raw_sessions:
                ws_client.session(s.id).delete()
                deleted_sessions.append(s.id)
                status(f"Deleted session '{s.id}'")

        ws_client.delete_workspace(workspace_id)
        status(f"Workspace '{workspace_id}' deletion accepted (processing in background)")
        result = {"deleted_workspace": workspace_id, "status": "accepted"}
        if cascade:
            result["deleted_sessions"] = deleted_sessions
        print_result(result)
    except Exception as e:
        _handle_error(e, "workspace", workspace_id)


@app.command()
def chat(
    query: str = typer.Argument(help="Question to ask about the workspace"),
    reasoning: Optional[str] = typer.Option(None, "--reasoning", "-r", help="Reasoning level: minimal, low, medium, high, max"),
    scope: Optional[list[str]] = typer.Option(
        None,
        "--scope",
        help="Recall only from this scope. Repeat or comma-separate for several (explicit conclusions only). Excludes -s.",
    ),
    evidence: bool = typer.Option(
        False,
        "--evidence",
        help="Also report what the answer was built from: the conclusions and messages read and the tools called.",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Query the dialectic across all peers in the workspace."""

    _REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")
    if reasoning and reasoning not in _REASONING_LEVELS:
        print_error("INVALID_REASONING", f"--reasoning must be one of: {', '.join(_REASONING_LEVELS)}")
        raise typer.Exit(1)

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    wid = _get_workspace_id(None)
    client, config = get_client()
    scope_names = parse_csv_repeatable(scope, kind="scope", flag="--scope")
    session_id = config.session_id or None
    reject_incompatible_recall(
        session_id=session_id,
        scope=scope_names,
        session_from_env=not get_flag_overrides()["session"],
    )

    chat_kwargs: dict[str, object] = {
        "session": session_id,
        "reasoning_level": reasoning or None,
    }
    scope_arg = scope_for_sdk(scope_names)
    if scope_arg is not None:
        chat_kwargs["scope"] = scope_arg

    try:
        if not evidence:
            response = client.chat(query, **chat_kwargs)
            print_result({"workspace_id": wid, "query": query, "response": response})
            return
        result = client.chat(query, include_evidence=True, **chat_kwargs)
        print_result(
            {
                "workspace_id": wid,
                "query": query,
                "response": result.content,
                "evidence": format_evidence(result.evidence),
            }
        )
    except Exception as e:
        _handle_chat_error(e, "workspace", wid)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    limit: int = typer.Option(10, help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search messages across workspace."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)

    wid = _get_workspace_id(None)
    client, config = get_client()

    try:
        results = client.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "content": m.content if use_json() else m.content[:200],
                "peer_id": m.peer_id,
                "session_id": m.session_id,
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "peer_id", "session_id", "content"], title=f"Search: {query}")
    except Exception as e:
        _handle_error(e, "workspace", wid)


@app.command("queue-status")
def queue_status(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    observer: Optional[str] = typer.Option(None, help="Filter by observer peer"),
    sender: Optional[str] = typer.Option(None, help="Filter by sender peer"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get queue processing status."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)

    _get_workspace_id(None)
    client, config = get_client()

    try:
        result = client.queue_status(observer=observer, sender=sender, session=config.session_id or None)
        print_result(result.model_dump(mode="json"))
    except Exception as e:
        _handle_error(e, "queue", "status")


@app.command()
def replay(
    source: str = typer.Option(..., "--from", help="Source workspace ID"),
    dest: Optional[str] = typer.Option(
        None,
        "--to",
        help="Destination workspace ID. Defaults to the --from ID, which needs a different destination instance.",
    ),
    from_url: Optional[str] = typer.Option(
        None, "--from-url", envvar="HONCHO_FROM_URL",
        help="Source Honcho URL, when it is not the configured instance. Its API key is read from HONCHO_FROM_API_KEY.",
    ),
    to_url: Optional[str] = typer.Option(
        None, "--to-url", envvar="HONCHO_TO_URL",
        help="Destination Honcho URL, when it is not the configured instance. Its API key is read from HONCHO_TO_API_KEY.",
    ),
    num_sessions: Optional[int] = typer.Option(
        None, "--num-sessions", min=1,
        help="Replay only the N sessions with the most recent message",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what a replay would do with each session, without writing anything"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Replay a workspace's sessions into another workspace, on this or another instance.

    Copies workspace, peer and session settings, then re-sends every message in
    order with its original timestamp. The destination derives its own
    conclusions; conclusions and peer cards are not copied, so the ones written
    through the API are not recreated. Rerunning resumes: messages already in
    the destination are skipped.

    Each side uses the configured instance unless its URL is given. A side on
    another instance needs its own API key, read from HONCHO_FROM_API_KEY /
    HONCHO_TO_API_KEY; the configured credentials are never sent to it.
    """
    handle_cmd_flags(json_output=json_output)
    # Keys come from the environment only: a key passed as an argument shows up
    # in process listings and shell history.
    from_api_key = os.environ.get("HONCHO_FROM_API_KEY") or None
    to_api_key = os.environ.get("HONCHO_TO_API_KEY") or None
    source_id = validate_resource_id(source, "workspace")
    dest_id = validate_resource_id(dest or source, "workspace")

    source_client = get_endpoint_client(workspace_id=source_id, base_url=from_url, api_key=from_api_key)
    dest_client = get_endpoint_client(workspace_id=dest_id, base_url=to_url, api_key=to_api_key)
    source_ref = {"url": source_client.base_url, "workspace_id": source_id}

    if source_id == dest_id and same_instance(source_client.base_url, dest_client.base_url):
        print_error(
            "SAME_WORKSPACE",
            "Source and destination are the same workspace. Pass a different --to, or --to-url for another instance.",
            {"workspace": source_id, "url": source_client.base_url},
        )
        raise typer.Exit(1)

    started = time.monotonic()
    replayer = replay_lib.Replayer(source_client, dest_client)
    all_peers = num_sessions is None

    with replay_view.Reporter() as progress:
        # Read the source, and look at the destination without writing to it.
        try:
            progress.phase(f"Reading source workspace '{source_id}'")
            metadata, configuration = replay_lib.read_workspace(source_client)
            plans = replay_lib.list_sessions(source_client)
            sessions_in_source = len(plans)
            if num_sessions is not None or dry_run:
                replay_lib.probe_sessions(plans, progress.tick)
            if num_sessions is not None:
                plans = replay_lib.select_most_recent(plans, num_sessions)
            not_copied = {
                "conclusions_in_source": replay_lib.count_conclusions(source_client),
                "note": replay_lib.NOT_COPIED_NOTE,
            }
        except replay_lib.NothingToReplay as e:
            print_error("NOTHING_TO_REPLAY", str(e), {"workspace": source_id})
            raise typer.Exit(1) from e
        except Exception as e:
            _handle_error(e, "workspace", source_id)
            return

        try:
            progress.phase(f"Checking destination workspace '{dest_id}'")
            dest_sessions = replayer.destination_session_count()
            dest_ref = {"url": dest_client.base_url, "workspace_id": dest_id, "sessions": dest_sessions}
            if dry_run:
                replayer.check_sessions(plans, progress.tick)
                peer_ids = replayer.peers_for(plans, all_peers=all_peers)
                peers_in_destination = replayer.peers_in_destination(peer_ids)
        except replay_lib.ReadOnlyUnsupported as e:
            print_error("SDK_UNSUPPORTED", str(e), {})
            raise typer.Exit(1) from e
        except Exception as e:
            _handle_error(e, "workspace", dest_id)
            return

        if not dry_run:
            progress.line(
                f"Replaying {len(plans)} of {replay_view.plural(sessions_in_source, 'session')} "
                f"from '{source_id}' ({source_client.base_url}) "
                f"to '{dest_id}' ({dest_client.base_url})"
                + (f". The destination already has {replay_view.plural(dest_sessions, 'session')}." if dest_sessions else "")
            )
            progress.phase("Copying peers and sessions")
            try:
                result = replay_lib.replay(
                    replayer,
                    plans,
                    workspace_metadata=metadata,
                    workspace_configuration=configuration,
                    all_peers=all_peers,
                    on_progress=progress.line,
                    on_tick=progress.tick,
                )
            except Exception as e:
                _handle_error(e, "workspace", dest_id)
                return

    if dry_run:
        report = {
            "dry_run": True,
            "source": source_ref,
            "destination": dest_ref,
            "sessions_in_source": sessions_in_source,
            "sessions_to_replay": len(plans),
            "actions": {
                action: sum(1 for plan in plans if plan.action == action)
                for action in ("create", "resume", "up_to_date", "conflict")
            },
            "messages_in_sessions": sum(plan.message_count or 0 for plan in plans),
            "messages_to_send": sum(plan.messages_to_send for plan in plans),
            "peers_to_copy": len(peer_ids),
            "peers_in_destination": len(peers_in_destination),
            "workspace_settings_copied": dest_sessions == 0,
            "not_copied": not_copied,
            "sessions": [plan.to_dict() for plan in plans],
        }
        if use_json():
            print_json(report)
        else:
            replay_view.print_plan(report)
        return

    messages_created = sum(s.messages_created for s in result.sessions)
    report = {
        "source": source_ref,
        "destination": dest_ref,
        "sessions_in_source": sessions_in_source,
        "sessions_replayed": len(result.sessions),
        "sessions_failed": len(result.failed),
        "actions": {
            action: sum(1 for s in result.sessions if s.action == action)
            for action in ("created", "resumed", "up_to_date")
        },
        "messages_created": messages_created,
        "messages_skipped": sum(s.messages_skipped for s in result.sessions),
        "peers_created": result.peers_created,
        "peers_existing": result.peers_existing,
        "workspace_settings_copied": result.workspace_copied,
        "not_copied": not_copied,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "check_derivation": _queue_status_command(dest_client.base_url, dest_id) if messages_created else None,
        "sessions": [s.to_dict() for s in result.sessions],
        "failed": result.failed,
    }
    if use_json():
        print_json(report)
    else:
        replay_view.print_summary(report)
    if result.failed:
        raise typer.Exit(1)


def _queue_status_command(dest_url: str, dest_id: str) -> str:
    """The command that shows the destination's derivation queue after a replay."""
    command = f"honcho workspace queue-status -w {dest_id}"
    if same_instance(dest_url, get_resolved_config().base_url):
        return command
    return f'HONCHO_BASE_URL={dest_url} HONCHO_API_KEY="$HONCHO_TO_API_KEY" {command}'


def _with_workspace(client, workspace_id: str):
    """Return a new client pointed at a different workspace."""
    return Honcho(
        base_url=str(client.base_url),
        api_key=client._http.api_key if hasattr(client._http, "api_key") else None,
        workspace_id=workspace_id,
        default_headers=identity_headers(),
    )


def _config_to_dict(config) -> dict:
    """Convert a config object to a dict, handling nested objects."""
    if hasattr(config, "__dict__"):
        result = {}
        for k, v in config.__dict__.items():
            if k.startswith("_"):
                continue
            result[k] = _config_to_dict(v) if hasattr(v, "__dict__") and not isinstance(v, str) else v
        return result
    return config


def _handle_chat_error(e: Exception, resource: str, resource_id: str) -> None:
    """Like ``_handle_error``, but keep the server/SDK message.

    Chat 404s include scope misses (``Scope X not found in workspace Y``). Rewriting
    those as ``Workspace 'id' not found`` would hide the real error.
    """
    if isinstance(e, NotFoundError):
        print_error("NOT_FOUND", str(e), {resource: resource_id})
        raise typer.Exit(1)
    if isinstance(e, ValueError):
        print_error("INVALID_ARGUMENT", str(e), {resource: resource_id})
        raise typer.Exit(1)
    _handle_error(e, resource, resource_id)


def _handle_error(e: Exception, resource: str, resource_id: str) -> None:
    """Handle SDK exceptions with structured error output.

    Dispatches on the SDK's typed exception hierarchy
    (``honcho.http.exceptions``) and falls back to its ``status`` field for
    any APIError subclass we don't enumerate. Substring matching on the
    message is used only as a last-ditch fallback for non-SDK exceptions.
    """
    if isinstance(e, NotFoundError):
        print_error(
            f"{resource.upper()}_NOT_FOUND",
            f"{resource.title()} '{resource_id}' not found",
            {resource: resource_id},
        )
        raise typer.Exit(1)
    if isinstance(e, AuthenticationError):
        print_error("AUTH_ERROR", f"Authentication failed: {e}", {})
        raise typer.Exit(3)
    if isinstance(e, PermissionDeniedError):
        print_error("PERMISSION_ERROR", f"Permission denied: {e}", {})
        raise typer.Exit(3)
    if isinstance(e, ServerError):
        print_error("SERVER_ERROR", f"Server error: {e}", {resource: resource_id})
        raise typer.Exit(2)
    if isinstance(e, APIError):
        # Catch-all for typed API errors we haven't special-cased
        # (BadRequest, Conflict, UnprocessableEntity, RateLimit, ...).
        print_error(
            "API_ERROR",
            f"API error ({e.status}): {e}",
            {resource: resource_id, "status": e.status},
        )
        raise typer.Exit(1)
    print_error("UNKNOWN_ERROR", str(e), {resource: resource_id})
    raise typer.Exit(1)
