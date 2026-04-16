"""Session commands: list, inspect, context, summaries, peers, search, representation, metadata."""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from honcho import HonchoError

from honcho_cli.commands.workspace import _config_to_dict, _handle_error, _raw_list
from honcho_cli.output import print_error, print_result, status, use_json
from honcho_cli.validation import validate_resource_id

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.common import add_common_options, get_client, get_resolved_config, handle_cmd_flags

app = typer.Typer(cls=HonchoTyperGroup, help="List, inspect, create, delete, and manage conversation sessions and their peers.")
add_common_options(app)


def _get_session_id(session_id: str | None) -> str:

    config = get_resolved_config()
    sid = session_id or config.session_id
    if not sid:
        print_error("NO_SESSION", "No session ID provided. Pass --session/-s or set HONCHO_SESSION_ID.")
        raise typer.Exit(1)
    return validate_resource_id(sid, "session")


@app.command("list")
def list_sessions(
    peer_id: Optional[str] = typer.Option(None, "--peer", "-p", help="Filter by peer"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List sessions in the workspace."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()

    try:
        if peer_id:
            peer = client.peer(peer_id)
            raw_sessions = _raw_list(peer.sessions())
        else:
            raw_sessions = _raw_list(client.sessions())

        items = [
            {
                "id": s.id,
                "is_active": s.is_active,
                "metadata": s.metadata,
                "created_at": str(s.created_at),
            }
            for s in raw_sessions
        ]
        print_result(items, columns=["id", "is_active", "metadata", "created_at"], title="Sessions")
    except Exception as e:
        _handle_error(e, "session", "list")


@app.command("create")
def create_session(
    session_id: str = typer.Argument(help="Session ID to create or get"),
    peers: Optional[str] = typer.Option(None, "--peers", help="Comma-separated peer IDs to add to the session"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create or get a session."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = validate_resource_id(session_id, "session")
    client, config = get_client()

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    peer_ids = [p.strip() for p in peers.split(",") if p.strip()] if peers else []
    for pid in peer_ids:
        validate_resource_id(pid, "peer")

    try:
        sess = client.session(sid, metadata=parsed_metadata)
        if peer_ids:
            sess.add_peers(peer_ids)
        result: dict[str, object] = {"session_id": sess.id}
        if parsed_metadata is not None:
            result["metadata"] = parsed_metadata
        if peer_ids:
            result["peers"] = peer_ids
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def inspect(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a session: peers, message count, summaries, config."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        peers = sess.peers()
        msg_page = sess.messages()
        summaries = sess.summaries()
        sess_config = sess.get_configuration()

        result = {
            "session_id": sid,
            "peers": [{"id": p.id} for p in peers],
            "message_count": msg_page.total,
            "summaries": {
                "short": summaries.short_summary if hasattr(summaries, "short_summary") else None,
                "long": summaries.long_summary if hasattr(summaries, "long_summary") else None,
            },
            "configuration": _config_to_dict(sess_config) if sess_config else None,
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def context(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    tokens: Optional[int] = typer.Option(None, help="Token budget"),
    summary: bool = typer.Option(True, help="Include summary"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get session context (what an agent would see)."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        ctx = sess.context(tokens=tokens, summary=summary)
        result = ctx.__dict__ if hasattr(ctx, "__dict__") else ctx
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def summaries(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get session summaries (short + long)."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        s = sess.summaries()
        result = {
            "session_id": sid,
            "short_summary": s.short_summary if hasattr(s, "short_summary") else None,
            "long_summary": s.long_summary if hasattr(s, "long_summary") else None,
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def delete(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete a session and all its data. Destructive — requires --yes or interactive confirm."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    if not yes:
        # Show a short preview so the user knows what's about to disappear.
        # Only in interactive/TTY mode — scripted (--json) callers already
        # know what they're deleting, and they still need to pass --yes.
        # Narrow the except to HonchoError so auth/network failures surface
        # before the user types 'y' on a destructive op.
        if not use_json():
            try:
                peers = sess.peers()
                msg_page = sess.messages()
                if msg_page.total is not None:
                    msg_count_str = str(msg_page.total)
                else:
                    msg_count_str = f"{len(msg_page.items)} (first page; more may exist)"
                peer_ids = [p.id for p in peers]
                typer.echo(
                    f"  session:  {sid}\n"
                    f"  peers:    {', '.join(peer_ids) if peer_ids else '(none)'}\n"
                    f"  messages: {msg_count_str}"
                )
            except HonchoError as preview_err:
                status(f"preview unavailable: {preview_err}")
        typer.confirm(f"Delete session '{sid}' and all its messages, conclusions, and queue items?", abort=True)

    try:
        sess.delete()
        status(f"Session '{sid}' deleted")
        print_result({"deleted": sid})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("peers")
def session_peers(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List peers in a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        peers = sess.peers()
        items = [{"id": p.id} for p in peers]
        print_result(items, columns=["id"], title=f"Session peers ({sid})")
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("add-peers")
def add_peers(
    session_id: str = typer.Argument(help="Session ID"),
    peer_ids: List[str] = typer.Argument(help="Peer IDs to add to the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Add peers to a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        sess.add_peers(peer_ids)
        print_result({"session_id": sid, "added_peers": peer_ids})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("remove-peers")
def remove_peers(
    session_id: str = typer.Argument(help="Session ID"),
    peer_ids: List[str] = typer.Argument(help="Peer IDs to remove from the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Remove peers from a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        sess.remove_peers(peer_ids)
        print_result({"session_id": sid, "removed_peers": peer_ids})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    limit: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search messages in a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        results = sess.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "peer_id": m.peer_id,
                "content": m.content if use_json() else m.content[:200],
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "peer_id", "content", "created_at"], title=f"Session search: {query}")
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def representation(
    peer_id: str = typer.Argument(help="Peer ID to get representation for"),
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    target: Optional[str] = typer.Option(None, help="Target peer (what peer_id knows about target)"),
    search_query: Optional[str] = typer.Option(None, help="Semantic search query to filter conclusions"),
    max_conclusions: Optional[int] = typer.Option(None, help="Maximum number of conclusions to include"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get the representation of a peer within a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        result = sess.representation(
            peer_id,
            target=target,
            search_query=search_query,
            max_conclusions=max_conclusions,
        )
        print_result({"session_id": sid, "peer_id": peer_id, "target": target, "representation": result})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("get-metadata")
def get_metadata(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get metadata for a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        result = sess.get_metadata()
        print_result({"session_id": sid, "metadata": result})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("set-metadata")
def set_metadata(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    metadata: str = typer.Option(..., "--data", "-d", help="JSON metadata to set (e.g. '{\"key\": \"value\"}')"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Set metadata for a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as e:
        print_error("INVALID_JSON", f"metadata must be valid JSON: {e}", {})
        raise typer.Exit(1)

    sess = client.session(sid)

    try:
        sess.set_metadata(parsed)
        print_result({"session_id": sid, "metadata": parsed})
    except Exception as e:
        _handle_error(e, "session", sid)
