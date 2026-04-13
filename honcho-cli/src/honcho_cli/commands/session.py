"""Session commands: list, inspect, messages, context, summaries, peers, search, representation, metadata."""

from __future__ import annotations

import json
from typing import List, Optional

import typer

from honcho_cli.commands.workspace import _config_to_dict, _handle_error
from honcho_cli.output import print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Session debugging operations.")
add_common_options(app)


def _get_session_id(session_id: str | None) -> str:
    from honcho_cli.main import get_resolved_config

    config = get_resolved_config()
    sid = session_id or config.session_id
    if not sid:
        from honcho_cli.output import print_error

        print_error("NO_SESSION", "No session ID provided. Pass --session/-s or set HONCHO_SESSION_ID.")
        raise typer.Exit(1)
    return validate_resource_id(sid, "session")


@app.command("list")
def list_sessions(
    peer_id: Optional[str] = typer.Option(None, "--peer", "-p", help="Filter by peer"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List sessions in the workspace."""
    from honcho_cli.commands.workspace import _raw_list
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
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


@app.command()
def inspect(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a session: peers, message count, summaries, config."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        peers = sess.peers()
        messages = list(sess.messages())
        summaries = sess.summaries()
        sess_config = sess.get_configuration()

        from honcho_cli.commands.workspace import _compact_config

        raw_config = _config_to_dict(sess_config) if sess_config else None
        result = {
            "id": sid,
            "peers": [{"id": p.id} for p in peers],
            "message_count": len(messages),
            "summaries": {
                "short": summaries.short_summary if hasattr(summaries, "short_summary") else None,
                "long": summaries.long_summary if hasattr(summaries, "long_summary") else None,
            },
            "configuration": _compact_config(raw_config) if isinstance(raw_config, dict) else raw_config,
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def messages(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    last: int = typer.Option(20, "--last", help="Number of recent messages"),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse order (oldest first)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List recent messages in a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        # SDK returns most-recent-first. Default case (newest N) only needs
        # the first page. --reverse (oldest N) still walks all pages until
        # the SDK accepts order=asc on session.messages().
        if not reverse:
            msgs = sess.messages().items[:last]
        else:
            msgs = list(sess.messages())[-last:]

        items = [
            {
                "id": m.id,
                "peer_id": m.peer_id,
                "content": m.content[:300],
                "token_count": m.token_count,
                "created_at": str(m.created_at),
            }
            for m in msgs
        ]
        print_result(items, columns=["id", "peer_id", "content", "created_at"], title=f"Messages ({sid})")
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
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

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
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

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
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()

    if not yes:
        typer.confirm(f"Delete session '{sid}' and all its messages, conclusions, and queue items?", abort=True)

    sess = client.session(sid)

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
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

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
    peer_ids: List[str] = typer.Argument(help="Peer IDs to add to the session"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Add peers to a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session_id)
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
    peer_ids: List[str] = typer.Argument(help="Peer IDs to remove from the session"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Remove peers from a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session_id)
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
    limit: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (uses default if omitted)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search messages in a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(None)
    client, config = get_client()
    sess = client.session(sid)

    try:
        results = sess.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "peer_id": m.peer_id,
                "content": m.content[:200],
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
    target: Optional[str] = typer.Option(None, help="Target peer (what peer_id knows about target)"),
    search_query: Optional[str] = typer.Option(None, help="Semantic search query to filter conclusions"),
    max_conclusions: Optional[int] = typer.Option(None, help="Maximum number of conclusions to include"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (uses default if omitted)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get the representation of a peer within a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(None)
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
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

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
    metadata: str = typer.Argument(help="JSON metadata to set (e.g. '{\"key\": \"value\"}')"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID (uses default if omitted)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Set metadata for a session."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(None)
    client, config = get_client()

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as e:
        from honcho_cli.output import print_error
        print_error("INVALID_JSON", f"metadata must be valid JSON: {e}", {})
        raise typer.Exit(1)

    sess = client.session(sid)

    try:
        sess.set_metadata(parsed)
        print_result({"session_id": sid, "metadata": parsed})
    except Exception as e:
        _handle_error(e, "session", sid)
