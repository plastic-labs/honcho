"""Session commands: list, inspect, messages, context, summaries."""

from __future__ import annotations

from typing import Optional

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

        print_error("NO_SESSION", "No session ID provided. Use --session, set HONCHO_SESSION_ID, or run `honcho config set session_id <id>`.")
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
        msgs = list(sess.messages())
        if not reverse:
            msgs = msgs[-last:]
        else:
            msgs = msgs[:last]

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
