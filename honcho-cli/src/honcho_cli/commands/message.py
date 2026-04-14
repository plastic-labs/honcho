"""Message commands: list, get."""

from __future__ import annotations

import hashlib
from typing import Optional

import typer

from honcho_cli.commands.workspace import _handle_error
from honcho_cli.output import print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Message operations.")
add_common_options(app)


@app.command("list")
def list_messages(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    last: int = typer.Option(20, "--last", help="Number of recent messages"),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse order"),
    brief: bool = typer.Option(False, "--brief", help="Show only IDs, peer, token count, and created_at (no content)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List messages in a session."""
    from honcho_cli.commands.session import _get_session_id
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        # Server supports ?reverse=true on messages/list, but the Python
        # SDK doesn't forward it from Session.messages() yet. Until then,
        # --reverse walks every page via the SDK iterator and slices
        # — O(pages) in the session size. Safe for small sessions.
        if not reverse:
            msgs = sess.messages().items[:last]
        else:
            msgs = list(sess.messages())[-last:]

        # Detect duplicate content
        content_hashes: dict[str, list[str]] = {}
        for m in msgs:
            h = hashlib.md5(m.content.encode()).hexdigest()
            content_hashes.setdefault(h, []).append(m.id)
        dupes = {h: ids for h, ids in content_hashes.items() if len(ids) > 1}
        if dupes:
            dupe_count = sum(len(ids) - 1 for ids in dupes.values())
            status(f"Warning: {dupe_count} duplicate message(s) detected (identical content, different IDs)")

        if brief:
            items = [
                {
                    "id": m.id,
                    "peer_id": m.peer_id,
                    "token_count": m.token_count,
                    "created_at": str(m.created_at),
                }
                for m in msgs
            ]
            print_result(items, columns=["id", "peer_id", "token_count", "created_at"], title="Messages")
        else:
            items = [
                {
                    "id": m.id,
                    "peer_id": m.peer_id,
                    "content": m.content,
                    "token_count": m.token_count,
                    "metadata": m.metadata,
                    "created_at": str(m.created_at),
                }
                for m in msgs
            ]
            print_result(items, columns=["id", "peer_id", "content", "created_at"], title="Messages")
    except Exception as e:
        _handle_error(e, "message", "list")


@app.command("get")
def get_message(
    message_id: str = typer.Argument(help="Message ID"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get a single message by ID."""
    from honcho_cli.commands.session import _get_session_id
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    validate_resource_id(message_id, "message")
    sid = _get_session_id(session_id)
    client, config = get_client()

    try:
        # Hit the direct message endpoint instead of paging the session.
        from honcho.http import routes
        from honcho.api_types import MessageResponse
        from honcho.message import Message

        sess = client.session(sid)
        data = client._http.get(routes.message(sess.workspace_id, sess.id, message_id))
        msg = Message.from_api_response(MessageResponse.model_validate(data))

        print_result({
            "id": msg.id,
            "peer_id": msg.peer_id,
            "content": msg.content,
            "token_count": msg.token_count,
            "metadata": msg.metadata,
            "created_at": str(msg.created_at),
        })
    except SystemExit:
        raise
    except Exception as e:
        _handle_error(e, "message", message_id)
