"""Message commands: list, get, create."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import typer

from honcho.api_types import MessageCreateParams

from honcho_cli.commands.session import _get_session_id
from honcho_cli.commands.workspace import _handle_error
from honcho_cli.output import print_error, print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options, get_client, handle_cmd_flags

app = typer.Typer(help="List, create, and get messages within a session.")
add_common_options(app)


@app.command("list")
def list_messages(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    last: int = typer.Option(20, "--last", help="Number of recent messages"),
    reverse: bool = typer.Option(False, "--reverse", help="Show oldest first (default is newest first)"),
    brief: bool = typer.Option(False, "--brief", help="Show only IDs, peer, token count, and created_at (no content)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Filter by peer ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List messages in a session. Scoped to a peer with -p."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        filters = {"peer_id": config.peer_id} if config.peer_id else None
        # Fetch newest-first so [:last] always gives the most recent N messages,
        # then flip to oldest-at-top / newest-at-bottom for readable display.
        # --reverse keeps the raw server order (oldest first, descending in table).
        msgs = sess.messages(filters=filters, reverse=True).items[:last]
        if not reverse:
            msgs = list(reversed(msgs))

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


@app.command("create")
def create_message(
    content: str = typer.Argument(help="Message content"),
    peer_id: str = typer.Option(..., "--peer", "-p", help="Peer ID of the message sender"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the message"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create a message in a session."""
    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session_id)
    sid = _get_session_id(None)
    validate_resource_id(peer_id, "peer")
    client, config = get_client()
    sess = client.session(sid)

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    try:
        msgs = sess.add_messages(MessageCreateParams(
            peer_id=peer_id,
            content=content,
            metadata=parsed_metadata,
        ))
        msg = msgs[0]
        print_result({
            "id": msg.id,
            "peer_id": msg.peer_id,
            "content": msg.content,
            "token_count": msg.token_count,
            "created_at": str(msg.created_at),
        })
    except Exception as e:
        _handle_error(e, "message", "create")


@app.command("get")
def get_message(
    message_id: str = typer.Argument(help="Message ID"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get a single message by ID."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    validate_resource_id(message_id, "message")
    sid = _get_session_id(session_id)
    client, config = get_client()

    try:
        sess = client.session(sid)
        msg = sess.get_message(message_id)

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
