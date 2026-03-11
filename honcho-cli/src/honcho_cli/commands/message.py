"""Message commands: list, get."""

from __future__ import annotations

from typing import Optional

import typer

from honcho_cli.commands.workspace import _handle_error
from honcho_cli.output import print_result
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Message operations.")
add_common_options(app)


@app.command("list")
def list_messages(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    last: int = typer.Option(20, "--last", help="Number of recent messages"),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse order"),
) -> None:
    """List messages in a session."""
    from honcho_cli.commands.session import _get_session_id
    from honcho_cli.main import get_client

    sid = _get_session_id(session_id)
    client, config = get_client()
    session = client.session(sid)

    try:
        msgs = list(session.messages())
        if not reverse:
            msgs = msgs[-last:]
        else:
            msgs = msgs[:last]

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
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID"),
) -> None:
    """Get a single message by ID."""
    from honcho_cli.commands.session import _get_session_id
    from honcho_cli.main import get_client

    validate_resource_id(message_id, "message")
    sid = _get_session_id(session_id)
    client, config = get_client()

    try:
        # Use raw HTTP to get a single message
        session = client.session(sid)
        msgs = list(session.messages())
        msg = next((m for m in msgs if m.id == message_id), None)

        if msg is None:
            from honcho_cli.output import print_error

            print_error("MESSAGE_NOT_FOUND", f"Message '{message_id}' not found in session '{sid}'", {"message_id": message_id, "session_id": sid})
            raise typer.Exit(1)

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
