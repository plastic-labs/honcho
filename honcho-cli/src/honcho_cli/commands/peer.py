"""Peer commands: list, inspect, card, chat, search."""

from __future__ import annotations

from typing import Optional

import typer

from honcho_cli.commands.workspace import _config_to_dict, _handle_error
from honcho_cli.output import print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Peer debugging operations.")
add_common_options(app)


def _get_peer_id(peer_id: str | None) -> str:
    from honcho_cli.main import get_resolved_config

    config = get_resolved_config()
    pid = peer_id or config.peer_id
    if not pid:
        from honcho_cli.output import print_error

        print_error("NO_PEER", "No peer ID provided. Use --peer, set HONCHO_PEER_ID, or run `honcho config set peer_id <id>`.")
        raise typer.Exit(1)
    return validate_resource_id(pid, "peer")


@app.command("list")
def list_peers(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List all peers in the workspace."""
    from honcho_cli.commands.workspace import _compact_config, _raw_list
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()

    try:
        raw_peers = _raw_list(client.peers())
        items = [
            {
                "id": p.id,
                "metadata": p.metadata,
                "configuration": _compact_config(_config_to_dict(p.configuration)) if p.configuration else None,
                "created_at": str(p.created_at),
            }
            for p in raw_peers
        ]
        print_result(items, columns=["id", "metadata", "created_at"], title="Peers")
    except Exception as e:
        _handle_error(e, "peer", "list")


@app.command()
def inspect(
    peer_id: Optional[str] = typer.Argument(None, help="Peer ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a peer: card, session count, recent conclusions."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        card = p.get_card()
        sessions = list(p.sessions())
        conclusions = list(p.conclusions.list())

        result = {
            "id": pid,
            "card": card,
            "session_count": len(sessions),
            "conclusion_count": len(conclusions),
            "recent_conclusions": [
                {"id": c.id, "content": c.content[:200], "created_at": str(c.created_at)}
                for c in conclusions[:10]
            ],
            "sessions": [{"id": s.id} for s in sessions[:10]],
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command()
def card(
    peer_id: Optional[str] = typer.Argument(None, help="Peer ID (uses default if omitted)"),
    target: Optional[str] = typer.Option(None, help="Target peer for relationship card"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get raw peer card content."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        result = p.get_card(target=target)
        print_result({"peer_id": pid, "target": target, "card": result})
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command()
def chat(
    query: str = typer.Argument(help="Question to ask about the peer"),
    peer_id: Optional[str] = typer.Option(None, help="Peer ID (uses default if omitted)"),
    target: Optional[str] = typer.Option(None, help="Target peer for perspective"),
    session: Optional[str] = typer.Option(None, help="Session context"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Query the dialectic about a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        response = p.chat(query, target=target, session=session)
        print_result({"peer_id": pid, "query": query, "response": response})
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    peer_id: Optional[str] = typer.Option(None, help="Peer ID (uses default if omitted)"),
    limit: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search a peer's messages."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        results = p.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "content": m.content[:200],
                "session_id": m.session_id,
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "session_id", "content", "created_at"], title=f"Peer search: {query}")
    except Exception as e:
        _handle_error(e, "peer", pid)
