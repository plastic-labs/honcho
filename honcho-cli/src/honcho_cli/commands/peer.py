"""Peer commands: list, inspect, card, chat, search, create, metadata, representation."""

from __future__ import annotations

import json
from typing import Optional

import typer

from honcho_cli.commands.workspace import _config_to_dict, _handle_error
from honcho_cli.output import print_result, use_json
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

        print_error("NO_PEER", "No peer ID provided. Pass --peer/-p or set HONCHO_PEER_ID.")
        raise typer.Exit(1)
    return validate_resource_id(pid, "peer")


@app.command("list")
def list_peers(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List all peers in the workspace."""
    from honcho_cli.commands.workspace import _raw_list
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
                "configuration": _config_to_dict(p.configuration) if p.configuration else None,
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
        peer_config = p.get_configuration()
        # First page only; SyncPage.total (when the server supplies it) is
        # authoritative for counts without walking every page.
        session_page = p.sessions()
        conclusion_page = p.conclusions.list(size=10)

        session_items = session_page.items
        conclusion_items = conclusion_page.items

        result = {
            "id": pid,
            "card": card,
            "configuration": _config_to_dict(peer_config) if peer_config else None,
            "session_count": session_page.total if session_page.total is not None else len(session_items),
            "conclusion_count": conclusion_page.total if conclusion_page.total is not None else len(conclusion_items),
            "recent_conclusions": [
                {"id": c.id, "content": c.content if use_json() else c.content[:200], "created_at": str(c.created_at)}
                for c in conclusion_items
            ],
            "sessions": [{"id": s.id} for s in session_items[:10]],
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
    target: Optional[str] = typer.Option(None, help="Target peer for perspective"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Query the dialectic about a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(None)
    client, config = get_client()
    p = client.peer(pid)

    try:
        # Session scope comes from the global -s pipeline (config.session_id),
        # keeping chat consistent with every other peer/session command.
        response = p.chat(query, target=target, session=config.session_id or None)
        print_result({"peer_id": pid, "query": query, "response": response})
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search a peer's messages."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(None)
    client, config = get_client()
    p = client.peer(pid)

    try:
        results = p.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "content": m.content if use_json() else m.content[:200],
                "session_id": m.session_id,
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "session_id", "content", "created_at"], title=f"Peer search: {query}")
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command("create")
def create_peer(
    peer_id: str = typer.Argument(help="Peer ID to create or get"),
    observe_me: Optional[bool] = typer.Option(None, "--observe-me/--no-observe-me", help="Whether Honcho will form a representation of this peer"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the peer"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create or get a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client
    from honcho.api_types import PeerConfig

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    pid = validate_resource_id(peer_id, "peer")
    client, config = get_client()

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            from honcho_cli.output import print_error
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    peer_config = PeerConfig(observe_me=observe_me) if observe_me is not None else None

    try:
        p = client.peer(pid, configuration=peer_config, metadata=parsed_metadata)
        # Only round-trip to the server when the caller passed config or
        # metadata — in that case get-or-create may have returned a
        # pre-existing peer and the echoed output would lie. When no input
        # was passed, skip the two extra API calls entirely.
        result: dict[str, object] = {"peer_id": p.id}
        if peer_config is not None or parsed_metadata is not None:
            result["metadata"] = p.get_metadata()
            result["configuration"] = _config_to_dict(p.get_configuration())
        print_result(result)
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command("get-metadata")
def get_metadata(
    peer_id: Optional[str] = typer.Argument(None, help="Peer ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get metadata for a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        result = p.get_metadata()
        print_result({"peer_id": pid, "metadata": result})
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command("set-metadata")
def set_metadata(
    metadata: str = typer.Argument(help="JSON metadata to set (e.g. '{\"key\": \"value\"}')"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Peer ID (uses default if omitted)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Set metadata for a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(None)
    client, config = get_client()

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as e:
        from honcho_cli.output import print_error
        print_error("INVALID_JSON", f"metadata must be valid JSON: {e}", {})
        raise typer.Exit(1)

    p = client.peer(pid)

    try:
        p.set_metadata(parsed)
        print_result({"peer_id": pid, "metadata": parsed})
    except Exception as e:
        _handle_error(e, "peer", pid)


@app.command()
def representation(
    peer_id: Optional[str] = typer.Argument(None, help="Peer ID (uses default if omitted)"),
    target: Optional[str] = typer.Option(None, help="Target peer to get representation about"),
    search_query: Optional[str] = typer.Option(None, help="Semantic search query to filter conclusions"),
    max_conclusions: Optional[int] = typer.Option(None, help="Maximum number of conclusions to include"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get the formatted representation for a peer."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    pid = _get_peer_id(peer_id)
    client, config = get_client()
    p = client.peer(pid)

    try:
        result = p.representation(
            target=target,
            session=config.session_id or None,
            search_query=search_query,
            max_conclusions=max_conclusions,
        )
        print_result({"peer_id": pid, "target": target, "representation": result})
    except Exception as e:
        _handle_error(e, "peer", pid)
