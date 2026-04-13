"""Conclusion commands: list, search, create, delete."""

from __future__ import annotations

import json
from typing import Optional

import typer

from honcho_cli.commands.workspace import _handle_error
from honcho_cli.output import print_error, print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Conclusion (observation) operations.")
add_common_options(app)


@app.command("list")
def list_conclusions(
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List conclusions."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    client, config = get_client()

    if not observer:
        observer = config.peer_id
    if not observer:
        print_error("NO_PEER", "Observer peer ID required. Use --observer or set default peer.")
        raise typer.Exit(1)

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        conclusions = list(scope.list())
        items = [
            {
                "id": c.id,
                "content": c.content[:200],
                "workspace_id": config.workspace_id,
                "observer_id": c.observer_id,
                "observed_id": c.observed_id,
                "session_id": c.session_id,
                "created_at": str(c.created_at),
            }
            for c in conclusions
        ]
        print_result(items, columns=["id", "content", "workspace_id", "observer_id", "observed_id", "session_id", "created_at"], title="Conclusions")
    except Exception as e:
        _handle_error(e, "conclusion", "list")


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    top_k: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Semantic search over conclusions."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    client, config = get_client()

    if not observer:
        observer = config.peer_id
    if not observer:
        print_error("NO_PEER", "Observer peer ID required. Use --observer or set default peer.")
        raise typer.Exit(1)

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        results = scope.query(query, top_k=top_k)
        items = [
            {
                "id": c.id,
                "content": c.content[:200],
                "workspace_id": config.workspace_id,
                "observer_id": c.observer_id,
                "observed_id": c.observed_id,
                "session_id": c.session_id,
                "created_at": str(c.created_at),
            }
            for c in results
        ]
        print_result(items, columns=["id", "content", "workspace_id", "session_id", "created_at"], title=f"Conclusion search: {query}")
    except Exception as e:
        _handle_error(e, "conclusion", "search")


@app.command()
def create(
    content: str = typer.Argument(help="Conclusion content or JSON payload"),
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session context"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create a conclusion."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    client, config = get_client()

    if not observer:
        observer = config.peer_id
    if not observer:
        print_error("NO_PEER", "Observer peer ID required. Use --observer or set default peer.")
        raise typer.Exit(1)

    # If content looks like JSON, try to parse it
    try:
        payload = json.loads(content)
        content = payload.get("content", content)
    except (json.JSONDecodeError, AttributeError):
        pass

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        params: dict[str, object] = {"content": content}
        if session_id:
            params["session_id"] = session_id
        results = scope.create([params])
        result = results[0] if results else None
        if result is None:
            print_error("CREATE_FAILED", "Conclusion create returned no results")
            raise typer.Exit(1)
        print_result({
            "id": result.id,
            "content": result.content,
            "workspace_id": config.workspace_id,
            "observer_id": result.observer_id,
            "observed_id": result.observed_id,
            "session_id": result.session_id,
            "created_at": str(result.created_at),
        })
    except Exception as e:
        _handle_error(e, "conclusion", "create")


@app.command()
def delete(
    conclusion_id: str = typer.Argument(help="Conclusion ID to delete"),
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete a conclusion."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    validate_resource_id(conclusion_id, "conclusion")
    client, config = get_client()

    if not observer:
        observer = config.peer_id
    if not observer:
        print_error("NO_PEER", "Observer peer ID required. Use --observer or set default peer.")
        raise typer.Exit(1)

    if not yes:
        typer.confirm(f"Delete conclusion '{conclusion_id}'?", abort=True)

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        scope.delete(conclusion_id)
        status(f"Conclusion '{conclusion_id}' deleted")
        print_result({"deleted": conclusion_id})
    except Exception as e:
        _handle_error(e, "conclusion", conclusion_id)
