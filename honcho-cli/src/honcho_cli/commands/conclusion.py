"""Conclusion commands: list, search, create, delete."""

from __future__ import annotations

import json
from typing import Optional

import typer

from honcho_cli.commands.workspace import _handle_error
from honcho_cli.output import print_error, print_result, status, use_json
from honcho_cli.validation import validate_resource_id

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.common import add_common_options, get_client, get_resolved_config, handle_cmd_flags

app = typer.Typer(cls=HonchoTyperGroup, help="List, search, create, and delete peer conclusions (Honcho's memory atoms).")
add_common_options(app)


def _require_observer(observer: str | None) -> str:
    """Resolve observer peer ID; emit combined error if peer+workspace both missing."""
    config = get_resolved_config()
    obs = observer or config.peer_id
    if not obs:
        if not config.workspace_id:
            print_error(
                "NO_SCOPE",
                "No peer or workspace scoped. Pass --peer/-p and --workspace/-w, or set HONCHO_PEER_ID and HONCHO_WORKSPACE_ID.",
            )
        else:
            print_error("NO_PEER", "Peer required. Pass --peer/-p: honcho conclusion <cmd> -p <peer>")
        raise typer.Exit(1)
    return obs


@app.command("list")
def list_conclusions(
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    limit: int = typer.Option(10, "--limit", help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List conclusions."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    observer = _require_observer(observer)
    client, config = get_client()

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        conclusions = scope.list(size=limit).items
        items = [
            {
                "id": c.id,
                "content": c.content if use_json() else c.content[:200],
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

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    observer = _require_observer(observer)
    client, config = get_client()

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
                "content": c.content if use_json() else c.content[:200],
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

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer, session=session_id)
    observer = _require_observer(observer)
    client, config = get_client()

    # If content looks like JSON, try to parse it
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            content = payload.get("content", content)
    except json.JSONDecodeError:
        pass

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        params: dict[str, object] = {"content": content}
        if config.session_id:
            params["session_id"] = config.session_id
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

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    validate_resource_id(conclusion_id, "conclusion")
    client, config = get_client()

    if not observer:
        observer = config.peer_id
    if not observer:
        print_error("NO_PEER", "Peer required. Pass --peer/-p: honcho conclusion <cmd> -p <peer>")
        raise typer.Exit(1)

    p = client.peer(observer)

    if not yes:
        # SDK doesn't expose a get-by-id on ConclusionScope, so we can't
        # preview content cheaply — don't paginate the list just to
        # decorate the prompt. Show identifying fields only.
        if not use_json():
            typer.echo(
                f"  id:       {conclusion_id}\n"
                f"  observer: {observer}\n"
                f"  observed: {observed or '(self)'}"
            )
        typer.confirm(f"Delete conclusion '{conclusion_id}'?", abort=True)

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
