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


_LEVELS = ("explicit", "deductive", "inductive", "contradiction")

#: Columns carrying a conclusion's attribution, shown alongside the content.
_ATTRIBUTION_COLUMNS = ["id", "level", "source_ids", "times_derived", "content"]


def _format_conclusion(c, workspace_id: str | None) -> dict:
    """Shape a Conclusion for output.

    ``level``, ``source_ids`` and ``times_derived`` are the attribution the
    server started returning in Honcho v3.2.0. In table mode ``source_ids``
    collapses to a count, since the ids are nanoids and a list of them makes
    the row unreadable; JSON mode keeps the full list so it can be piped back
    into ``honcho conclusion get``.
    """
    source_ids = c.source_ids or []
    return {
        "id": c.id,
        "level": c.level,
        "source_ids": source_ids if use_json() else len(source_ids),
        "times_derived": c.times_derived,
        "content": c.content if use_json() else c.content[:160],
        "workspace_id": workspace_id,
        "observer_id": c.observer_id,
        "observed_id": c.observed_id,
        "session_id": c.session_id,
        "created_at": str(c.created_at),
    }


def _validate_level(level: str | None) -> str | None:
    if level and level not in _LEVELS:
        print_error("INVALID_LEVEL", f"--level must be one of: {', '.join(_LEVELS)}")
        raise typer.Exit(1)
    return level


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
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="Only this reasoning level: explicit, deductive, inductive, contradiction",
    ),
    derived_from: Optional[str] = typer.Option(
        None,
        "--derived-from",
        help="Only conclusions derived from this conclusion ID",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List conclusions."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    _validate_level(level)
    observer = _require_observer(observer)
    client, config = get_client()

    p = client.peer(observer)

    filters: dict = {}
    if level:
        filters["level"] = level
    if derived_from:
        filters["source_ids"] = {"contains": derived_from}

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        conclusions = scope.list(size=limit, filters=filters or None).items
        items = [_format_conclusion(c, config.workspace_id) for c in conclusions]
        print_result(items, columns=_ATTRIBUTION_COLUMNS + ["observed_id", "created_at"], title="Conclusions")
    except Exception as e:
        _handle_error(e, "conclusion", "list")


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    observer: Optional[str] = typer.Option(None, "--observer", help="Observer peer ID"),
    observed: Optional[str] = typer.Option(None, "--observed", help="Observed peer ID"),
    top_k: int = typer.Option(10, help="Max results"),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="Only this reasoning level: explicit, deductive, inductive, contradiction",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Semantic search over conclusions."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    _validate_level(level)
    observer = _require_observer(observer)
    client, config = get_client()

    p = client.peer(observer)

    try:
        if observed:
            scope = p.conclusions_of(observed)
        else:
            scope = p.conclusions

        results = scope.query(query, top_k=top_k, filters={"level": level} if level else None)
        items = [_format_conclusion(c, config.workspace_id) for c in results]
        print_result(items, columns=_ATTRIBUTION_COLUMNS + ["created_at"], title=f"Conclusion search: {query}")
    except Exception as e:
        _handle_error(e, "conclusion", "search")


@app.command()
def get(
    conclusion_ids: list[str] = typer.Argument(help="Conclusion IDs to fetch"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Fetch conclusions by ID, from anywhere in the workspace.

    Pass a conclusion's source_ids to see the premises it was derived from,
    and repeat to walk a reasoning chain down to the explicit facts it rests
    on. IDs that no longer exist are reported rather than failing the command.
    """

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()

    for cid in conclusion_ids:
        validate_resource_id(cid, "conclusion")

    try:
        conclusions = client.conclusions.get_many(list(conclusion_ids))
        found = {c.id for c in conclusions}
        missing = [cid for cid in conclusion_ids if cid not in found]

        items = [_format_conclusion(c, config.workspace_id) for c in conclusions]
        print_result(items, columns=_ATTRIBUTION_COLUMNS + ["observed_id", "created_at"], title="Conclusions")
        if missing:
            print_error(
                "MISSING_CONCLUSIONS",
                f"Not in this workspace (consolidated or deleted): {', '.join(missing)}",
            )
    except Exception as e:
        _handle_error(e, "conclusion", "get")


@app.command()
def derived(
    conclusion_id: str = typer.Argument(help="The premise conclusion ID"),
    limit: int = typer.Option(10, "--limit", help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List the conclusions derived FROM a conclusion.

    Walks the reasoning tree upward (premise -> conclusion); `conclusion get`
    on a conclusion's source_ids walks it downward. Worth checking before
    deleting or correcting a fact.
    """

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    validate_resource_id(conclusion_id, "conclusion")
    client, config = get_client()

    try:
        page = client.conclusions.list(
            size=limit, filters={"source_ids": {"contains": conclusion_id}}
        )
        items = [_format_conclusion(c, config.workspace_id) for c in page.items]
        print_result(
            items,
            columns=_ATTRIBUTION_COLUMNS + ["observed_id", "created_at"],
            title=f"Derived from {conclusion_id}",
        )
    except Exception as e:
        _handle_error(e, "conclusion", "derived")


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
