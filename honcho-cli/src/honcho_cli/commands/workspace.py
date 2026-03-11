"""Workspace commands: list, inspect, delete, search, queue-status."""

from __future__ import annotations

from typing import Optional

import typer

from honcho_cli.output import print_error, print_result, status
from honcho_cli.validation import validate_resource_id

from honcho_cli.common import add_common_options

app = typer.Typer(help="Workspace operations.")
add_common_options(app)


def _get_workspace_id(workspace_id: str | None) -> str:
    from honcho_cli.main import get_resolved_config

    config = get_resolved_config()
    wid = workspace_id or config.workspace_id
    if not wid:
        print_error("NO_WORKSPACE", "No workspace ID provided. Use --workspace, set HONCHO_WORKSPACE_ID, or run `honcho config set workspace_id <id>`.")
        raise typer.Exit(1)
    return validate_resource_id(wid, "workspace")


def _raw_list(page) -> list:
    """Collect all raw API response items across all pages of a SyncPage."""
    items = list(page._raw_items)
    while page.has_next_page():
        page = page.get_next_page()
        if page is None:
            break
        items.extend(page._raw_items)
    return items


def _compact_config(config_dict: dict) -> dict | str:
    """Return '(defaults)' if all config values are None, else the dict."""
    if all(v is None for v in config_dict.values()):
        return "(defaults)"
    return config_dict


@app.command("list")
def list_workspaces(
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List all accessible workspaces."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output)
    client, config = get_client()

    try:
        workspaces = list(client.workspaces())
        items = [{"id": w} for w in workspaces]
        print_result(items, columns=["id"], title="Workspaces")
    except Exception as e:
        _handle_error(e, "workspace", "list")


@app.command()
def inspect(
    workspace_id: Optional[str] = typer.Argument(None, help="Workspace ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a workspace: peers, sessions, config."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace)

    wid = _get_workspace_id(workspace_id)
    client, config = get_client()

    # Override workspace if positional arg given
    if workspace_id:
        client = _with_workspace(client, workspace_id)

    try:
        ws_config = client.get_configuration()
        ws_metadata = client.get_metadata()

        # Use raw API response objects to get all fields (created_at, is_active)
        raw_peers = _raw_list(client.peers())
        raw_sessions = _raw_list(client.sessions())

        result = {
            "workspace_id": wid,
            "metadata": ws_metadata,
            "configuration": _compact_config(_config_to_dict(ws_config)),
            "peer_count": len(raw_peers),
            "session_count": len(raw_sessions),
            "peers": [
                {"id": p.id, "metadata": p.metadata, "created_at": str(p.created_at)}
                for p in raw_peers[:20]
            ],
            "sessions": [
                {"id": s.id, "is_active": s.is_active, "metadata": s.metadata, "created_at": str(s.created_at)}
                for s in raw_sessions[:20]
            ],
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "workspace", wid)


@app.command()
def delete(
    workspace_id: str = typer.Argument(help="Workspace ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete a workspace. Destructive — requires --yes or interactive confirm."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output)

    validate_resource_id(workspace_id, "workspace")
    client, config = get_client()
    client = _with_workspace(client, workspace_id)

    if dry_run:
        sessions = list(client.sessions())
        peers = list(client.peers())
        print_result({
            "dry_run": True,
            "workspace_id": workspace_id,
            "sessions_to_delete": len(sessions),
            "peers_to_delete": len(peers),
        })
        return

    if not yes:
        typer.confirm(f"Delete workspace '{workspace_id}' and all its data?", abort=True)

    try:
        client.delete_workspace(workspace_id)
        status(f"Workspace '{workspace_id}' deleted")
        print_result({"deleted": workspace_id})
    except Exception as e:
        _handle_error(e, "workspace", workspace_id)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    workspace_id: Optional[str] = typer.Option(None, help="Workspace ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    limit: int = typer.Option(10, help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search messages across workspace."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace)

    wid = _get_workspace_id(workspace_id)
    client, config = get_client()

    try:
        results = client.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "content": m.content[:200],
                "peer_id": m.peer_id,
                "session_id": m.session_id,
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "peer_id", "session_id", "content"], title=f"Search: {query}")
    except Exception as e:
        _handle_error(e, "workspace", wid)


@app.command("queue-status")
def queue_status(
    workspace_id: Optional[str] = typer.Option(None, help="Workspace ID"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    observer: Optional[str] = typer.Option(None, help="Filter by observer peer"),
    sender: Optional[str] = typer.Option(None, help="Filter by sender peer"),
    session: Optional[str] = typer.Option(None, help="Filter by session"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get queue processing status."""
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_client

    handle_cmd_flags(json_output=json_output, workspace=workspace)

    _get_workspace_id(workspace_id)
    client, config = get_client()

    try:
        result = client.queue_status(observer=observer, sender=sender, session=session)
        print_result(result.__dict__ if hasattr(result, "__dict__") else result)
    except Exception as e:
        _handle_error(e, "queue", "status")


def _with_workspace(client, workspace_id: str):
    """Return a new client pointed at a different workspace."""
    from honcho import Honcho

    return Honcho(
        base_url=str(client.base_url),
        api_key=client._http._api_key if hasattr(client._http, "_api_key") else None,
        workspace_id=workspace_id,
    )


def _config_to_dict(config) -> dict:
    """Convert a config object to a dict, handling nested objects."""
    if hasattr(config, "__dict__"):
        result = {}
        for k, v in config.__dict__.items():
            if k.startswith("_"):
                continue
            result[k] = _config_to_dict(v) if hasattr(v, "__dict__") and not isinstance(v, str) else v
        return result
    return config


def _handle_error(e: Exception, resource: str, resource_id: str) -> None:
    """Handle SDK exceptions with structured error output."""
    error_str = str(e)
    if "404" in error_str or "not found" in error_str.lower():
        print_error(
            f"{resource.upper()}_NOT_FOUND",
            f"{resource.title()} '{resource_id}' not found",
            {resource: resource_id},
        )
        raise typer.Exit(1)
    elif "401" in error_str or "403" in error_str or "auth" in error_str.lower():
        print_error("AUTH_ERROR", f"Authentication failed: {error_str}", {})
        raise typer.Exit(3)
    elif "500" in error_str or "server" in error_str.lower():
        print_error("SERVER_ERROR", f"Server error: {error_str}", {resource: resource_id})
        raise typer.Exit(2)
    else:
        print_error("UNKNOWN_ERROR", str(e), {resource: resource_id})
        raise typer.Exit(1)
