"""Shared runtime state, client factory, and command-level flag helpers.

Flags like --json, -w, -p, -s work in TWO positions:
  1. Group-level (before subcommand):  honcho workspace --json list
  2. Command-level (after subcommand): honcho workspace list --json -w granola

Both positions are idempotent — if set at group level, the command-level is a no-op.
"""

from __future__ import annotations

from typing import Optional

import typer

from honcho import Honcho

from honcho_cli.config import CLIConfig, get_client_kwargs
from honcho_cli.output import print_error, set_json_mode
from honcho_cli.validation import validate_resource_id


# Global overrides from flags (commands read these)
_global_overrides: dict[str, str | None] = {
    "workspace": None,
    "peer": None,
    "session": None,
}


def get_resolved_config():
    """Get config with global flag overrides applied.

    Overrides flow through ``validate_resource_id`` so that a malformed
    ``-w``/``-p``/``-s`` value fails fast with a structured error rather than
    reaching the API and surfacing as an opaque ``UNKNOWN_ERROR``.
    """
    config = CLIConfig.load()

    if _global_overrides["workspace"]:
        config.workspace_id = validate_resource_id(_global_overrides["workspace"], "workspace")
    if _global_overrides["peer"]:
        config.peer_id = validate_resource_id(_global_overrides["peer"], "peer")
    if _global_overrides["session"]:
        config.session_id = validate_resource_id(_global_overrides["session"], "session")

    return config


def get_client(*, require_workspace: bool = True):
    """Create a Honcho client from resolved config.

    By default, refuses to build a client when no workspace is scoped — the
    SDK's get-or-create semantics would otherwise silently operate on an empty
    workspace. Commands that legitimately run without a workspace (e.g.
    ``workspace list``) pass ``require_workspace=False``.
    """
    config = get_resolved_config()
    if require_workspace and not config.workspace_id:
        print_error(
            "NO_WORKSPACE",
            "No workspace scoped. Pass --workspace/-w or set HONCHO_WORKSPACE_ID.",
        )
        raise typer.Exit(1)
    return Honcho(**get_client_kwargs(config)), config


def handle_cmd_flags(
    json_output: bool = False,
    workspace: str | None = None,
    peer: str | None = None,
    session: str | None = None,
    **_kwargs,
) -> None:
    """Apply command-level flags. Idempotent if already set by group callback."""
    if json_output:
        set_json_mode(True)

    if workspace:
        _global_overrides["workspace"] = workspace
    if peer:
        _global_overrides["peer"] = peer
    if session:
        _global_overrides["session"] = session


def add_common_options(app: typer.Typer) -> None:
    """Add a callback to a sub-app that accepts --json, -w, -p, -s."""

    @app.callback(invoke_without_command=True)
    def _callback(
        ctx: typer.Context,
        json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
        workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
        peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
        session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    ) -> None:
        if json_output:
            set_json_mode(True)

        if workspace:
            _global_overrides["workspace"] = workspace
        if peer:
            _global_overrides["peer"] = peer
        if session:
            _global_overrides["session"] = session

        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
