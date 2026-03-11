"""Shared callback and command-level flag helpers.

Flags like --json, -w, -p, -s work in TWO positions:
  1. Group-level (before subcommand):  honcho workspace --json list
  2. Command-level (after subcommand): honcho workspace list --json -w granola

Both positions are idempotent — if set at group level, the command-level is a no-op.
"""

from __future__ import annotations

from typing import Optional

import typer

from honcho_cli.output import set_json_mode, set_quiet_mode


def handle_cmd_flags(
    json_output: bool = False,
    workspace: str | None = None,
    peer: str | None = None,
    session: str | None = None,
) -> None:
    """Apply command-level flags. Idempotent if already set by group callback."""
    if json_output:
        set_json_mode(True)

    from honcho_cli.main import _global_overrides

    if workspace:
        _global_overrides["workspace"] = workspace
    if peer:
        _global_overrides["peer"] = peer
    if session:
        _global_overrides["session"] = session


def add_common_options(app: typer.Typer) -> None:
    """Add a callback to a sub-app that accepts --json, --quiet, -w, -p, -s."""

    @app.callback(invoke_without_command=True)
    def _callback(
        ctx: typer.Context,
        json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress status messages"),
        workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
        peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Override peer ID"),
        session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    ) -> None:
        if json_output:
            set_json_mode(True)
        if quiet:
            set_quiet_mode(True)

        from honcho_cli.main import _global_overrides

        if workspace:
            _global_overrides["workspace"] = workspace
        if peer:
            _global_overrides["peer"] = peer
        if session:
            _global_overrides["session"] = session

        if ctx.invoked_subcommand is None:
            ctx.get_help()
