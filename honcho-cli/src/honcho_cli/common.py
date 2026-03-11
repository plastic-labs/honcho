"""Shared callback for subcommand groups to accept global-style flags."""

from __future__ import annotations

from typing import Optional

import typer

from honcho_cli.output import set_json_mode, set_quiet_mode


def add_common_options(app: typer.Typer) -> None:
    """Add a callback to a sub-app that accepts --json, --quiet, -w, -p, -s."""
    # Allow flags like --json after subcommands (e.g., `honcho workspace inspect granola --json`)
    app.info.context_settings = {"allow_interspersed_args": True}

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

        # Import here to avoid circular imports
        from honcho_cli.main import _global_overrides

        if workspace:
            _global_overrides["workspace"] = workspace
        if peer:
            _global_overrides["peer"] = peer
        if session:
            _global_overrides["session"] = session

        if ctx.invoked_subcommand is None:
            ctx.get_help()
