"""Honcho CLI — a terminal for Honcho.

Entry point and top-level command group.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer
from rich.console import Console

from honcho_cli import __version__
from honcho_cli.branding import BANNER, BRAND
from honcho_cli.common import _global_overrides
from honcho_cli.output import set_json_mode, use_json

app = typer.Typer(
    name="honcho",
    help="A terminal for Honcho — memory that reasons.",
    invoke_without_command=True,
    pretty_exceptions_enable=False,
)


def _json_requested_early() -> bool:
    """Best-effort JSON detection before Typer parses flags.

    version_callback is eager and fires before set_json_mode() runs, so we
    can't call use_json() here. Mirror its logic against argv/env/TTY.
    """
    return (
        "--json" in sys.argv
        or os.environ.get("HONCHO_JSON", "").lower() in ("1", "true")
        or not sys.stdout.isatty()
    )


def version_callback(value: bool) -> None:
    if value:
        if not _json_requested_early():
            print(BANNER)
        print(f"  honcho-cli {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", envvar="HONCHO_WORKSPACE_ID", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", envvar="HONCHO_PEER_ID", help="Override peer ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", envvar="HONCHO_SESSION_ID", help="Override session ID"),
    version: bool = typer.Option(False, "--version", "-V", callback=version_callback, is_eager=True, help="Show version"),
) -> None:
    """Honcho CLI — admin & debugging tool for Honcho workspaces."""
    set_json_mode(json_output)

    # Store global overrides for commands to access
    _global_overrides["workspace"] = workspace
    _global_overrides["peer"] = peer
    _global_overrides["session"] = session

    if ctx.invoked_subcommand is None:
        console = Console()
        if not use_json():
            console.print(f"[bold {BRAND}]{BANNER}[/bold {BRAND}]")
            console.print(f"  [dim]v{__version__}[/dim]\n")
        console.print(ctx.get_help())
        raise typer.Exit()


# Register top-level commands
from honcho_cli.commands.setup import doctor, init

app.command()(init)
app.command()(doctor)

# Register command groups
from honcho_cli.commands.config_cmd import app as config_app
from honcho_cli.commands.conclusion import app as conclusion_app
from honcho_cli.commands.message import app as message_app
from honcho_cli.commands.peer import app as peer_app
from honcho_cli.commands.session import app as session_app
from honcho_cli.commands.workspace import app as workspace_app

app.add_typer(config_app, name="config")
app.add_typer(workspace_app, name="workspace")
app.add_typer(peer_app, name="peer")
app.add_typer(session_app, name="session")
app.add_typer(message_app, name="message")
app.add_typer(conclusion_app, name="conclusion")


if __name__ == "__main__":
    app()
