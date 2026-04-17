"""Honcho CLI — a terminal for Honcho.

Entry point and top-level command group.
"""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console

from honcho_cli import __version__
from honcho_cli._help import HonchoTyperGroup, print_welcome
from honcho_cli.branding import BANNER
from honcho_cli.output import set_json_mode


app = typer.Typer(
    name="honcho",
    cls=HonchoTyperGroup,
    help="A terminal for Honcho — memory that reasons.",
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    add_completion=False,
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
    version: bool = typer.Option(False, "--version", "-V", callback=version_callback, is_eager=True, help="Show version"),
) -> None:
    """Honcho CLI — admin & debugging tool for Honcho workspaces."""
    set_json_mode(json_output)

    if ctx.invoked_subcommand is None:
        print_welcome(Console())
        raise typer.Exit()


# Register top-level commands
from honcho_cli.commands.setup import doctor, init

app.command()(init)
app.command()(doctor)


@app.command("help", hidden=True)
def help_cmd(ctx: typer.Context) -> None:
    """Show help message."""
    Console().print(ctx.parent.get_help() if ctx.parent else "")
    raise typer.Exit()


# Register command groups
from honcho_cli.commands.config_cmd import app as config_app
from honcho_cli.commands.conclusion import app as conclusion_app
from honcho_cli.commands.message import app as message_app
from honcho_cli.commands.peer import app as peer_app
from honcho_cli.commands.session import app as session_app
from honcho_cli.commands.workspace import app as workspace_app

app.add_typer(peer_app,       name="peer")
app.add_typer(session_app,    name="session")
app.add_typer(message_app,    name="message")
app.add_typer(conclusion_app, name="conclusion")
app.add_typer(workspace_app,  name="workspace")
app.add_typer(config_app,     name="config")


if __name__ == "__main__":
    app()
