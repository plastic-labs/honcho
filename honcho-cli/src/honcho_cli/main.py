"""Honcho CLI — a terminal for Honcho.

Entry point and top-level command group.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import click
import typer
import typer.rich_utils as ru
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperCommand, TyperGroup

from honcho_cli import __version__
from honcho_cli.branding import BANNER, BRAND
from honcho_cli.common import _global_overrides
from honcho_cli.output import set_json_mode, use_json


# Theme Typer's rich help renderer to match the welcome's voice.
# Applies to --help at every level (top, group, command).
ru.STYLE_COMMANDS_PANEL_BORDER = "dim"
ru.STYLE_OPTIONS_PANEL_BORDER = "dim"
ru.STYLE_ERRORS_PANEL_BORDER = "dim"
ru.STYLE_OPTION = f"bold {BRAND}"
ru.STYLE_SWITCH = f"bold {BRAND}"
ru.STYLE_USAGE = "dim"
ru.STYLE_USAGE_COMMAND = f"bold {BRAND}"


def _pattern_and_example(self, ctx):
    """Replace Click's 'Usage: …' line with pattern/example rows.

    Applied to every TyperGroup/TyperCommand via monkey-patch so sub-groups
    (honcho workspace --help, honcho peer chat --help, …) show a consistent
    pattern+example block where the Usage line used to be. Top-level
    honcho --help short-circuits in HonchoTyperGroup.format_help and never
    reaches this.
    """
    original = click.Command.get_usage(self, ctx)
    pattern = original.replace("Usage: ", "", 1) if original.startswith("Usage: ") else original
    path = ctx.command_path

    # Pick a reasonable example per shape
    if isinstance(self, click.Group):
        subcmds = self.list_commands(ctx)
        example = f"{path} {subcmds[0]}" if subcmds else f"{path} --help"
    else:
        example = path

    return f"pattern: {pattern}\nexample: {example}"


TyperGroup.get_usage = _pattern_and_example
TyperCommand.get_usage = _pattern_and_example


class HonchoTyperGroup(TyperGroup):
    """Top-level --help renders the curated welcome; sub-groups fall through to Typer."""

    def format_help(self, ctx, formatter):
        if ctx.parent is None:
            _print_welcome(Console())
            return
        super().format_help(ctx, formatter)


app = typer.Typer(
    name="honcho",
    cls=HonchoTyperGroup,
    help="A terminal for Honcho — memory that reasons.",
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)


def _cmd_table(rows: list[tuple[str, str]]) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 2, 0, 0), expand=False)
    t.add_column("cmd", style=f"bold {BRAND}", no_wrap=True)
    t.add_column("desc", style="default")
    for cmd, desc in rows:
        t.add_row(cmd, desc)
    return t


def _welcome_panel(title: str, rows: list[tuple[str, str]]) -> Panel:
    return Panel(
        _cmd_table(rows),
        title=f"[dim]{title}[/dim]",
        title_align="left",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def _print_welcome(console: Console) -> None:
    if use_json():
        return
    console.print(f"[bold {BRAND}]{BANNER}[/bold {BRAND}]")
    console.print(f"  [dim]v{__version__}[/dim]\n", highlight=False)

    start_rows = [
        ("honcho init",   "configure API key and server URL"),
        ("honcho doctor", "verify connection and workspace health"),
        ("", ""),
        ("[dim]pattern[/dim]", "[dim]honcho [-w workspace] [-p peer] [-s session] <command>[/dim]"),
        ("[dim]example[/dim]", "[dim]honcho -p alice peer chat \"what does alice prefer?\"[/dim]"),
    ]
    memory_rows = [
        ("honcho peer chat \"...\"",          "query the dialectic about a peer"),
        ("honcho peer inspect",               "review what honcho knows about a peer"),
        ("honcho peer representation",        "get the full peer representation"),
        ("honcho -p <peer> conclusion list",  "browse peer memory atoms"),
    ]
    cmd_rows = [
        ("peer",       "chat · inspect · card · search · representation"),
        ("",           "list · create · get-metadata · set-metadata"),
        ("session",    "list · inspect · context · summaries · peers"),
        ("",           "add-peers · remove-peers · search · representation"),
        ("",           "create · delete · get-metadata · set-metadata"),
        ("message",    "list · get · create"),
        ("conclusion", "list · search · create · delete"),
        ("workspace",  "list · inspect · create · delete · search · queue-status"),
        ("config",     "inspect current configuration"),
    ]

    console.print(_welcome_panel("getting started", start_rows))
    console.print(_welcome_panel("memory", memory_rows))
    console.print(_welcome_panel("commands", cmd_rows))
    console.print()


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

    _global_overrides["workspace"] = workspace
    _global_overrides["peer"] = peer
    _global_overrides["session"] = session

    if ctx.invoked_subcommand is None:
        _print_welcome(Console())
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
