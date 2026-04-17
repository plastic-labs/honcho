"""Themed help rendering for honcho CLI.

Single source of truth for:

- Rich-utils theme constants (dim borders, brand color)
- HonchoTyperGroup: subclass applied via ``cls=`` at every Typer app in
  this package — replaces Click's terse ``Usage: …`` line with
  pattern/example rows and prints a curated welcome at the top-level.

Lives in its own module so every ``commands/*.py`` can import it without
pulling in ``main.py`` (which would create an import cycle).
"""

from __future__ import annotations

import click
import typer.rich_utils as ru
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperGroup

from honcho_cli import __version__
from honcho_cli.branding import BANNER, BRAND
from honcho_cli.output import use_json


# Theme Typer's rich help renderer. Module-level side effect limited to
# styling — no behavior changes that could surprise other Typer users.
ru.STYLE_COMMANDS_PANEL_BORDER = "dim"
ru.STYLE_OPTIONS_PANEL_BORDER = "dim"
ru.STYLE_ERRORS_PANEL_BORDER = "dim"
ru.STYLE_OPTION = f"bold {BRAND}"
ru.STYLE_SWITCH = f"bold {BRAND}"
ru.STYLE_USAGE = "dim"
ru.STYLE_USAGE_COMMAND = f"bold {BRAND}"


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


def print_welcome(console: Console) -> None:
    """Render the curated 3-panel welcome (banner + getting started / memory / commands)."""
    if use_json():
        return
    console.print(f"[bold {BRAND}]{BANNER}[/bold {BRAND}]")
    console.print(f"  [dim]v{__version__}[/dim]\n", highlight=False)

    start_rows = [
        ("honcho init",   "configure API key and server URL"),
        ("honcho doctor", "verify connection and workspace health"),
    ]
    cmd_rows = [
        ("[dim]pattern[/dim]", "[dim]honcho [-w workspace] [-p peer] [-s session] <command>[/dim]"),
        ("[dim]example[/dim]", "[dim]honcho -p alice peer chat \"what does alice prefer?\"[/dim]"),
        ("", ""),
        ("workspace",  "list · create · search · delete · inspect · queue-status"),
        ("peer",       "list · create · search · inspect · card · chat"),
        ("",           "get-metadata · set-metadata · representation"),
        ("session",    "list · create · search · delete · inspect · add-peers"),
        ("",           "context · get-metadata · set-metadata · peers"),
        ("",           "remove-peers · representation · summaries"),
        ("message",    "list · create · get"),
        ("conclusion", "list · create · search · delete"),
        ("config",     "inspect current configuration"),
    ]
    memory_rows = [
        ("honcho -w <workspace> -p <peer> peer chat \"...\"",   "query the dialectic about a peer"),
        ("honcho -w <workspace> -p <peer> peer inspect",        "review what honcho knows about a peer"),
        ("honcho -w <workspace> -p <peer> peer representation", "get the full peer representation"),
        ("honcho -w <workspace> -p <peer> conclusion list",     "browse peer memory atoms"),
    ]

    option_rows = [
        ("-w / --workspace", "scope to a workspace (env: HONCHO_WORKSPACE_ID)"),
        ("-p / --peer",      "scope to a peer (env: HONCHO_PEER_ID)"),
        ("-s / --session",   "scope to a session (env: HONCHO_SESSION_ID)"),
        ("--json",           "force JSON output for scripts and agents"),
        ("--help",           "show help for any command (e.g. honcho peer --help)"),
    ]

    console.print(_welcome_panel("getting started", start_rows))
    console.print(_welcome_panel("commands", cmd_rows))
    console.print(_welcome_panel("memory", memory_rows))
    console.print(_welcome_panel("options", option_rows))
    console.print()


class HonchoTyperGroup(TyperGroup):
    """Typer group with pattern/example usage and top-level welcome.

    Applied via ``cls=`` on every ``typer.Typer(...)`` in this package,
    so no class-level monkey-patching is needed.
    """

    def get_usage(self, ctx):
        """Replace Click's 'Usage: …' with pattern/example rows."""
        original = click.Command.get_usage(self, ctx)
        pattern = original.replace("Usage: ", "", 1) if original.startswith("Usage: ") else original
        subs = self.list_commands(ctx)
        example = f"{ctx.command_path} {subs[0]}" if subs else f"{ctx.command_path} --help"
        return f"pattern: {pattern}\nexample: {example}"

    def format_help(self, ctx, formatter):
        """Top-level --help renders the welcome; sub-groups fall through to Typer."""
        if ctx.parent is None:
            print_welcome(Console())
            return
        super().format_help(ctx, formatter)
