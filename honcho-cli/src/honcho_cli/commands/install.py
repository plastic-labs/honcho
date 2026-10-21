"""`honcho install` — detect local AI agents and add Honcho to them.

Detects catalog plugin agents and MCP clients, displays Honcho status,
and installs plugins or opens MCP guides via an interactive picker or direct IDs.
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser

import typer
from rich.console import Console
from rich.table import Table

from honcho_cli.agents import Agent, detect_agents, find_agent, get_all_agents
from honcho_cli.branding import BRAND, ICON_FAIL, ICON_OK
from honcho_cli.config import CONFIG_FILE, CLIConfig
from honcho_cli.output import print_json, set_json_mode
from honcho_cli.picker import prompt_checkbox_picker

_console = Console()


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ensure_api_key() -> str:
    """Ensure an API key is persisted in ~/.honcho/config.json before installing plugins."""
    cfg = CLIConfig.load()
    if cfg.api_key:
        # load() overlays HONCHO_API_KEY; persist it for sibling plugins too.
        cfg.save()
        return cfg.api_key

    _console.print(
        "\n[bold yellow]No Honcho API key configured in ~/.honcho/config.json[/bold yellow]"
    )
    _console.print(
        "[dim]Most agent plugins read your API key from ~/.honcho/config.json.[/dim]"
    )
    try:
        key = typer.prompt(
            "  Enter your Honcho API key (get one at https://app.honcho.dev)",
            default="",
            hide_input=True,
        ).strip()
    except (typer.Abort, EOFError, OSError):
        key = ""

    if key:
        cfg.api_key = key
        cfg.save()
        _console.print(f"  {ICON_OK} [dim]Saved API key to {CONFIG_FILE}[/dim]\n")
        return key

    _console.print(
        f"  {ICON_FAIL} No API key saved in {CONFIG_FILE}; installation cancelled."
    )
    raise typer.Exit(code=1)


def install(
    targets: list[str] = typer.Argument(
        None,
        help="Agent IDs to install (or 'all' for all detected agents without Honcho)",
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List detected agents and Honcho status without installing",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output detected agents and status as JSON without installing",
    ),
    all_catalog: bool = typer.Option(
        False,
        "--all",
        hidden=True,
        help="Include all catalog agents in report even if not detected",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="DeepSeek Harness profile to install into",
    ),
) -> None:
    """Detect local AI agents and install Honcho memory.

    Detects which agents from the integrations catalog are on your machine,
    checks if Honcho is already configured, and lets you install plugins
    or view MCP setup guides.
    """
    if json_output:
        set_json_mode(True)

    detected = detect_agents()
    all_agents = get_all_agents()

    # --json and --list only report
    if json_output:
        report_agents = all_agents if all_catalog else detected
        print_json([a.to_dict(detected=a in detected) for a in report_agents])
        return

    if list_only:
        report_agents = all_agents if all_catalog else detected
        if not report_agents:
            _console.print("No supported AI agents detected on this machine.\n")
            _console.print(
                "Run [bold]honcho install --list --all[/bold] to see all catalog agents."
            )
            return

        table = Table(
            title="Honcho Integrations Catalog",
            show_header=True,
            header_style=f"bold {BRAND}",
        )
        table.add_column("Agent", style="bold")
        table.add_column("Type")
        table.add_column("Detected")
        table.add_column("Honcho Status")
        table.add_column("Install / Guide")

        for a in report_agents:
            is_det = a in detected
            det_str = "[green]Yes[/green]" if is_det else "[dim]No[/dim]"
            has_hch = a.has_honcho() if is_det else False
            hch_str = (
                "[green]Installed[/green]" if has_hch else "[dim]Not installed[/dim]"
            )
            action = a.get_command() if a.kind == "plugin" else a.guide_url
            table.add_row(a.name, a.kind, det_str, hch_str, f"[dim]{action}[/dim]")

        _console.print(table)
        return

    # Installation flow
    selected: list[Agent] = []

    if targets:
        target_tokens = [t.strip().lower() for t in targets]
        if "all" in target_tokens:
            selected = [
                a
                for a in detected
                if not a.has_honcho() or (a.id == "deepseek-harness" and profile)
            ]
            if not detected:
                _console.print("No supported AI agents detected on this machine.")
                raise typer.Exit()
            if not selected:
                _console.print("All detected agents already have Honcho installed.")
                raise typer.Exit()
        else:
            for t in targets:
                agent = find_agent(t)
                if not agent:
                    supported = ", ".join(a.id for a in all_agents)
                    _console.print(
                        f"[bold red]Unknown agent '{t}'.[/bold red] Supported agents:\n  {supported}"
                    )
                    raise typer.Exit(code=1)
                selected.append(agent)
    else:
        if not _is_interactive():
            _console.print(
                "No targets specified. Use 'honcho install all' or 'honcho install <id>...', or run interactively in a terminal."
            )
            raise typer.Exit(code=1)

        if not detected:
            _console.print("No supported AI agents detected on this machine.")
            _console.print(
                "Run 'honcho install <id>' to install for a specific agent, or see https://honcho.dev/docs/v3/guides/overview."
            )
            raise typer.Exit()

        selected = prompt_checkbox_picker(detected)
        if not selected:
            _console.print("No agents selected.")
            raise typer.Exit()

    if any(a.id == "deepseek-harness" for a in selected) and not profile:
        if not _is_interactive():
            _console.print(
                "DeepSeek Harness requires --profile <name> (for example, web or headless)."
            )
            raise typer.Exit(code=1)
        profile = typer.prompt("DeepSeek Harness profile").strip()
        if not profile:
            raise typer.Exit(code=1)

    # If any plugin agent is selected, ensure API key is configured
    if any(a.kind == "plugin" for a in selected):
        _ensure_api_key()

    # Execute installs
    failed = False
    for agent in selected:
        if agent.kind == "plugin":
            _console.print(f"\n[bold]Installing Honcho for {agent.name}...[/bold]")
            cmd = agent.get_command()
            if agent.id == "deepseek-harness":
                args = [
                    "dsh",
                    "plugin",
                    "--profile",
                    profile or "",
                    "add",
                    "@honcho-ai/dsh-honcho",
                ]
                _console.print(
                    f"  [dim]$ dsh plugin --profile {profile} add @honcho-ai/dsh-honcho[/dim]"
                )
                ret = subprocess.run(args).returncode
            else:
                _console.print(f"  [dim]$ {cmd}[/dim]")
                ret = subprocess.run(cmd, shell=True).returncode
            if ret == 0:
                _console.print(
                    f"  {ICON_OK} Successfully installed Honcho for {agent.name}"
                )
            else:
                failed = True
                _console.print(
                    f"  {ICON_FAIL} Failed to install Honcho for {agent.name} (exit code {ret})"
                )
        elif agent.kind == "mcp":
            _console.print(
                f"\n[bold]Opening {agent.name} MCP guide in your browser...[/bold]"
            )
            _console.print(f"  [dim]{agent.guide_url}[/dim]")
            if webbrowser.open(agent.guide_url):
                _console.print(f"  {ICON_OK} Opened guide for {agent.name}")
            else:
                failed = True
                _console.print(
                    f"  {ICON_FAIL} Could not open a browser; visit the guide above."
                )
    if failed:
        raise typer.Exit(code=1)
