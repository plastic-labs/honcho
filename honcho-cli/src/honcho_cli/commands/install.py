"""``honcho install`` — find the agents on this machine and add Honcho to them.

Detection is read-only (binaries on PATH, config files on disk). On a TTY the
command then shows a checkbox picker: ticking a plugin agent runs its install
command with the terminal attached, ticking an MCP client opens its setup
guide. ``--list``, ``--json`` and non-TTY runs only print what was found.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Sequence
from typing import Any

import typer
from rich.console import Console

from honcho_cli.agents import AGENTS, CATALOG_URL, Detection, detect_all, dsh_profiles
from honcho_cli.branding import BRAND, ICON_FAIL, ICON_OK, ICON_RUN
from honcho_cli.config import CONFIG_FILE, DEFAULT_BASE_URL, CLIConfig
from honcho_cli.output import print_error, print_result, set_json_mode, use_json

_console = Console(stderr=True)

Runner = Callable[[Sequence[str]], int]


def _run(argv: Sequence[str]) -> int:
    """Run one install command with the terminal attached; return its exit code.

    argv[0] is resolved through PATH first so Windows npm ``.cmd`` shims run
    without a shell. In JSON mode the child's stdout goes to stderr, keeping
    our stdout for the JSON summary.
    """
    exe = shutil.which(argv[0]) or argv[0]
    stdout = None
    if use_json():
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stdout = sys.stderr.fileno()
    try:
        return subprocess.run([exe, *argv[1:]], check=False, stdout=stdout).returncode
    except OSError:
        return 127


# Swapped out in tests so nothing is actually installed.
runner: Runner = _run


def _fmt(commands: list[list[str]]) -> str:
    # shlex quotes the dsh "<profile>" placeholder; show it bare
    return " && ".join(shlex.join(argv) for argv in commands).replace("'<profile>'", "<profile>")


def _has_credentials() -> bool:
    """True when plugins will find an API key (env var or shared config)."""
    return bool(os.environ.get("HONCHO_API_KEY") or CLIConfig.load().api_key)


# --------------------------------------------------------------------------- #
# rendering


def _status_rows(found: list[Detection]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for d in found:
        if d.installed:
            rows.append((ICON_OK, d.agent.name, "[green]Honcho installed[/green]"))
        elif d.blocked:
            rows.append(("[yellow]![/yellow]", d.agent.name, f"[yellow]{d.blocked}[/yellow]"))
        elif d.agent.kind == "plugin":
            note = "  [dim](can't check if already added)[/dim]" if d.installed is None else ""
            rows.append((f"[{BRAND}]●[/{BRAND}]", d.agent.name, f"{_fmt(d.commands)}{note}"))
        else:
            rows.append((f"[{BRAND}]●[/{BRAND}]", d.agent.name, f"[dim]add the MCP server[/dim]  {d.agent.docs_url}"))
    return rows


def _print_status(detections: list[Detection]) -> None:
    found = [d for d in detections if d.found]
    missing = [d.agent.name for d in detections if not d.found]
    width = max((len(d.agent.name) for d in found), default=0) + 2

    _console.print()
    if found:
        _console.print("  [bold]Agents found on this machine[/bold]\n")
        for icon, name, detail in _status_rows(found):
            _console.print(f"  {icon} {name:<{width}}{detail}", highlight=False)
    else:
        _console.print("  [bold]No supported agents found on this machine[/bold]")
    if missing:
        _console.print(f"\n  [dim]Not found: {', '.join(missing)}[/dim]")
    _console.print(f"  [dim]Full list: {CATALOG_URL}[/dim]\n")


def _as_json(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "id": d.agent.id,
            "name": d.agent.name,
            "kind": d.agent.kind,
            "found": d.found,
            "installed": d.installed,
            "blocked": d.blocked,
            "command": _fmt(d.commands) if d.commands else None,
            "docs_url": d.agent.docs_url,
        }
        for d in detections
    ]


# --------------------------------------------------------------------------- #
# picker + installs


def _pick(candidates: list[Detection]) -> list[Detection] | None:
    """Checkbox picker. Returns the ticked rows, or None if the user cancelled."""
    import questionary

    style = questionary.Style([
        ("qmark", f"fg:{BRAND} bold"),
        ("pointer", f"fg:{BRAND} bold"),
        ("highlighted", f"fg:{BRAND} bold"),
        ("selected", f"fg:{BRAND}"),
        ("instruction", "fg:#808080"),
        ("disabled", "fg:#808080 italic"),
    ])
    choices: list[questionary.Choice] = []
    for d in candidates:
        if d.agent.kind == "mcp":
            title = f"{d.agent.name}  (MCP, opens the setup guide)"
        else:
            title = d.agent.name
        choices.append(questionary.Choice(
            title=title,
            value=d.agent.id,
            checked=d.agent.kind == "plugin" and not d.blocked and d.installed is False,
            disabled=d.blocked or None,
        ))
    picked = questionary.checkbox(
        "Add Honcho to which agents?",
        choices=choices,
        instruction="(space to toggle, a for all, enter to confirm)",
        style=style,
    ).ask()
    if picked is None:
        return None
    by_id = {d.agent.id: d for d in candidates}
    return [by_id[i] for i in picked]


def _resolve_commands(d: Detection, interactive: bool) -> list[list[str]] | None:
    """Fill in anything the commands still need from the user (dsh profile)."""
    if "<profile>" not in _fmt(d.commands):
        return d.commands
    if not interactive:
        return None
    # dsh has no default profile, and an unknown name silently creates a new one
    profiles = dsh_profiles()
    without = [p for p, has in profiles.items() if has is False]
    if profiles:
        _console.print(f"  [dim]{d.agent.name} profiles: {', '.join(profiles)}[/dim]")
    default = (without or list(profiles) or [None])[0]
    profile = typer.prompt(f"  {d.agent.name} profile to add Honcho to", default=default).strip()
    return [[a.replace("<profile>", profile) for a in argv] for argv in d.commands]


def _install_one(d: Detection, interactive: bool) -> bool:
    if d.agent.kind == "mcp":
        _console.print(f"  {ICON_RUN} {d.agent.name}: opening {d.agent.docs_url}")
        with contextlib.suppress(Exception):  # headless is fine — the URL is printed above
            webbrowser.open(d.agent.docs_url)
        return True

    if d.agent.install_interactive and not interactive:
        _console.print(f"  {ICON_FAIL} {d.agent.name}: setup is interactive, run it yourself: {_fmt(d.commands)}")
        return False
    commands = _resolve_commands(d, interactive)
    if commands is None:
        _console.print(f"  {ICON_FAIL} {d.agent.name}: needs a profile name, run it yourself: {_fmt(d.commands)}")
        return False

    for argv in commands:
        _console.print(f"\n  {ICON_RUN} [bold]{d.agent.name}[/bold]  [dim]{shlex.join(argv)}[/dim]", highlight=False)
        code = runner(argv)
        if code != 0:
            _console.print(f"  {ICON_FAIL} {d.agent.name}: exited {code}. Retry with: {shlex.join(argv)}", highlight=False)
            return False
    _console.print(f"  {ICON_OK} {d.agent.name}: Honcho added")
    return True


def _ensure_api_key(interactive: bool) -> None:
    """Plugins read a long-lived API key from the shared config.

    A browser (device) login stores an OAuth token instead, which most plugins
    can't use and which ``honcho init``'s default "keep current credentials"
    would leave in place, so ask for the key directly.
    """
    if _has_credentials():
        return
    cfg = CLIConfig.load()
    if cfg.oauth and cfg.oauth.access_token:
        why = "You're signed in with a browser login, but most plugins only read an API key."
    else:
        why = f"Most plugins read one from {CONFIG_FILE}."
    if not interactive:
        _console.print(f"  [yellow]![/yellow] No Honcho API key yet. {why} Run: honcho init --api-key <key>")
        return

    hosted = cfg.base_url.rstrip("/") == DEFAULT_BASE_URL
    where = "Create one at https://app.honcho.dev and paste it here" if hosted else f"Paste an API key for {cfg.base_url}"
    _console.print(f"  [yellow]![/yellow] No Honcho API key yet. {why}")
    _console.print(f"  [dim]{where}. Press Enter to skip.[/dim]")
    from honcho_cli.commands.setup import _test_connection

    while True:
        key = typer.prompt("  API key", default="", show_default=False, hide_input=True).strip()
        if not key:
            _console.print("  [dim]Skipped. Plugins that need a key will say so.[/dim]")
            return
        ok, detail = _test_connection(cfg.base_url, key)
        if ok:
            break
        _console.print(f"  {ICON_FAIL} {detail}. Try again, or press Enter to skip.")
    cfg.api_key = key
    cfg.save()
    _console.print(f"  {ICON_OK} [dim]Saved to {CONFIG_FILE}[/dim]")


def _auto_installable(d: Detection) -> bool:
    """Rows ``all`` may run unattended: known missing, no prompts, no placeholders."""
    return (
        d.agent.kind == "plugin"
        and not d.blocked
        and d.installed is False
        and not d.agent.install_interactive
        and "<profile>" not in _fmt(d.commands)
    )


def _run_installs(chosen: list[Detection], interactive: bool, results: dict[str, str]) -> int:
    """Install each row, recording ``installed`` / ``opened`` / ``failed`` in results."""
    if not chosen:
        _console.print("  Nothing to install.")
        return 0
    if any(d.agent.kind == "plugin" for d in chosen):
        _ensure_api_key(interactive)
    failed: list[str] = []
    done: list[str] = []
    for d in chosen:
        ok = _install_one(d, interactive)
        results[d.agent.id] = ("opened" if d.agent.kind == "mcp" else "installed") if ok else "failed"
        if not ok:
            failed.append(d.agent.name)
        elif d.agent.kind == "plugin":
            done.append(d.agent.name)
    _console.print()
    if done:
        _console.print(f"  Restart {', '.join(done)} to load Honcho.")
    for d in chosen:
        if d.agent.activate and results.get(d.agent.id) == "installed":
            _console.print(f"  {d.agent.name}: run [bold]{d.agent.activate}[/bold] to activate.")
    if failed:
        _console.print(f"  [red]{len(failed)} failed:[/red] {', '.join(failed)}")
        return 1
    return 0


def _resolve_targets(
    targets: list[str], detections: list[Detection], results: dict[str, str]
) -> tuple[list[Detection], bool]:
    """Turn ids (or ``all``) into rows to install. Returns (rows, any_problem)."""
    candidates = [d for d in detections if d.found and not d.installed]
    if "all" in targets:
        for d in candidates:
            if d.agent.kind == "plugin" and d.blocked:
                results[d.agent.id] = "blocked"
                _console.print(f"  [dim]Skipping {d.agent.name}: {d.blocked}[/dim]")
            elif d.agent.kind == "plugin" and not _auto_installable(d):
                results[d.agent.id] = "needs_input"
                _console.print(f"  [dim]Skipping {d.agent.name}: setup needs input. Run: honcho install {d.agent.id}[/dim]")
        return [d for d in candidates if _auto_installable(d)], False

    by_id = {d.agent.id: d for d in detections}
    chosen: list[Detection] = []
    problem = False
    for t in dict.fromkeys(targets):
        d = by_id[t]
        if not d.found:
            results[t] = "not_found"
            problem = True
            _console.print(f"  {ICON_FAIL} {d.agent.name} isn't on this machine.")
        elif d.installed:
            results[t] = "already_installed"
            _console.print(f"  {ICON_OK} {d.agent.name} already has Honcho.")
        elif d.blocked:
            results[t] = "blocked"
            problem = True
            _console.print(f"  {ICON_FAIL} {d.agent.name}: {d.blocked}.")
        else:
            chosen.append(d)
    return chosen, problem


# --------------------------------------------------------------------------- #
# command


def install(
    targets: list[str] | None = typer.Argument(
        None,
        help=f"Agents to add Honcho to without the picker, or 'all' for every one found. Ids: {', '.join(a.id for a in AGENTS)}",
        show_default=False,
    ),
    list_only: bool = typer.Option(False, "--list", help="Only show what was found; install nothing"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Find the AI agents on this machine and add Honcho to them.

    With no arguments on a terminal, shows a checkbox picker. Plugin agents run
    their install command; MCP clients open their setup guide.
    """
    if json_output:
        set_json_mode(True)

    detections = detect_all()

    if use_json() and (not targets or list_only):
        print_result(_as_json(detections))
        return

    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not use_json()
    if not use_json():
        _print_status(detections)
    if list_only:
        return

    candidates = [d for d in detections if d.found and not d.installed]
    results: dict[str, str] = {}

    if targets:
        known = {a.id for a in AGENTS}
        unknown = [t for t in targets if t != "all" and t not in known]
        if unknown:
            print_error("UNKNOWN_AGENT", f"Unknown agent: {', '.join(unknown)}", {"known": sorted(known)})
            raise typer.Exit(2)
        chosen, problem = _resolve_targets(targets, detections, results)
        code = _run_installs(chosen, interactive, results)
        if use_json():
            print_result([{"id": k, "status": v} for k, v in results.items()])
        raise typer.Exit(1 if problem or code else 0)

    if not candidates:
        _console.print("  Nothing to add. Every agent found already has Honcho.\n")
        return
    if all(d.blocked for d in candidates):
        _console.print("  Nothing can be added here yet. See the notes above.\n")
        return
    if not interactive:
        return

    chosen = _pick(candidates)
    if chosen is None:
        _console.print("  Cancelled.")
        raise typer.Exit(1)
    raise typer.Exit(_run_installs(chosen, interactive, results))
