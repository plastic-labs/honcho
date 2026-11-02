"""``honcho install`` — find the agents on this machine and add Honcho to them.

Finding agents is a filesystem check (see ``honcho_cli.agents``). Adding Honcho
runs only each agent's own commands: plugin installs for Claude Code, Hermes,
OpenClaw and Pi, ``codex mcp add`` and the honcho-memory skill for Codex. All
of them use the API key in ``~/.honcho/config.json``. Other agents are pointed
at their setup guide, and Grok Bot gets a prompt to paste into a Bot.

On a terminal with no arguments it shows a checkbox picker. Run from inside an
agent's tool call (Claude Code, Codex), it adds Honcho to that agent and never
prompts.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

import typer
from rich.console import Console

from honcho_cli.agents import (
    AGENTS,
    CATALOG_URL,
    GROK_BOT_PROMPT,
    Agent,
    Credentials,
    Detection,
    Quiet,
    calling_agent,
    detect_all,
)
from honcho_cli.branding import BRAND, ICON_FAIL, ICON_OK, ICON_RUN
from honcho_cli.config import CONFIG_FILE, DEFAULT_BASE_URL, CLIConfig
from honcho_cli.output import print_error, print_result, set_json_mode, use_json

_console = Console(stderr=True)

Runner = Callable[[Sequence[str], bool], int]

KEY_PLACEHOLDER = "<api-key>"


def _run(argv: Sequence[str], quiet: bool) -> int:
    """Run one install command with the terminal attached; return its exit code.

    argv[0] is resolved through PATH first so Windows npm ``.cmd`` shims run
    without a shell. In JSON mode the child's stdout goes to stderr, keeping
    our stdout for the JSON summary. A quiet step's output is captured and
    shown only if it fails.
    """
    exe = shutil.which(argv[0]) or argv[0]
    try:
        if quiet:
            proc = subprocess.run(
                [exe, *argv[1:]], check=False, capture_output=True, text=True, errors="replace",
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout + proc.stderr)
            return proc.returncode
        stdout = None
        if use_json():
            with contextlib.suppress(AttributeError, OSError, ValueError):
                stdout = sys.stderr.fileno()
        return subprocess.run([exe, *argv[1:]], check=False, stdout=stdout).returncode
    except OSError:
        return 127


# Swapped out in tests so nothing is actually installed.
runner: Runner = _run


def _mask(text: str, key: str) -> str:
    if not key or key == KEY_PLACEHOLDER:
        return text
    return text.replace(key, f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "****")


def _fmt(commands: list[list[str]], key: str) -> str:
    return _mask(" && ".join(shlex.join(argv) for argv in commands), key)


# --------------------------------------------------------------------------- #
# the API key and user peer every install uses


def _read_config() -> dict[str, Any] | None:
    """The config file as a dict: empty when missing, None when it exists but can't be parsed."""
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _saved_field(key: str) -> str:
    """A string field from the config file itself (not the env vars CLIConfig overlays)."""
    value = (_read_config() or {}).get(key)
    return value if isinstance(value, str) else ""


def _saved_key() -> str:
    return _saved_field("apiKey")


def _save_key(cfg: CLIConfig, key: str) -> None:
    cfg.api_key = key
    cfg.save()
    _console.print(f"  {ICON_OK} [dim]Saved the API key to {CONFIG_FILE}[/dim]")


def _ensure_api_key(interactive: bool) -> str | None:
    """The key in ``~/.honcho/config.json``, asking for one (or saving HONCHO_API_KEY) if absent.

    Every agent reads its key from that file or gets it copied from there, so
    nothing is installed without one. A key only in HONCHO_API_KEY is saved to
    the file, because the OpenClaw and Hermes gateways run as services and never
    see the shell's environment. A browser (device) login stores an OAuth token
    instead, which the plugins can't use.
    """
    env_key = os.environ.get("HONCHO_API_KEY", "").strip()
    key = _saved_key()
    if key:
        if env_key and env_key != key:
            _console.print(
                f"  [yellow]![/yellow] HONCHO_API_KEY differs from the key in {CONFIG_FILE}. "
                "Agents read the file, so that key is the one used."
            )
        return key
    from honcho_cli.commands.setup import _test_connection

    cfg = CLIConfig.load()
    if env_key:
        ok, detail = _test_connection(cfg.base_url, env_key)
        if ok:
            _save_key(cfg, env_key)
            return env_key
        _console.print(f"  {ICON_FAIL} HONCHO_API_KEY was rejected: {detail}")

    if cfg.oauth and cfg.oauth.access_token:
        why = "You're signed in with a browser login, but agents need an API key."
    else:
        why = f"Agents read it from {CONFIG_FILE}."
    if not interactive:
        _console.print(f"  {ICON_FAIL} No Honcho API key yet. {why} Run: honcho init --api-key <key>")
        return None

    hosted = cfg.base_url.rstrip("/") == DEFAULT_BASE_URL
    where = "Create one at https://app.honcho.dev and paste it here" if hosted else f"Paste an API key for {cfg.base_url}"
    _console.print(f"  [yellow]![/yellow] No Honcho API key yet. {why}")
    _console.print(f"  [dim]{where}. Press Enter to cancel.[/dim]")
    while True:
        key = typer.prompt("  API key", default="", show_default=False, hide_input=True).strip()
        if not key:
            return None
        ok, detail = _test_connection(cfg.base_url, key)
        if ok:
            _save_key(cfg, key)
            return key
        _console.print(f"  {ICON_FAIL} {detail}. Try again, or press Enter to cancel.")


def _default_peer_name() -> str:
    try:
        return getpass.getuser() or "user"
    except Exception:  # no username in the environment or the password database
        return "user"


def _ensure_peer_name(interactive: bool) -> None:
    """A shared ``peerName`` in ``~/.honcho/config.json``, asking for one if absent.

    Hermes saves nothing from CLI turns until it knows the user's peer, and the
    other plugins fall back to the same field, so every agent names the user the
    same way. Defaults to the OS username, as the Claude Code plugin does.
    """
    if _saved_field("peerName"):
        return
    data = _read_config()
    if data is None:
        _console.print(f"  [yellow]![/yellow] Couldn't read {CONFIG_FILE}, so no peerName was saved.")
        return
    default = _default_peer_name()
    name = typer.prompt("  What should your agents call you?", default=default).strip() if interactive else ""
    data["peerName"] = name or default
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")
    with contextlib.suppress(OSError):
        os.chmod(CONFIG_FILE, 0o600)
    _console.print(f"  {ICON_OK} [dim]Agents will call you {data['peerName']} (peerName in {CONFIG_FILE})[/dim]")


# --------------------------------------------------------------------------- #
# rendering


def _preview(d: Detection, creds: Credentials) -> str:
    assert d.agent.install is not None
    return _fmt(d.agent.install(creds), creds.api_key)


def _print_status(detections: list[Detection], creds: Credentials, caller: Agent | None) -> None:
    installable = [d for d in detections if d.found and d.agent.install]
    guides = [d for d in detections if d.found and not d.agent.install]
    missing = [d.agent.name for d in detections if not d.found and not d.agent.cloud]
    width = max((len(d.agent.name) for d in installable + guides), default=0) + 2

    _console.print()
    if caller:
        _console.print(f"  [dim]Running inside {caller.name}: setting up {caller.name} only, no prompts.[/dim]\n")
    if installable:
        _console.print("  [bold]Agents found on this machine[/bold]\n")
        for d in installable:
            if d.installed:
                icon, detail = ICON_OK, "[green]Honcho installed[/green]"
            elif d.blocked:
                icon, detail = "[yellow]![/yellow]", f"[yellow]{d.blocked}[/yellow]"
            else:
                note = "  [dim](couldn't check if already added)[/dim]" if d.installed is None else ""
                icon, detail = f"[{BRAND}]●[/{BRAND}]", f"{_preview(d, creds)}{note}"
            _console.print(f"  {icon} {d.agent.name:<{width}}{detail}", highlight=False)
    else:
        names = ", ".join(a.name for a in AGENTS if a.install)
        _console.print(f"  [bold]None of these is on this machine: {names}[/bold]")
    if guides:
        _console.print("\n  [bold]Also found[/bold] [dim](set up from the guide)[/dim]\n")
        for d in guides:
            _console.print(f"  [dim]○[/dim] {d.agent.name:<{width}}[dim]{d.agent.docs_url}[/dim]", highlight=False)
    _console.print("\n  [dim]Grok Bot: `honcho install grok-bot` copies a setup prompt to paste into any Bot.[/dim]")
    if missing:
        _console.print(f"  [dim]Not found: {', '.join(missing)}[/dim]")
    _console.print(f"  [dim]Full list: {CATALOG_URL}[/dim]\n")


def _as_json(detections: list[Detection], creds: Credentials) -> list[dict[str, Any]]:
    return [
        {
            "id": d.agent.id,
            "name": d.agent.name,
            "kind": d.agent.kind,
            "found": d.found,
            "installed": d.installed,
            "blocked": d.blocked,
            "command": _preview(d, creds) if d.found and d.agent.install and not d.installed else None,
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
    choices = [
        questionary.Choice(
            title=d.agent.name if d.agent.install else f"{d.agent.name}  (copies a setup prompt)",
            value=d.agent.id,
            checked=bool(d.agent.install) and not d.blocked,
            disabled=d.blocked or None,
        )
        for d in candidates
    ]
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


def _copy(text: str) -> bool:
    for argv in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip"]):
        exe = shutil.which(argv[0])
        if not exe:
            continue
        try:
            subprocess.run([exe, *argv[1:]], input=text, text=True, check=True, timeout=5)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _show_guide(agent: Agent) -> None:
    if agent.id != "grok-bot":
        _console.print(f"  {ICON_RUN} {agent.name}: follow {agent.docs_url}")
        return
    copied = " (copied to your clipboard)" if _copy(GROK_BOT_PROMPT) else ""
    _console.print(f"\n  {ICON_RUN} [bold]Grok Bot[/bold]: paste this into any Bot{copied}:\n")
    _console.print(f"  {GROK_BOT_PROMPT}\n", highlight=False, soft_wrap=True)
    _console.print(f"  [dim]The Bot signs you in to Honcho itself. Guide: {agent.docs_url}[/dim]")


def _install_one(agent: Agent, creds: Credentials) -> bool:
    assert agent.install is not None
    for argv in agent.install(creds):
        shown = _fmt([argv], creds.api_key)
        _console.print(f"\n  {ICON_RUN} [bold]{agent.name}[/bold]  [dim]{shown}[/dim]", highlight=False)
        code = runner(argv, isinstance(argv, Quiet))
        if code != 0:
            _console.print(f"  {ICON_FAIL} {agent.name}: exited {code}. Retry with: {shown}", highlight=False)
            return False
    if agent.has_honcho() is False:
        _console.print(f"  {ICON_FAIL} {agent.name}: the commands ran, but {agent.name} doesn't report Honcho yet.")
        return False
    _console.print(f"  {ICON_OK} {agent.name}: Honcho added")
    for note in agent.notes():
        _console.print(f"  [yellow]![/yellow] {note}", highlight=False)
    return True


def _run_installs(chosen: list[Detection], interactive: bool, results: dict[str, str]) -> int:
    """Install or show each row, recording the outcome in results."""
    if not chosen:
        _console.print("  Nothing to install.")
        return 0
    creds: Credentials | None = None
    if any(d.agent.install for d in chosen):
        key = _ensure_api_key(interactive)
        if key is None:
            for d in chosen:
                if d.agent.install:
                    results[d.agent.id] = "no_api_key"
            _console.print("  Nothing installed: every agent here needs the API key.")
            return 1
        _ensure_peer_name(interactive)
        creds = Credentials(api_key=key, base_url=CLIConfig.load().base_url)

    failed: list[str] = []
    done: list[Agent] = []
    for d in chosen:
        if not d.agent.install:
            _show_guide(d.agent)
            results[d.agent.id] = "shown"
            continue
        assert creds is not None
        ok = _install_one(d.agent, creds)
        results[d.agent.id] = "installed" if ok else "failed"
        if ok:
            done.append(d.agent)
        else:
            failed.append(d.agent.name)
    _console.print()
    if done:
        _console.print(f"  Restart {', '.join(a.name for a in done)} to load Honcho.")
    for agent in done:
        if agent.activate:
            _console.print(f"  {agent.name}: run [bold]{agent.activate}[/bold] to activate.")
    if failed:
        _console.print(f"  [red]{len(failed)} failed:[/red] {', '.join(failed)}")
        return 1
    return 0


def _resolve_targets(
    targets: list[str], detections: list[Detection], results: dict[str, str]
) -> tuple[list[Detection], bool]:
    """Turn ids (or ``all``) into rows to install. Returns (rows, any_problem)."""
    if "all" in targets:
        chosen = []
        for d in detections:
            if not (d.found and d.agent.install) or d.installed:
                continue
            if d.blocked:
                results[d.agent.id] = "blocked"
                _console.print(f"  [dim]Skipping {d.agent.name}: {d.blocked}[/dim]")
            else:
                chosen.append(d)
        return chosen, False

    by_id = {d.agent.id: d for d in detections}
    chosen: list[Detection] = []
    problem = False
    for t in dict.fromkeys(targets):
        d = by_id[t]
        if not d.agent.install:
            chosen.append(d)
        elif not d.found:
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
        help=f"Agents to set up without the picker, or 'all' for every one found. Ids: {', '.join(a.id for a in AGENTS)}",
        show_default=False,
    ),
    list_only: bool = typer.Option(False, "--list", help="Only show what was found; install nothing"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Find the AI agents on this machine and add Honcho to them.

    Claude Code, Hermes, OpenClaw and Pi get their Honcho plugin, and Codex the
    Honcho MCP server and skill, each through the agent's own commands and the
    API key in ~/.honcho/config.json. Other agents link to their setup guide.
    """
    if json_output:
        set_json_mode(True)

    creds = Credentials(api_key=_saved_key() or KEY_PLACEHOLDER, base_url=CLIConfig.load().base_url)
    detections = detect_all(creds)
    caller = None if targets or list_only else calling_agent()
    if caller:
        targets = [caller.id]

    if use_json() and (not targets or list_only):
        print_result(_as_json(detections, creds))
        return

    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not use_json() and caller is None
    if not use_json():
        _print_status(detections, creds, caller)
    if list_only:
        return

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

    candidates = [d for d in detections if d.found and d.agent.install and not d.installed]
    candidates += [d for d in detections if d.agent.cloud]
    if not interactive:
        _console.print("  No terminal to ask on. Run `honcho install all`, or `honcho install <id>`.\n")
        raise typer.Exit(1)
    chosen = _pick(candidates)
    if chosen is None:
        _console.print("  Cancelled.")
        raise typer.Exit(1)
    raise typer.Exit(_run_installs(chosen, interactive, results))
