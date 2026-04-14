"""Top-level onboarding and health-check commands.

`honcho init`    — confirm or set apiKey + Honcho URL in ~/.honcho/config.json
`honcho doctor`  — verify connectivity, config validity, queue health
"""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.panel import Panel

from honcho_cli import __version__
from honcho_cli.config import (
    CONFIG_FILE,
    DEFAULT_BASE_URL,
    CLIConfig,
)
from honcho_cli.branding import BANNER, BRAND, ICON_FAIL, ICON_OK, ICON_RUN
from honcho_cli.output import print_error, print_result

_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# shared helpers

def _redact(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 16:
        return "***"
    return api_key[:8] + "..." + api_key[-4:]


def _read_file_values() -> tuple[str, str]:
    """Return (apiKey, environmentUrl) persisted on disk (or empty strings)."""
    if not CONFIG_FILE.exists():
        return "", ""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    key = data.get("apiKey") if isinstance(data.get("apiKey"), str) else ""
    url = data.get("environmentUrl") if isinstance(data.get("environmentUrl"), str) else ""
    return key, url


def _test_connection(base_url: str, api_key: str) -> tuple[bool, str]:
    """Probe the Honcho API by listing workspaces. Returns (ok, detail)."""
    try:
        from honcho import Honcho

        list(Honcho(base_url=base_url, api_key=api_key).workspaces())
        return True, "OK"
    except Exception as e:
        msg = str(e)
        if "Connection refused" in msg or "ConnectError" in msg:
            return False, "Connection refused — is the server running?"
        if "timed out" in msg.lower() or "Timeout" in msg:
            return False, "Request timed out"
        if "401" in msg or "Unauthorized" in msg:
            return False, "Unauthorized — check your API key"
        return False, msg


def _resolve_source(param: str | None, env_val: str, env_name: str, flag_name: str, file_val: str) -> tuple[str, str]:
    """Return (value, source_label) for one field, flag/env > file."""
    if param:
        return param, f"{env_name} env var" if env_val and env_val == param else f"{flag_name} flag"
    return (file_val, str(CONFIG_FILE)) if file_val else ("", "")


# --------------------------------------------------------------------------- #
# honcho init

def init(
    api_key: str | None = typer.Option(None, "--api-key", envvar="HONCHO_API_KEY", help="API key (admin JWT)"),
    base_url: str | None = typer.Option(None, "--base-url", envvar="HONCHO_BASE_URL", help="Honcho API URL (e.g. https://api.honcho.dev, http://localhost:8000)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Confirm or set ``apiKey`` + Honcho URL in ``~/.honcho/config.json``.

    For each value we either (a) show what we have and ask for a Y/N
    confirmation, or (b) prompt for it if missing. Foreign top-level keys
    (``hosts``, ``sessions``, …) are preserved.

    Workspace / peer / session scoping is per-command via ``-w`` / ``-p`` /
    ``-s`` or ``HONCHO_*`` env vars — never persisted.
    """
    from honcho_cli.output import set_json_mode, use_json

    if json_output:
        set_json_mode(True)

    file_key, file_url = _read_file_values()
    key_val, key_src = _resolve_source(
        api_key, os.environ.get("HONCHO_API_KEY", ""), "HONCHO_API_KEY", "--api-key", file_key,
    )
    url_val, url_src = _resolve_source(
        base_url, os.environ.get("HONCHO_BASE_URL", ""), "HONCHO_BASE_URL", "--base-url", file_url,
    )
    url_val = url_val.strip()

    if not use_json():
        _console.print()
        _console.print(Panel(
            f"[bold {BRAND}]{BANNER}[/bold {BRAND}]\n\n     Memory that reasons",
            expand=False, subtitle=f"Honcho CLI · v{__version__}",
        ))
        _console.print()
        if not key_val and not url_val:
            _console.print(f"[bold]No existing config at {CONFIG_FILE} — let's create one.[/bold]\n")

    final_key = _confirm_or_prompt_api_key(key_val, key_src)
    final_url = _confirm_or_prompt_url(url_val, url_src)

    # Persist if anything changed or if the value came from env/flag.
    if final_key != file_key or final_url != file_url:
        CLIConfig(base_url=final_url, api_key=final_key).save()
        if not use_json():
            _console.print(f"  {ICON_OK} [dim]Saved to {CONFIG_FILE}[/dim]")

    _check_connection(final_url, final_key)

    if use_json():
        print_result({"apiKey": _redact(final_key), "baseUrl": final_url})


def _confirm_or_prompt_api_key(value: str, source: str) -> str:
    from honcho_cli.output import use_json

    if value:
        if not use_json():
            _console.print(f"  API key:     [dim]{_redact(value)}[/dim]  [{BRAND}](from {source})[/{BRAND}]")
        if use_json() or typer.confirm("  Use this API key?", default=True):
            return value
    # Never set ``default=`` to the raw key — typer would echo it in brackets.
    new = typer.prompt("  API key")
    if not new:
        print_error("MISSING_VALUE", "API key is required", {})
        raise typer.Exit(1)
    return new


def _confirm_or_prompt_url(value: str, source: str) -> str:
    from honcho_cli.output import use_json

    if value:
        if not use_json():
            _console.print(f"  Honcho URL:  [dim]{value}[/dim]  [{BRAND}](from {source})[/{BRAND}]")
        if use_json() or typer.confirm("  Use this URL?", default=True):
            return value
    if not use_json():
        _console.print("  [dim](e.g. https://api.honcho.dev for managed, http://localhost:8000 for local)[/dim]")
    return typer.prompt("  Honcho URL", default=DEFAULT_BASE_URL).strip()


def _check_connection(base_url: str, api_key: str) -> None:
    from honcho_cli.output import use_json

    if not use_json():
        _console.print(f"\n  {ICON_RUN} [dim]Testing connection to {base_url}...[/dim]", end=" ")
    ok, detail = _test_connection(base_url, api_key)
    if not ok:
        if use_json():
            print_error("CONNECTION_FAILED", detail, {"base_url": base_url})
        else:
            _console.print(f"{ICON_FAIL} [red]Failed[/red]: {detail}")
        raise typer.Exit(1)
    if not use_json():
        _console.print(f"{ICON_OK} [green]Connected[/green]")


# --------------------------------------------------------------------------- #
# honcho doctor

def doctor(
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Verify config, connectivity, and — when scoped via ``-w`` / ``-p`` —
    workspace, peer, and queue health.
    """
    from honcho_cli.output import set_json_mode, use_json

    if json_output:
        set_json_mode(True)

    checks: list[dict] = []

    def _add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})
        if not use_json():
            icon = ICON_OK if ok else ICON_FAIL
            line = f"  {icon}  {name:<22}"
            if detail:
                line += f"  [dim]{detail}[/dim]"
            _console.print(line)

    if not use_json():
        _console.print(f"\n[bold {BRAND}]Honcho Doctor[/bold {BRAND}]\n")

    from honcho_cli.main import get_resolved_config

    config = get_resolved_config()
    _add("Config file", CONFIG_FILE.exists(),
         str(CONFIG_FILE) if CONFIG_FILE.exists() else f"{CONFIG_FILE} not found")
    _add("API key configured", bool(config.api_key),
         "set" if config.api_key else "missing — run `honcho init`")

    if config.base_url and config.api_key:
        _add("API connectivity", *_test_connection(config.base_url, config.api_key))
    else:
        _add("API connectivity", False, "skipped — no base_url or api_key")

    # Workspace / peer / queue run only when scoped via -w / -p.
    ws_ok, client = False, None
    if config.workspace_id and config.api_key:
        try:
            from honcho import Honcho

            client = Honcho(base_url=config.base_url, api_key=config.api_key, workspace_id=config.workspace_id)
            client.get_configuration()
            ws_ok = True
            _add("Workspace reachable", True, config.workspace_id)
        except Exception as e:
            _add("Workspace reachable", False, f"{config.workspace_id}: {e}")
        if ws_ok:
            try:
                q = client.queue_status()
                _add("Queue health", True, f"{q.completed_work_units}/{q.total_work_units} completed, {q.pending_work_units} pending")
            except Exception:
                _add("Queue health", True, "endpoint not available (non-critical)")

    if config.peer_id:
        if ws_ok and client is not None:
            try:
                client.peer(config.peer_id).get_card()
                _add("Peer exists", True, config.peer_id)
            except Exception as e:
                _add("Peer exists", False, f"{config.peer_id}: {e}")
        else:
            _add("Peer exists", False, "skipped — workspace not reachable")

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)

    if use_json():
        print_result({"checks": checks, "passed": passed, "total": total})
    else:
        color = BRAND if passed == total else ("yellow" if passed > total // 2 else "red")
        hint = "" if config.workspace_id else "  [dim](pass -w / -p to include workspace, peer, queue checks)[/dim]"
        _console.print(f"\n  [{color}]{passed}/{total}[/{color}] checks passed{hint}\n")

    # apiKey must live in config.json → missing file is a hard failure.
    critical = {"Config file", "API key configured", "API connectivity"}
    if config.workspace_id:
        critical.add("Workspace reachable")
    if any(not c["ok"] for c in checks if c["check"] in critical):
        raise typer.Exit(1)
