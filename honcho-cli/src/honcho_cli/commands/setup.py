"""Top-level onboarding and health-check commands.

`honcho init`    — confirm or set apiKey + Honcho URL in ~/.honcho/config.json
`honcho doctor`  — verify connectivity, config validity, queue health
"""

from __future__ import annotations

import json

import typer
from honcho import Honcho
from rich.console import Console
from rich.panel import Panel

from honcho_cli import __version__
from honcho_cli.branding import BANNER, BRAND, ICON_FAIL, ICON_OK, ICON_RUN
from honcho_cli.common import get_resolved_config
from honcho_cli.config import (
    CONFIG_FILE,
    DEFAULT_BASE_URL,
    CLIConfig,
)
from honcho_cli.output import print_error, print_result, set_json_mode, use_json

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


def _pick(flag_val: str | None, file_val: str) -> str:
    """Return best available value. Flag/env wins over file."""
    return flag_val or file_val or ""


# --------------------------------------------------------------------------- #
# honcho init

def init(
    api_key: str | None = typer.Option(None, "--api-key", envvar="HONCHO_API_KEY", help="API key (admin JWT)"),
    base_url: str | None = typer.Option(None, "--base-url", envvar="HONCHO_BASE_URL", help="Honcho API URL (e.g. https://api.honcho.dev, http://localhost:8000)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Set ``apiKey`` + Honcho URL in ``~/.honcho/config.json``.

    Each value is shown with its current default — press Enter to keep it or
    type a replacement. Foreign top-level keys (``hosts``, ``sessions``, …)
    are preserved.

    Workspace / peer / session scoping is per-command via ``-w`` / ``-p`` /
    ``-s`` or ``HONCHO_*`` env vars — never persisted.
    """

    if json_output:
        set_json_mode(True)

    file_key, file_url = _read_file_values()
    key_val = _pick(api_key, file_key)
    url_val = _pick(base_url, file_url).strip()

    if not use_json():
        _console.print()
        _console.print(Panel(
            f"[bold {BRAND}]{BANNER}[/bold {BRAND}]\n\n     Memory that reasons",
            expand=False, subtitle=f"Honcho CLI · v{__version__}",
        ))
        _console.print()
        _console.print()

    final_key = _prompt_api_key(key_val)
    final_url = _prompt_url(url_val)

    # Persist if anything changed or if the value came from env/flag.
    if final_key != file_key or final_url != file_url:
        CLIConfig(base_url=final_url, api_key=final_key).save()
        if not use_json():
            _console.print(f"  {ICON_OK} [dim]Saved to {CONFIG_FILE}[/dim]")

    _check_connection(final_url, final_key)

    if use_json():
        print_result({"apiKey": _redact(final_key), "baseUrl": final_url})


def _prompt_api_key(value: str) -> str:
    """Prompt for API key.

    When a key already exists (from env var or config file), the user picks
    between keeping it or entering a replacement.  When no key exists, the
    user can paste one or press Enter to skip (local dev with auth disabled
    doesn't need a key).
    """
    if use_json():
        return value

    if value:
        redacted = _redact(value)
        _console.print(f"  [dim]Current API key: {redacted}[/dim]")
        _console.print("  [dim](1)[/dim] Keep current key")
        _console.print("  [dim](2)[/dim] Enter a new key")
        choice = typer.prompt("  Choice", default="1", show_default=True, prompt_suffix=": ").strip()
        if choice == "2":
            raw = typer.prompt("  API key", default="", show_default=False, prompt_suffix=": ").strip()
            return raw
        return value
    else:
        _console.print("  [dim]Not needed for local dev — press Enter to skip[/dim]")
        raw = typer.prompt("  API key", default="", show_default=False, prompt_suffix=": ").strip()
        return raw


def _normalize_url(url: str) -> str:
    """Strip whitespace from the URL."""
    return url.strip()


def _prompt_url(value: str) -> str:
    """Prompt for Honcho URL. Shows current value as the default; Enter keeps it.

    First run defaults to DEFAULT_BASE_URL. After that, whatever is saved
    in config becomes the default so the user isn't fighting back to their
    custom URL every time.
    """
    if use_json():
        if value:
            return _normalize_url(value)
        print_error("MISSING_VALUE", "Honcho URL is required", {})
        raise typer.Exit(1)

    default = _normalize_url(value) if value else DEFAULT_BASE_URL
    _console.print("  [dim]Use https://api.honcho.dev for the hosted Honcho instance[/dim]")
    while True:
        raw = typer.prompt("  Honcho URL", default=default, show_default=True, prompt_suffix=": ").strip()
        url = _normalize_url(raw)
        if url.startswith(("http://", "https://")):
            return url
        _console.print("  [red]URL must start with http:// or https://[/red]")


def _check_connection(base_url: str, api_key: str) -> None:


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

    # Config file + API connectivity are hard requirements.
    critical = {"Config file", "API key configured", "API connectivity"}
    if config.workspace_id:
        critical.add("Workspace reachable")
    if any(not c["ok"] for c in checks if c["check"] in critical):
        raise typer.Exit(1)
