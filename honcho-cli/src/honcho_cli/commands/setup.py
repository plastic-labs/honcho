"""Top-level onboarding and health-check commands.

`honcho init`    — interactive onboarding (human) or flags (agent/CI)
`honcho doctor`  — verify connectivity, config validity, queue health
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from honcho_cli import __version__
from honcho_cli.config import CLIConfig, CONFIG_FILE
from honcho_cli.main import BANNER
from honcho_cli.output import print_error, print_result, status

_console = Console(stderr=True)


def _test_connection(base_url: str, api_key: str) -> tuple[bool, str]:
    """Test connectivity to Honcho API by listing workspaces via SDK."""
    try:
        from honcho import Honcho

        client = Honcho(base_url=base_url, api_key=api_key)
        # List workspaces as a connectivity + auth check
        list(client.workspaces())
        return True, "OK"
    except Exception as e:
        msg = str(e)
        if "ConnectError" in msg or "Connection refused" in msg:
            return False, "Connection refused — is the server running?"
        if "timed out" in msg.lower() or "Timeout" in msg:
            return False, "Request timed out"
        if "401" in msg or "Unauthorized" in msg:
            return False, "Unauthorized — check your API key"
        return False, msg


def _list_workspaces(base_url: str, api_key: str) -> list[str]:
    """Fetch workspace IDs from the API."""
    from honcho import Honcho

    client = Honcho(base_url=base_url, api_key=api_key)
    return list(client.workspaces())


def _list_peers(base_url: str, api_key: str, workspace_id: str) -> list:
    """Fetch peers from a workspace."""
    from honcho import Honcho

    client = Honcho(base_url=base_url, api_key=api_key, workspace_id=workspace_id)
    page = client.peers()
    return list(page)


def init(
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="HONCHO_API_KEY", help="API key (admin JWT)"),
    base_url: str = typer.Option("https://api.honcho.dev", "--base-url", envvar="HONCHO_BASE_URL", help="Honcho API base URL"),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace ID to use"),
    peer: Optional[str] = typer.Option(None, "--peer", help="Default peer ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive mode (requires --api-key and --workspace)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Configure the CLI to talk to a Honcho deployment.

    Interactive wizard (human):
        honcho init

    Agent/CI-safe:
        honcho init --api-key $KEY --workspace my-app --yes
    """
    from honcho_cli.output import set_json_mode, use_json

    if json_output:
        set_json_mode(True)

    # --- Non-interactive path ---
    if yes:
        # Fall back to existing config for any values not explicitly provided
        existing = CLIConfig.load()
        resolved_url = base_url
        resolved_key = api_key or existing.api_key
        resolved_ws = workspace or existing.workspace_id
        resolved_peer = peer or existing.peer_id

        if not resolved_key:
            print_error("MISSING_FLAG", "--api-key is required (not found in config or env)", {})
            raise typer.Exit(1)
        if not resolved_ws:
            print_error("MISSING_FLAG", "--workspace is required (not found in config or env)", {})
            raise typer.Exit(1)

        # Test connection
        ok, detail = _test_connection(resolved_url, resolved_key)
        if not ok:
            print_error("CONNECTION_FAILED", f"Cannot reach {resolved_url}: {detail}", {"base_url": resolved_url})
            raise typer.Exit(1)

        config = CLIConfig(
            base_url=resolved_url,
            api_key=resolved_key,
            workspace_id=resolved_ws,
            peer_id=resolved_peer,
        )
        config.save()
        print_result(config.redacted())
        return

    # --- Interactive wizard ---
    banner_content = f"[bold #B6DAFD]{BANNER}[/bold #B6DAFD]\n\n     The Memory Layer for AI Agents"
    _console.print()
    _console.print(Panel(banner_content, expand=False, subtitle=f"Python SDK · v{__version__}"))
    _console.print()

    _console.print("[bold]Welcome! Let's set up your Honcho CLI.[/bold]\n")

    # Step 1: Base URL
    base_url = typer.prompt("  Base URL", default=base_url)

    # Step 2: API key
    if not api_key:
        api_key = typer.prompt("  API key")
    if not api_key:
        print_error("MISSING_VALUE", "API key is required", {})
        raise typer.Exit(1)

    # Step 3: Test connection
    _console.print("\n  [dim]Testing connection...[/dim]", end=" ")
    ok, detail = _test_connection(base_url, api_key)
    if ok:
        _console.print(f"[green]Connected[/green] ({detail})")
    else:
        _console.print(f"[red]Failed[/red]: {detail}")
        if not typer.confirm("  Continue anyway?", default=False):
            raise typer.Exit(1)

    # Step 4: Workspace selection
    workspace_id = workspace
    if not workspace_id:
        try:
            workspaces = _list_workspaces(base_url, api_key)
        except Exception:
            workspaces = []

        if workspaces:
            _console.print(f"\n  [bold]Available workspaces[/bold] ({len(workspaces)}):")
            for i, ws in enumerate(workspaces[:20], 1):
                _console.print(f"    {i}. {ws}")
            _console.print()

            choice = typer.prompt(
                "  Enter workspace ID or number from list",
                default=str(workspaces[0]) if len(workspaces) == 1 else "",
            )

            # Allow picking by number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(workspaces):
                    workspace_id = str(workspaces[idx])
                else:
                    workspace_id = choice
            except ValueError:
                workspace_id = choice
        else:
            workspace_id = typer.prompt("  Workspace ID")

    # Step 5: Peer selection
    peer_id = peer
    if not peer_id:
        try:
            peers = _list_peers(base_url, api_key, workspace_id)
        except Exception:
            peers = []

        if peers:
            _console.print(f"\n  [bold]Peers in workspace[/bold] ({len(peers)}):")
            _console.print("  [dim]Select the peer whose memory and conclusions you want to query by default.[/dim]")
            for i, p in enumerate(peers[:20], 1):
                _console.print(f"    {i}. {p.id}")
            _console.print()

            choice = typer.prompt(
                "  Default peer ID or number (leave blank to skip)",
                default="",
            )

            if choice:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(peers):
                        peer_id = peers[idx].id
                    else:
                        peer_id = choice
                except ValueError:
                    peer_id = choice
        else:
            peer_id = typer.prompt("  Default peer ID (leave blank to skip)", default="")

    # Step 6: Confirm and write
    config = CLIConfig(
        base_url=base_url,
        api_key=api_key,
        workspace_id=workspace_id,
        peer_id=peer_id or "",
    )

    _console.print("\n  [bold]Configuration:[/bold]")
    for k, v in config.redacted().items():
        if v:
            _console.print(f"    {k}: {v}")

    _console.print(f"\n    Config file: {CONFIG_FILE}")
    if not typer.confirm("\n  Save this configuration?", default=True):
        _console.print("  [dim]Aborted.[/dim]")
        raise typer.Exit(0)

    config.save()
    _console.print(f"\n  [green]Config saved to {CONFIG_FILE}[/green]")
    _console.print("\n  Get started:")
    _console.print("    honcho doctor                      Verify your setup")
    _console.print("    honcho workspace list              List workspaces")
    _console.print("    honcho conclusion list             Browse stored conclusions")
    _console.print("    honcho peer list                   List peers in workspace")
    _console.print()


def doctor(
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Verify connectivity, config validity, and queue health.

    Checks:
      1. Config file exists and parses
      2. API connectivity
      3. Workspace is reachable
      4. Default peer exists
      5. Queue health
    """
    from honcho_cli.output import set_json_mode, use_json

    if json_output:
        set_json_mode(True)

    checks: list[dict] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": passed, "detail": detail})
        if not use_json():
            icon = "[green]pass[/green]" if passed else "[red]FAIL[/red]"
            msg = f"  {icon}  {name}"
            if detail:
                msg += f"  [dim]({detail})[/dim]"
            _console.print(msg)

    if not use_json():
        _console.print("\n[bold]Honcho Doctor[/bold]\n")

    # 1. Config file
    config = CLIConfig.load()
    config_exists = CONFIG_FILE.exists()
    _check("Config file", config_exists, str(CONFIG_FILE) if config_exists else f"{CONFIG_FILE} not found")

    # 2. API key present
    has_key = bool(config.api_key)
    _check("API key configured", has_key, "set" if has_key else "missing — run `honcho init`")

    # 3. Connectivity
    if config.base_url and config.api_key:
        ok, detail = _test_connection(config.base_url, config.api_key)
        _check("API connectivity", ok, detail)
    else:
        _check("API connectivity", False, "skipped — no base_url or api_key")

    # 4. Workspace
    ws_ok = False
    if config.workspace_id and config.api_key:
        try:
            from honcho import Honcho

            client = Honcho(
                base_url=config.base_url,
                api_key=config.api_key,
                workspace_id=config.workspace_id,
            )
            # Try to access the workspace config to verify it exists
            client.get_configuration()
            ws_ok = True
            _check("Workspace reachable", True, config.workspace_id)
        except Exception as e:
            _check("Workspace reachable", False, f"{config.workspace_id}: {e}")
    else:
        _check("Workspace reachable", False, "no workspace_id configured")

    # 5. Peer
    if config.peer_id and ws_ok:
        try:
            peer = client.peer(config.peer_id)
            # Try to access the peer to verify it exists
            peer.get_card()
            _check("Default peer exists", True, config.peer_id)
        except Exception as e:
            _check("Default peer exists", False, f"{config.peer_id}: {e}")
    elif config.peer_id:
        _check("Default peer exists", False, "skipped — workspace not reachable")
    else:
        _check("Default peer exists", False, "no peer_id configured (optional)")

    # 6. Queue health
    if ws_ok:
        try:
            q = client.queue_status()
            summary = f"{q.completed_work_units}/{q.total_work_units} completed, {q.pending_work_units} pending"
            _check("Queue health", True, summary)
        except Exception:
            _check("Queue health", True, "endpoint not available (non-critical)")
    else:
        _check("Queue health", False, "skipped — workspace not reachable")

    if not use_json():
        passed = sum(1 for c in checks if c["ok"])
        total = len(checks)
        _console.print(f"\n  {passed}/{total} checks passed\n")

    if use_json():
        print_result({"checks": checks, "passed": sum(1 for c in checks if c["ok"]), "total": len(checks)})

    # Exit non-zero if critical checks fail
    critical_failed = any(not c["ok"] for c in checks if c["check"] in ("Config file", "API connectivity", "Workspace reachable"))
    if critical_failed:
        raise typer.Exit(1)
