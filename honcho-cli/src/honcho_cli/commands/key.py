"""Key management commands: generate scoped JWTs."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer

from honcho_cli.output import print_error, print_result, status

from honcho_cli.common import add_common_options

app = typer.Typer(help="JWT key management.")
add_common_options(app)


def _parse_duration(duration: str) -> datetime:
    """Parse duration string like '30d', '24h', '90d' into expiry datetime."""
    match = re.match(r"^(\d+)([dhm])$", duration)
    if not match:
        print_error(
            "INVALID_DURATION",
            f"Invalid duration format: '{duration}'. Use format like '30d', '24h', '60m'.",
            {"duration": duration},
        )
        raise typer.Exit(1)

    value = int(match.group(1))
    unit = match.group(2)

    delta = {
        "d": timedelta(days=value),
        "h": timedelta(hours=value),
        "m": timedelta(minutes=value),
    }[unit]

    return datetime.now(timezone.utc) + delta


@app.command()
def generate(
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Scope to workspace"),
    peer: Optional[str] = typer.Option(None, "--peer", help="Scope to peer"),
    session: Optional[str] = typer.Option(None, "--session", help="Scope to session"),
    admin: bool = typer.Option(False, "--admin", help="Generate admin key"),
    expires: str = typer.Option("90d", "--expires", help="Expiry duration (e.g., 30d, 24h)"),
    no_expire: bool = typer.Option(False, "--no-expire", help="No expiration (use with caution)"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Generate a scoped JWT key.

    Requires an admin JWT in config. Generated keys are scoped down from admin.
    """
    from honcho_cli.common import handle_cmd_flags
    from honcho_cli.main import get_resolved_config

    handle_cmd_flags(json_output=json_output)

    config = get_resolved_config()

    if not config.api_key:
        print_error("NO_API_KEY", "Admin API key required in config. Run `honcho config set api_key <key>`.")
        raise typer.Exit(3)

    if not admin and not workspace and not peer and not session:
        # Default to current workspace
        workspace = config.workspace_id
        if not workspace:
            print_error(
                "NO_SCOPE",
                "Must specify at least one scope: --workspace, --peer, --session, or --admin",
            )
            raise typer.Exit(1)

    # Build expiry
    expires_at = None
    if not no_expire:
        expires_at = _parse_duration(expires)

    # Use raw HTTP to call the /keys endpoint
    import httpx

    base_url = config.base_url.rstrip("/")
    params: dict = {}
    if workspace:
        params["workspace_id"] = workspace
    if peer:
        params["peer_id"] = peer
    if session:
        params["session_id"] = session
    if expires_at:
        params["expires_at"] = expires_at.isoformat()

    if admin:
        params = {"admin": "true"}
        if expires_at:
            params["expires_at"] = expires_at.isoformat()

    try:
        resp = httpx.post(
            f"{base_url}/v3/keys",
            params=params,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        print_result(data)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            print_error("AUTH_ERROR", "Admin authentication required for key generation", {})
            raise typer.Exit(3)
        print_error("KEY_ERROR", f"Failed to generate key: {e.response.text}", {})
        raise typer.Exit(1)
    except Exception as e:
        print_error("KEY_ERROR", f"Failed to generate key: {e}", {})
        raise typer.Exit(1)
