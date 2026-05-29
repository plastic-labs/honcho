"""Auto-refresh module for Nous OAuth credentials.

This module handles automatic detection and renewal of expired Nous API
agent keys. It is designed to be called from within the Honcho async
runtime when an AuthenticationError (401) is encountered.

The flow:
  1. Load refresh_token from persistent state file.
  2. Exchange refresh_token for a new access_token.
  3. Mint a fresh agent_key (TTL=7200 seconds, max allowed).
  4. Update the .env file and in-memory settings.LLM.NOUS_API_KEY.
  5. Return the new agent key so the caller can retry the request.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from src.exceptions import NousAuthError

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

PORTAL_URL = "https://portal.nousresearch.com"
TOKEN_ENDPOINT = f"{PORTAL_URL}/api/oauth/token"
AGENT_KEY_ENDPOINT = f"{PORTAL_URL}/api/oauth/agent-key"
CLIENT_ID = "hermes-cli"
SCOPE = "inference:mint_agent_key"
MIN_TTL_SECONDS = 7200  # Nous maximum allowed TTL (2 hours)

# Default state file location — override with NOUS_OAUTH_STATE_PATH
STATE_FILE = Path(
    os.getenv("NOUS_OAUTH_STATE_PATH", "~/.honcho/nous_oauth_state.json")
).expanduser()


# ── State management ────────────────────────────────────────────────────────


def load_state() -> dict[str, Any]:
    """Load persisted OAuth state from disk, with Hermes auth.json and .env fallbacks."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse %s: %s", STATE_FILE, exc)
    # State file missing or corrupt — try env var first (Docker env_file)
    env_token = os.getenv("NOUS_REFRESH_TOKEN")
    if env_token:
        logger.info("Bootstrapping refresh_token from environment variable")
        return {"refresh_token": env_token}
    # Fallback: Hermes auth.json
    auth_state = _load_from_hermes_auth()
    if auth_state:
        logger.info("Bootstrapping refresh_token from Hermes auth.json")
        return auth_state
    # Finally, check .env on disk (local development without Docker)
    env_refresh = _read_refresh_from_env()
    if env_refresh:
        logger.info("Bootstrapping refresh_token from .env")
        return {"refresh_token": env_refresh}
    return {}


def save_state(**fields: Any) -> None:
    """Merge and persist OAuth state atomically."""
    state = load_state()
    state.update({k: v for k, v in fields.items() if v is not None})
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(STATE_FILE.parent, 0o700)
    # Atomic write via temp file + rename
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(STATE_FILE)
    os.chmod(STATE_FILE, 0o600)


# ── Environment file update ─────────────────────────────────────────────────


def _load_from_hermes_auth() -> dict[str, Any] | None:
    """Bootstrap refresh_token from Hermes auth.json if state file is empty."""
    auth_path = Path.home() / ".hermes" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        auth = json.loads(auth_path.read_text())
        provider = auth.get("providers", {}).get("nous", {})
        refresh = provider.get("refresh_token")
        if refresh:
            return {"refresh_token": refresh}
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read Hermes auth.json: %s", exc)
    return None


def _read_refresh_from_env() -> str | None:
    """Read NOUS_REFRESH_TOKEN from project .env (backward compatibility)."""
    env_path = _find_project_root() / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("NOUS_REFRESH_TOKEN="):
                val = stripped.split("=", 1)[1].strip()
                # Strip surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                return val if val else None
    except Exception as exc:
        logger.debug("Failed to read .env for refresh token: %s", exc)
    return None


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up from start (or __file__) until we find .env or docker-compose.yml."""
    cur = (start or Path(__file__)).resolve()
    for p in [cur, *cur.parents]:
        if (p / ".env").exists() or (p / "docker-compose.yml").exists():
            return p
    # Fallback: cwd
    return Path.cwd()


def update_env_key(env_path: Path, new_key: str) -> None:
    """Update LLM_NOUS_API_KEY in the given .env file, creating it if missing."""
    if not env_path.exists():
        env_path.write_text(f"LLM_NOUS_API_KEY={new_key}\n")
        logger.info("Created .env with new Nous API key at %s", env_path)
        return

    lines = env_path.read_text().splitlines(keepends=True)
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("LLM_NOUS_API_KEY="):
            # Preserve any surrounding quotes/whitespace
            lines[i] = f"LLM_NOUS_API_KEY={new_key}\n"
            updated = True
            break
    if not updated:
        lines.append(f"\nLLM_NOUS_API_KEY={new_key}\n")
    env_path.write_text("".join(lines))
    logger.info("Updated .env with new Nous API key")


# ── HTTP helpers (httpx) ─────────────────────────────────────────────────────


async def refresh_access_token(refresh_token: str) -> tuple[str, str]:
    """Exchange refresh_token for a new access_token and refresh_token."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise NousAuthError(f"Token refresh failed {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["access_token"], data.get("refresh_token", refresh_token)


async def mint_agent_key(access_token: str) -> str:
    """Mint a new agent key using the access token."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            AGENT_KEY_ENDPOINT,
            json={"min_ttl_seconds": MIN_TTL_SECONDS},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
    if resp.status_code != 200:
        raise NousAuthError(f"Agent key mint failed {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["api_key"]


# ── Public orchestrator ──────────────────────────────────────────────────────


async def refresh_nous_credentials() -> str | None:
    """Full refresh+mint flow; returns new agent_key or None on failure."""
    state = load_state()
    refresh_token = state.get("refresh_token")
    if not refresh_token:
        logger.error("No refresh_token found in state — manual login required")
        return None

    try:
        # 1. Refresh access token
        logger.info("Refreshing Nous access token...")
        access_token, new_refresh_token = await refresh_access_token(refresh_token)

        # 2. Mint new agent key
        logger.info("Minting new Nous agent key (TTL=%ds)...", MIN_TTL_SECONDS)
        agent_key = await mint_agent_key(access_token)

        # 3. Compute expiry timestamp (UTC ISO 8601)
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=MIN_TTL_SECONDS)
        ).isoformat()

        # 4. Persist state
        save_state(
            refresh_token=new_refresh_token,
            access_token=access_token,
            agent_key=agent_key,
            expires_at=expires_at,
        )

        # 5. Update .env on disk
        project_root = _find_project_root()
        update_env_key(project_root / ".env", agent_key)

        # 6. Update in-memory settings globally (if Honcho is running)
        try:
            from src.config import settings

            settings.LLM.NOUS_API_KEY = agent_key
            logger.info("In-memory settings.LLM.NOUS_API_KEY updated")
        except (ImportError, AttributeError) as exc:
            # settings may not be importable in all contexts (tests, CLI)
            logger.debug("Could not import settings for in-memory update: %s", exc)

        logger.info("Nous OAuth refresh complete — new key expires at %s", expires_at)
        return agent_key

    except Exception as exc:
        logger.error("Nous credential refresh failed: %s", exc, exc_info=True)
        return None
