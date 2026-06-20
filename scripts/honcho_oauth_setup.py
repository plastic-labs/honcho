#!/usr/bin/env python3
"""OAuth setup for Honcho using the OpenAI Codex CLI.

The Codex CLI handles the browser-based OAuth flow correctly (including
Cloudflare challenges on auth.openai.com).  This script delegates
authentication to it, then reads the resulting refresh token and prints
the .env snippet to add.

Requirements:
    Install the Codex CLI via snap:     sudo snap install codex
    Or via npm:                         npm install -g @openai/codex

Usage:
    uv run python scripts/honcho_oauth_setup.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Locations where the Codex CLI stores its auth file, in preference order.
# The snap revision directory (e.g. ~/snap/codex/34/) is discovered at runtime.
_AUTH_FILE_CANDIDATES: list[Path] = [
    Path.home() / ".codex" / "auth.json",
]


def _find_snap_auth_file() -> Path | None:
    """Return the auth.json from the active snap revision, if present."""
    snap_base = Path.home() / "snap" / "codex"
    if not snap_base.exists():
        return None
    # Each revision is a numeric subdirectory; pick the highest (newest).
    revisions = sorted(
        (d for d in snap_base.iterdir() if d.name.isdigit()),
        key=lambda d: int(d.name),
        reverse=True,
    )
    for rev in revisions:
        candidate = rev / "auth.json"
        if candidate.exists():
            return candidate
    return None


def _find_auth_file() -> Path | None:
    snap = _find_snap_auth_file()
    if snap:
        return snap
    for candidate in _AUTH_FILE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _read_refresh_token(auth_file: Path) -> str:
    data = json.loads(auth_file.read_text())
    token: str = data.get("tokens", {}).get("refresh_token", "")
    if not token:
        print(f"No refresh_token found in {auth_file}")
        print("File contents:", json.dumps(data, indent=2)[:500])
        sys.exit(1)
    return token


def _codex_is_installed() -> bool:
    return shutil.which("codex") is not None


def _codex_login_status() -> bool:
    """Return True if Codex reports being logged in."""
    result = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _run_codex_device_auth() -> None:
    """Invoke `codex login --device-auth` interactively (inherits stdin/stdout)."""
    print("Running: codex login --device-auth")
    print()
    result = subprocess.run(["codex", "login", "--device-auth"])
    if result.returncode != 0:
        print("\ncodex login failed. Please run it manually and try again.")
        sys.exit(1)


def main() -> None:
    print()
    print("=" * 60)
    print("  Honcho OAuth Setup (via OpenAI Codex CLI)")
    print("=" * 60)
    print()

    # 1. Ensure Codex CLI is available.
    if not _codex_is_installed():
        print("The Codex CLI is not installed.")
        print()
        print("Install it with one of:")
        print("  sudo snap install codex          # Ubuntu / Snap")
        print("  npm install -g @openai/codex     # Node.js")
        print()
        print("Then re-run this script.")
        sys.exit(1)

    # 2. Authenticate if needed.
    auth_file = _find_auth_file()
    if auth_file is None or not _codex_login_status():
        print("You are not logged in to the Codex CLI.")
        print("Starting device authentication — follow the prompts below.")
        print()
        _run_codex_device_auth()
        print()
        auth_file = _find_auth_file()

    if auth_file is None:
        print("Could not locate Codex auth file after login.")
        print("Expected locations:")
        print("  ~/.codex/auth.json")
        print("  ~/snap/codex/<revision>/auth.json")
        sys.exit(1)

    # 3. Read the refresh token.
    refresh_token = _read_refresh_token(auth_file)

    print(f"Found credentials: {auth_file}")
    print()
    print("=" * 60)
    print("  Authentication successful!")
    print("=" * 60)
    print()
    print("Add these lines to your .env file, then rebuild/restart Honcho:")
    print()
    print("  # Remove or comment out LLM_OPENAI_API_KEY and LLM_OPENAI_BASE_URL")
    print("  # (or keep them for OpenRouter fallback — see note below)")
    print("  LLM_OPENAI_AUTH_MODE=oauth")
    print(f"  LLM_OPENAI_REFRESH_TOKEN={refresh_token}")
    print()
    print("NOTE: The OAuth token authenticates against api.openai.com (not")
    print("OpenRouter).  If your model configs use OpenRouter-specific models")
    print("(google/gemini-*, deepseek/*, etc.) via OVERRIDES__BASE_URL, those")
    print("overrides continue to use their own base URL.  They will need an")
    print("explicit API key (set OVERRIDES__API_KEY_ENV to an env var that")
    print("holds your OpenRouter key) if you remove LLM_OPENAI_API_KEY.")
    print()
    print("For a fully OAuth-only setup, switch those models to OpenAI-native")
    print("equivalents (gpt-4.1-mini, o3-mini, etc.) and remove the OpenRouter")
    print("base URL overrides.")


if __name__ == "__main__":
    main()
