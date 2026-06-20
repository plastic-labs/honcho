#!/usr/bin/env python3
"""OAuth setup for Honcho using the OpenAI ChatGPT Plus device code flow.

Opens a browser-based login at auth.openai.com/codex/device, polls for
authorization, exchanges the code for tokens, and prints the .env snippet
to add.

Requirements: Python 3.11+ with httpx (already a Honcho dependency).

Usage:
    uv run python scripts/honcho_oauth_setup.py
"""

from __future__ import annotations

import sys
import time
import webbrowser

import httpx

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_DEVICE_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
_POLL_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
_AUTH_PAGE = "https://auth.openai.com/codex/device"
_MAX_POLL_SECONDS = 900  # 15 minutes


def _request_device_code(client: httpx.Client) -> tuple[str, str, int]:
    """Return (device_auth_id, user_code, poll_interval_seconds)."""
    resp = client.post(_DEVICE_CODE_URL, json={"client_id": _CLIENT_ID})
    resp.raise_for_status()
    data = resp.json()
    return (
        data["device_auth_id"],
        data["user_code"],
        max(int(data.get("interval", 5)), 3),
    )


def _poll_for_auth_code(
    client: httpx.Client,
    device_auth_id: str,
    user_code: str,
    interval: int,
) -> tuple[str, str]:
    """Poll until the user completes login.

    Returns (authorization_code, code_verifier) provided by the server.
    """
    deadline = time.time() + _MAX_POLL_SECONDS
    dots = 0
    while time.time() < deadline:
        time.sleep(interval)
        resp = client.post(
            _POLL_URL,
            json={"device_auth_id": device_auth_id, "user_code": user_code},
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["authorization_code"], data["code_verifier"]
        # 403/404 means still pending — keep waiting
        dots = (dots + 1) % 4
        print(f"\rWaiting for browser login{'.' * dots}   ", end="", flush=True)
    raise TimeoutError(
        f"No login detected after {_MAX_POLL_SECONDS // 60} minutes. "
        "Please run the script again."
    )


def _exchange_code(
    client: httpx.Client,
    authorization_code: str,
    code_verifier: str,
) -> str:
    """Exchange authorization_code for a refresh token."""
    resp = client.post(
        _TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": _REDIRECT_URI,
            "client_id": _CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    data = resp.json()
    refresh_token: str = data["refresh_token"]
    return refresh_token


def main() -> None:
    print()
    print("=" * 60)
    print("  Honcho OAuth Setup (ChatGPT Plus / Codex)")
    print("=" * 60)
    print()

    with httpx.Client(timeout=30.0) as client:
        # Step 1: request a device code
        try:
            device_auth_id, user_code, interval = _request_device_code(client)
        except httpx.HTTPStatusError as exc:
            print(f"Failed to request device code: {exc}")
            sys.exit(1)

        # Step 2: direct the user to the login page
        print(f"Open this URL in your browser to log in with your ChatGPT Plus account:")
        print()
        print(f"  {_AUTH_PAGE}")
        print()
        print(f"When prompted, enter this code:  {user_code}")
        print()

        opened = webbrowser.open(_AUTH_PAGE)
        if opened:
            print("(Browser opened automatically.)")
        print()

        # Step 3: poll for completion
        try:
            authorization_code, code_verifier = _poll_for_auth_code(
                client, device_auth_id, user_code, interval
            )
        except (TimeoutError, httpx.HTTPStatusError, KeyError) as exc:
            print(f"\nError waiting for login: {exc}")
            sys.exit(1)

        print("\r" + " " * 50 + "\r", end="")  # clear waiting line
        print("Login detected! Exchanging code for tokens...")
        print()

        # Step 4: exchange for refresh token
        try:
            refresh_token = _exchange_code(client, authorization_code, code_verifier)
        except httpx.HTTPStatusError as exc:
            print(f"Token exchange failed: {exc}\n{exc.response.text[:500]}")
            sys.exit(1)

    print("=" * 60)
    print("  Authentication successful!")
    print("=" * 60)
    print()
    print("Add these lines to your .env file, then rebuild/restart Honcho:")
    print()
    print("  LLM_OPENAI_AUTH_MODE=oauth")
    print(f"  LLM_OPENAI_REFRESH_TOKEN={refresh_token}")
    print()
    print("The OAuth client targets https://chatgpt.com/backend-api/codex")
    print("(OpenAI Responses API, included with ChatGPT Plus — no API credits needed).")
    print()
    print("NOTE: Model configs with an explicit OVERRIDES__BASE_URL still use")
    print("that provider directly and need their own API key. To route a feature")
    print("through ChatGPT Plus, remove its OVERRIDES__BASE_URL override and set")
    print("its MODEL to a supported model (e.g. gpt-5.4-mini, gpt-4.1-mini).")


if __name__ == "__main__":
    main()
