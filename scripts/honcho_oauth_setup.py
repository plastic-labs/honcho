#!/usr/bin/env python3
"""OAuth setup for Honcho using PKCE authorization code flow (RFC 7636).

The script:
  1. Generates a PKCE challenge locally (no network call needed)
  2. Prints the OpenAI authorization URL for you to open in a browser
  3. Starts a local HTTP server on localhost to catch the redirect
  4. Exchanges the authorization code for an access + refresh token
  5. Prints the .env lines to add, then exits

Usage:
    uv run python scripts/honcho_oauth_setup.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.server
import os
import secrets
import sys
import threading
import urllib.parse

import httpx

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_AUTH_URL = "https://auth.openai.com/authorize"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_SCOPES = "openid profile email offline_access"
_AUDIENCE = "https://api.openai.com/v1"
_REDIRECT_PORT = 54321
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def _build_auth_url(state: str, code_challenge: str) -> str:
    return _AUTH_URL + "?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": _CLIENT_ID,
        "redirect_uri": _REDIRECT_URI,
        "scope": _SCOPES,
        "audience": _AUDIENCE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })


# Shared result written by the HTTP handler, read by the async main loop.
_auth_result: dict[str, str] = {}
_auth_event = threading.Event()


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if "code" in params:
            _auth_result["code"] = params["code"]
            _auth_result["state"] = params.get("state", "")
            html = (
                b"<html><body style='font-family:sans-serif;padding:2rem'>"
                b"<h2>&#x2705; Authentication successful</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            error = params.get("error", "unknown_error")
            desc = params.get("error_description", "")
            _auth_result["error"] = error
            _auth_result["error_description"] = desc
            html = (
                f"<html><body style='font-family:sans-serif;padding:2rem'>"
                f"<h2>&#x274C; Authentication failed</h2>"
                f"<p>{error}: {desc}</p>"
                f"</body></html>"
            ).encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html)
        _auth_event.set()

    def log_message(self, *_args: object) -> None:
        pass  # suppress request logs


async def main() -> None:
    code_verifier, code_challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    auth_url = _build_auth_url(state, code_challenge)

    # Start the local redirect server before printing the URL so the port is
    # ready by the time the browser follows the redirect.
    try:
        server = http.server.HTTPServer(("localhost", _REDIRECT_PORT), _CallbackHandler)
    except OSError as exc:
        print(f"Could not bind to localhost:{_REDIRECT_PORT}: {exc}")
        print("Is another process using that port? Kill it and try again.")
        sys.exit(1)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print()
    print("=" * 60)
    print("  Honcho OAuth Setup")
    print("=" * 60)
    print()
    print("Open this URL in your browser to log in with your ChatGPT")
    print("Plus / OpenAI Codex account:")
    print()
    print(f"  {auth_url}")
    print()
    print("After you log in the browser will redirect to localhost and")
    print("this script will finish automatically.")
    print()
    print("Waiting for browser redirect", end="", flush=True)

    # Poll until the callback handler fires, printing dots to show activity.
    while not _auth_event.wait(timeout=1.0):
        print(".", end="", flush=True)

    server.shutdown()
    print()

    if "error" in _auth_result:
        print()
        print(f"Authentication failed: {_auth_result['error']}")
        desc = _auth_result.get("error_description")
        if desc:
            print(f"  {desc}")
        sys.exit(1)

    if _auth_result.get("state") != state:
        print()
        print("State mismatch — possible CSRF. Aborting.")
        sys.exit(1)

    auth_code = _auth_result["code"]
    print("Code received. Exchanging for tokens...", end="", flush=True)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code": auth_code,
                "redirect_uri": _REDIRECT_URI,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )

    if resp.status_code != 200:
        print()
        print(f"Token exchange failed ({resp.status_code}):")
        print(resp.text[:500])
        sys.exit(1)

    data = resp.json()
    refresh_token: str = data.get("refresh_token", "")

    if not refresh_token:
        print()
        print("Token exchange succeeded but no refresh_token was returned.")
        print("The 'offline_access' scope may not be enabled for this client.")
        print(f"Response keys: {list(data.keys())}")
        sys.exit(1)

    print(" done.")
    print()
    print("=" * 60)
    print("  Authentication successful!")
    print("=" * 60)
    print()
    print("Add these lines to your .env file, then rebuild/restart Honcho:")
    print()
    print("  LLM_OPENAI_AUTH_MODE=oauth")
    print(f"  LLM_OPENAI_REFRESH_TOKEN={refresh_token}")
    print()
    print("You can remove LLM_OPENAI_API_KEY once you've confirmed OAuth works.")


if __name__ == "__main__":
    asyncio.run(main())
