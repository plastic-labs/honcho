#!/usr/bin/env python3
"""Interactive OAuth setup for Honcho using the OpenAI device code flow (RFC 8628).

Run this script once to authenticate with your ChatGPT / OpenAI Codex account
and obtain a refresh token that Honcho can use to make LLM calls without a
separate API key.

Usage:
    python scripts/honcho_oauth_setup.py

After authenticating, add the printed values to your .env file and restart Honcho.
"""

import asyncio
import sys

import httpx

_DEVICE_CODE_URL = "https://auth.openai.com/oauth/device/code"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_SCOPES = "openid profile email offline_access"
_AUDIENCE = "https://api.openai.com/v1"


async def main() -> None:
    async with httpx.AsyncClient() as client:
        # Step 1: Request device and user codes.
        resp = await client.post(
            _DEVICE_CODE_URL,
            data={
                "client_id": _CLIENT_ID,
                "scope": _SCOPES,
                "audience": _AUDIENCE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            print(f"Error requesting device code: {resp.status_code} {resp.text}")
            sys.exit(1)

        device_data = resp.json()
        device_code: str = device_data["device_code"]
        user_code: str = device_data["user_code"]
        verification_uri: str = device_data["verification_uri"]
        interval: int = device_data.get("interval", 5)
        expires_in: int = device_data.get("expires_in", 300)

        print()
        print("=" * 60)
        print("  Honcho OAuth Setup")
        print("=" * 60)
        print()
        print(f"  1. Open this URL in your browser:")
        print(f"     {verification_uri}")
        print()
        print(f"  2. Enter this code when prompted:")
        print(f"     {user_code}")
        print()
        print(f"  (Code expires in {expires_in // 60} minutes)")
        print()
        print("Waiting for authentication", end="", flush=True)

        # Step 2: Poll the token endpoint until the user completes auth.
        elapsed = 0
        while elapsed < expires_in:
            await asyncio.sleep(interval)
            elapsed += interval

            token_resp = await client.post(
                _TOKEN_URL,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": _CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
            token_data = token_resp.json()

            if token_resp.status_code == 200:
                refresh_token: str = token_data.get("refresh_token", "")
                if not refresh_token:
                    print()
                    print()
                    print("Authentication succeeded but no refresh token was returned.")
                    print("Make sure the 'offline_access' scope is enabled for this client.")
                    sys.exit(1)

                print()
                print()
                print("=" * 60)
                print("  Authentication successful!")
                print("=" * 60)
                print()
                print("Add the following lines to your .env file, then restart Honcho:")
                print()
                print("  LLM_OPENAI_AUTH_MODE=oauth")
                print(f"  LLM_OPENAI_REFRESH_TOKEN={refresh_token}")
                print()
                print("You can also unset LLM_OPENAI_API_KEY if you no longer need it.")
                return

            error = token_data.get("error", "unknown_error")
            if error == "authorization_pending":
                print(".", end="", flush=True)
            elif error == "slow_down":
                interval += 5
                print(".", end="", flush=True)
            elif error == "expired_token":
                print()
                print()
                print("The device code expired. Please run this script again.")
                sys.exit(1)
            else:
                print()
                print()
                description = token_data.get("error_description", "")
                print(f"Authentication error: {error}")
                if description:
                    print(f"  {description}")
                sys.exit(1)

        print()
        print()
        print("Timed out waiting for authentication. Please run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
