"""OAuth 2.0 Device Authorization Grant (RFC 8628) client for the CLI.

Transport-only: HTTP calls plus the poll loop, no Typer or config writes, so it
can be unit-tested by mocking httpx.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

DEVICE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

DEFAULT_CLIENT_ID = "honcho-cli"
DEFAULT_SCOPE = "write"

# self-declared requesting surface; tells the consent screen not to offer config
# delivery (a CLI has nowhere to write it)
DEVICE_SOURCE = "honcho-cli"

# extra seconds added to the poll interval on a slow_down response (RFC 8628 §3.5)
SLOW_DOWN_STEP = 5


class OAuthFlowError(Exception):
    """A device-flow request failed. ``error`` is the RFC error code when known."""

    def __init__(self, error: str, description: str | None = None):
        self.error: str = error
        self.description: str | None = description
        super().__init__(description or error)


class AccessDenied(OAuthFlowError):
    """The user denied the authorization request."""


class DeviceCodeExpired(OAuthFlowError):
    """The device code expired before the user approved it."""


class AuthorizationTimeout(OAuthFlowError):
    """Polling ran past the device code's lifetime with no decision."""


@dataclass(frozen=True)
class Endpoints:
    """Resolved authorization-server URLs and client identity."""

    device_auth_url: str
    token_url: str
    client_id: str
    scope: str


@dataclass(frozen=True)
class DeviceCode:
    """RFC 8628 §3.2 device authorization response."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class TokenResponse:
    """An access/refresh token pair minted for a grant."""

    access_token: str
    refresh_token: str
    expires_in: int
    scope: str
    config: dict[str, Any] = field(default_factory=dict)


def resolve_endpoints(base_url: str) -> Endpoints:
    """Derive OAuth endpoints and client identity from the API ``base_url``."""
    host = base_url.rstrip("/")
    return Endpoints(
        device_auth_url=f"{host}/oauth/device_authorization",
        token_url=f"{host}/oauth/token",
        client_id=DEFAULT_CLIENT_ID,
        scope=DEFAULT_SCOPE,
    )


# RFC 8414 authorization-server metadata; presence of the device grant tells us
# whether this host can do browser login at all (managed only, not core)
AUTH_SERVER_METADATA_PATH = "/.well-known/oauth-authorization-server"


def supports_device_login(base_url: str, *, timeout: float = 5.0) -> bool:
    """Whether the host advertises the device grant in its RFC 8414 metadata.

    Fails closed: any connection error, non-200, unparseable body, or missing
    capability returns False, so self-hosted / non-managed instances simply
    don't offer device login.
    """
    host = base_url.rstrip("/")
    try:
        resp = httpx.get(f"{host}{AUTH_SERVER_METADATA_PATH}", timeout=timeout)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    try:
        body = resp.json()
    except ValueError:
        return False
    grants = body.get("grant_types_supported") if isinstance(body, dict) else None
    return isinstance(grants, list) and DEVICE_GRANT_TYPE in grants


def _post(url: str, data: dict[str, str]) -> httpx.Response:
    """POST form data, surfacing transport failures as ``OAuthFlowError``.

    Connection refusals, DNS failures, and timeouts would otherwise escape as
    raw ``httpx.HTTPError`` past callers that only catch ``OAuthFlowError``.
    """
    try:
        return httpx.post(url, data=data)
    except httpx.HTTPError as e:
        raise OAuthFlowError("connection_error", f"could not reach {url}: {e}") from e


def _error_from_response(resp: httpx.Response) -> tuple[str, str | None]:
    """Pull ``(error, error_description)`` out of an OAuth error body."""
    try:
        body = resp.json()
    except ValueError:
        return "invalid_response", resp.text[:200] or None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"]), body.get("error_description")
    return "invalid_response", None


def request_device_code(endpoints: Endpoints) -> DeviceCode:
    """Request a device + user code pair (RFC 8628 §3.1)."""
    resp = _post(
        endpoints.device_auth_url,
        {
            "client_id": endpoints.client_id,
            "scope": endpoints.scope,
            "source": DEVICE_SOURCE,
        },
    )
    if resp.status_code != 200:
        error, desc = _error_from_response(resp)
        raise OAuthFlowError(error, desc)
    try:
        body = resp.json()
        return DeviceCode(
            device_code=body["device_code"],
            user_code=body["user_code"],
            verification_uri=body["verification_uri"],
            verification_uri_complete=body.get(
                "verification_uri_complete", body["verification_uri"]
            ),
            expires_in=int(body["expires_in"]),
            interval=int(body["interval"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise OAuthFlowError(
            "invalid_response", f"malformed device authorization response: {e}"
        ) from e


def _token_from_body(body: dict[str, Any]) -> TokenResponse:
    # refresh_token is optional on the refresh grant (RFC 6749 §5.1); a
    # malformed/missing field is a server fault, surfaced as OAuthFlowError so
    # callers' existing handling catches it instead of a raw KeyError/ValueError
    try:
        return TokenResponse(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", ""),
            expires_in=int(body["expires_in"]),
            scope=body.get("scope", ""),
            config=body.get("config") or {},
        )
    except (KeyError, TypeError, ValueError) as e:
        raise OAuthFlowError("invalid_response", f"malformed token response: {e}") from e


def poll_for_token(
    endpoints: Endpoints,
    device: DeviceCode,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> TokenResponse:
    """Poll the token endpoint until the grant is approved (RFC 8628 §3.4/§3.5).

    Sleeps ``interval`` between polls, bumping it on ``slow_down``. Raises
    ``AccessDenied`` / ``DeviceCodeExpired`` / ``AuthorizationTimeout`` on the
    terminal outcomes. ``sleep`` / ``monotonic`` are injectable for tests.
    """
    interval = device.interval
    deadline = monotonic() + device.expires_in
    while True:
        if monotonic() >= deadline:
            raise AuthorizationTimeout("expired_token", "Timed out waiting for approval")
        sleep(interval)
        resp = _post(
            endpoints.token_url,
            {
                "grant_type": DEVICE_GRANT_TYPE,
                "device_code": device.device_code,
                "client_id": endpoints.client_id,
            },
        )
        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError as e:
                raise OAuthFlowError("invalid_response", "non-JSON token response") from e
            return _token_from_body(body)

        error, desc = _error_from_response(resp)
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += SLOW_DOWN_STEP
            continue
        if error == "access_denied":
            raise AccessDenied(error, desc)
        if error == "expired_token":
            raise DeviceCodeExpired(error, desc)
        raise OAuthFlowError(error, desc)


def refresh_access_token(endpoints: Endpoints, refresh_token: str) -> TokenResponse:
    """Exchange a refresh token for a fresh access/refresh pair.

    The response may rotate the refresh token; the caller must persist the
    returned ``refresh_token`` before reusing it — replaying a superseded one
    revokes the grant.
    """
    resp = _post(
        endpoints.token_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": endpoints.client_id,
        },
    )
    if resp.status_code != 200:
        error, desc = _error_from_response(resp)
        raise OAuthFlowError(error, desc)
    try:
        body = resp.json()
    except ValueError as e:
        raise OAuthFlowError("invalid_response", "non-JSON token response") from e
    return _token_from_body(body)
