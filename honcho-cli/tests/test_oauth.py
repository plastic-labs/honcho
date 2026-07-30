"""Tests for the device-authorization OAuth engine (transport-only)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from honcho_cli import oauth
from honcho_cli.oauth import (
    AccessDenied,
    AuthorizationTimeout,
    DeviceCode,
    DeviceCodeExpired,
    Endpoints,
    OAuthFlowError,
)


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _endpoints() -> Endpoints:
    return Endpoints(
        device_auth_url="https://api.honcho.dev/oauth/device_authorization",
        token_url="https://api.honcho.dev/oauth/token",
        client_id="honcho-cli",
        scope="write",
    )


DEVICE = DeviceCode(
    device_code="dev-abc",
    user_code="WXYZ-1234",
    verification_uri="https://app.honcho.dev/device",
    verification_uri_complete="https://app.honcho.dev/device?user_code=WXYZ-1234",
    expires_in=600,
    interval=5,
)


# --------------------------------------------------------------------------- #
# resolve_endpoints

class TestResolveEndpoints:
    def test_derives_urls_from_base_url(self):
        ep = oauth.resolve_endpoints("https://api.honcho.dev")
        assert ep.device_auth_url == "https://api.honcho.dev/oauth/device_authorization"
        assert ep.token_url == "https://api.honcho.dev/oauth/token"
        assert ep.client_id == "honcho-cli"
        assert ep.scope == "write"

    def test_strips_trailing_slash(self):
        ep = oauth.resolve_endpoints("http://localhost:8000/")
        assert ep.token_url == "http://localhost:8000/oauth/token"


# --------------------------------------------------------------------------- #
# supports_device_login

class TestSupportsDeviceLogin:
    def test_true_when_device_grant_advertised(self):
        body = {"grant_types_supported": ["authorization_code", oauth.DEVICE_GRANT_TYPE]}
        with patch("honcho_cli.oauth.httpx.get", return_value=FakeResponse(200, body)):
            assert oauth.supports_device_login("https://api.honcho.dev") is True

    def test_false_when_device_grant_absent(self):
        body = {"grant_types_supported": ["authorization_code", "refresh_token"]}
        with patch("honcho_cli.oauth.httpx.get", return_value=FakeResponse(200, body)):
            assert oauth.supports_device_login("https://api.honcho.dev") is False

    @pytest.mark.parametrize("status", [404, 500])
    def test_false_on_non_200(self, status):
        with patch("honcho_cli.oauth.httpx.get", return_value=FakeResponse(status, "")):
            assert oauth.supports_device_login("http://localhost:8000") is False

    def test_false_on_connection_error(self):
        with patch("honcho_cli.oauth.httpx.get", side_effect=httpx.ConnectError("no route")):
            assert oauth.supports_device_login("http://localhost:8000") is False

    def test_false_on_unparseable_body(self):
        with patch("honcho_cli.oauth.httpx.get", return_value=FakeResponse(200, ValueError())):
            assert oauth.supports_device_login("https://api.honcho.dev") is False


# --------------------------------------------------------------------------- #
# request_device_code

class TestRequestDeviceCode:
    def test_success(self):
        body = {
            "device_code": "dev-abc",
            "user_code": "WXYZ-1234",
            "verification_uri": "https://app.honcho.dev/device",
            "verification_uri_complete": "https://app.honcho.dev/device?user_code=WXYZ-1234",
            "expires_in": 600,
            "interval": 5,
        }
        with patch("honcho_cli.oauth.httpx.post", return_value=FakeResponse(200, body)) as post:
            dc = oauth.request_device_code(_endpoints())
        assert dc.device_code == "dev-abc"
        assert dc.user_code == "WXYZ-1234"
        assert dc.interval == 5
        assert post.call_args.kwargs["data"]["source"] == "honcho-cli"

    def test_error_raises(self):
        body = {"error": "invalid_client", "error_description": "unknown client"}
        with patch("honcho_cli.oauth.httpx.post", return_value=FakeResponse(401, body)):
            with pytest.raises(OAuthFlowError) as exc:
                oauth.request_device_code(_endpoints())
        assert exc.value.error == "invalid_client"

    def test_transport_failure_wrapped(self):
        with patch("honcho_cli.oauth.httpx.post", side_effect=httpx.ConnectError("no route")):
            with pytest.raises(OAuthFlowError) as exc:
                oauth.request_device_code(_endpoints())
        assert exc.value.error == "connection_error"


# --------------------------------------------------------------------------- #
# poll_for_token

class TestPollForToken:
    def _run(self, responses, monotonic_vals=None):
        """Poll with a scripted response sequence, capturing sleep durations."""
        sleeps: list[float] = []
        clock = iter(monotonic_vals or [0.0] * (len(responses) + 2))
        with patch("honcho_cli.oauth.httpx.post", side_effect=responses):
            token = oauth.poll_for_token(
                _endpoints(),
                DEVICE,
                sleep=sleeps.append,
                monotonic=lambda: next(clock),
            )
        return token, sleeps

    def test_pending_then_slowdown_then_success(self):
        success = {
            "access_token": "hch-at-1",
            "refresh_token": "hch-rt-1",
            "expires_in": 3600,
            "scope": "write",
            "config": {"k": "v"},
        }
        responses = [
            FakeResponse(400, {"error": "authorization_pending"}),
            FakeResponse(400, {"error": "slow_down"}),
            FakeResponse(200, success),
        ]
        token, sleeps = self._run(responses)
        assert token.access_token == "hch-at-1"
        assert token.refresh_token == "hch-rt-1"
        assert token.config == {"k": "v"}
        # interval starts at 5, bumps by 5 after slow_down → third sleep is 10
        assert sleeps == [5, 5, 10]

    def test_access_denied(self):
        responses = [FakeResponse(400, {"error": "access_denied"})]
        with pytest.raises(AccessDenied):
            self._run(responses)

    def test_expired_token(self):
        responses = [FakeResponse(400, {"error": "expired_token"})]
        with pytest.raises(DeviceCodeExpired):
            self._run(responses)

    def test_unexpected_error_raises_generic(self):
        responses = [FakeResponse(400, {"error": "invalid_grant"})]
        with pytest.raises(OAuthFlowError) as exc:
            self._run(responses)
        assert exc.value.error == "invalid_grant"

    def test_transport_failure_wrapped(self):
        # an exception in the side_effect list is raised on that poll
        responses = [httpx.ReadTimeout("timed out")]
        with pytest.raises(OAuthFlowError) as exc:
            self._run(responses)
        assert exc.value.error == "connection_error"

    def test_times_out_past_deadline(self):
        # monotonic jumps past deadline (0 + expires_in) on the first check
        with patch("honcho_cli.oauth.httpx.post") as post:
            with pytest.raises(AuthorizationTimeout):
                oauth.poll_for_token(
                    _endpoints(),
                    DEVICE,
                    sleep=lambda _s: None,
                    monotonic=iter([0.0, 9999.0]).__next__,
                )
        post.assert_not_called()


# --------------------------------------------------------------------------- #
# refresh_access_token

class TestRefresh:
    def test_success_returns_rotated_pair(self):
        body = {
            "access_token": "hch-at-2",
            "refresh_token": "hch-rt-2",
            "expires_in": 3600,
            "scope": "write",
        }
        with patch("honcho_cli.oauth.httpx.post", return_value=FakeResponse(200, body)) as post:
            token = oauth.refresh_access_token(_endpoints(), "hch-rt-1")
        assert token.access_token == "hch-at-2"
        assert token.refresh_token == "hch-rt-2"
        sent = post.call_args.kwargs["data"]
        assert sent["grant_type"] == "refresh_token"
        assert sent["refresh_token"] == "hch-rt-1"

    def test_error_raises(self):
        body = {"error": "invalid_grant", "error_description": "revoked"}
        with patch("honcho_cli.oauth.httpx.post", return_value=FakeResponse(400, body)):
            with pytest.raises(OAuthFlowError) as exc:
                oauth.refresh_access_token(_endpoints(), "stale")
        assert exc.value.error == "invalid_grant"

    def test_transport_failure_wrapped(self):
        with patch("honcho_cli.oauth.httpx.post", side_effect=httpx.ConnectError("no route")):
            with pytest.raises(OAuthFlowError) as exc:
                oauth.refresh_access_token(_endpoints(), "hch-rt-1")
        assert exc.value.error == "connection_error"
