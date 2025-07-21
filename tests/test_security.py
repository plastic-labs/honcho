import datetime
import unittest.mock

import pytest

from src.exceptions import AuthenticationException
from src.security import JWTParams, create_jwt, verify_jwt


class TestSecurity:
    """Tests for security.py functions."""

    def test_create_jwt_raises_value_error_when_jwt_secret_not_set(self):
        """Test that create_jwt raises ValueError when AUTH_JWT_SECRET is not set."""
        params = JWTParams(t="test_timestamp")

        # Mock settings.AUTH.JWT_SECRET to be None/empty
        with (
            unittest.mock.patch("src.security.settings.AUTH.JWT_SECRET", None),
            pytest.raises(
                ValueError, match="AUTH_JWT_SECRET is not set, cannot create JWT."
            ),
        ):
            create_jwt(params)

        # Also test with empty string
        with (
            unittest.mock.patch("src.security.settings.AUTH.JWT_SECRET", ""),
            pytest.raises(
                ValueError, match="AUTH_JWT_SECRET is not set, cannot create JWT."
            ),
        ):
            create_jwt(params)

    @pytest.mark.asyncio
    async def test_verify_jwt_raises_value_error_when_jwt_secret_not_set(self):
        """Test that verify_jwt raises ValueError when AUTH_JWT_SECRET is not set."""
        token = "dummy_token"

        # Mock settings.AUTH.JWT_SECRET to be None
        with (
            unittest.mock.patch("src.security.settings.AUTH.JWT_SECRET", None),
            pytest.raises(
                ValueError, match="AUTH_JWT_SECRET is not set, cannot verify JWT."
            ),
        ):
            await verify_jwt(token)

        # Also test with empty string
        with (
            unittest.mock.patch("src.security.settings.AUTH.JWT_SECRET", ""),
            pytest.raises(
                ValueError, match="AUTH_JWT_SECRET is not set, cannot verify JWT."
            ),
        ):
            await verify_jwt(token)

    @pytest.mark.asyncio
    async def test_verify_jwt_raises_authentication_exception_when_token_expired(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt raises AuthenticationException when JWT is expired (covers lines 96-101)."""
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")

        # Create a JWT token manually with expired timestamp
        import jwt as jwt_lib

        exp_time = datetime.datetime.now() - datetime.timedelta(hours=1)  # 1 hour ago
        exp_timestamp = int(exp_time.timestamp())

        payload = {"t": "test_timestamp", "exp": exp_timestamp}
        token = jwt_lib.encode(payload, "test-secret", algorithm="HS256")

        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_jwt(token)

    @pytest.mark.asyncio
    async def test_verify_jwt_raises_authentication_exception_when_iso_token_expired(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt raises AuthenticationException when JWT with ISO exp string is expired (covers line 101)."""
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")

        # Create a JWT token using the create_jwt function with expired ISO format timestamp
        exp_time = datetime.datetime.now() - datetime.timedelta(hours=1)  # 1 hour ago
        exp_iso_string = exp_time.isoformat()

        params = JWTParams(t="test_timestamp", exp=exp_iso_string)
        token = create_jwt(params)

        with pytest.raises(AuthenticationException, match="Invalid JWT"):
            await verify_jwt(token)
