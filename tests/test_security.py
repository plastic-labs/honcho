import datetime
import unittest.mock

import pytest

from src.exceptions import AuthenticationException
from src.security import JWTParams, auth, create_jwt, verify_jwt


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

    @pytest.mark.asyncio
    async def test_verify_jwt_sets_exp_field_from_decoded_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt sets params.exp when JWT contains exp field (covers lines 95-96)."""
        import unittest.mock
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")

        # Create a valid JWT token without exp first, to establish baseline
        params_without_exp = JWTParams(t="test_timestamp")
        token_without_exp = create_jwt(params_without_exp)
        result_without_exp = await verify_jwt(token_without_exp)
        
        # Verify baseline works and exp is None by default
        assert result_without_exp.exp is None
        assert result_without_exp.t == "test_timestamp"

        # Now test the specific lines 95-96 by mocking jwt.decode to return what we need
        exp_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        exp_iso_string = exp_time.isoformat()
        
        # Mock decoded response that would trigger lines 95-96
        mock_decoded = {"t": "test_timestamp", "exp": exp_iso_string}
        
        with unittest.mock.patch("src.security.jwt.decode", return_value=mock_decoded):
            # This should execute:
            # Line 95: params.exp = decoded["exp"] 
            # Line 96: if (params.exp and datetime.datetime.fromisoformat(params.exp) < datetime.datetime.now()):
            result_with_exp = await verify_jwt("dummy_token")
            
            # Verify that params.exp was set (line 95 executed)
            assert result_with_exp.exp == exp_iso_string
            # Verify the condition check passed (line 96 executed but condition was false)
            assert result_with_exp.t == "test_timestamp"

    @pytest.mark.asyncio
    async def test_verify_jwt_sets_exp_field_to_none_when_decoded_exp_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt sets params.exp to None when decoded exp is None (covers line 95)."""
        import unittest.mock
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        
        # Mock decoded response with exp set to None
        mock_decoded = {"t": "test_timestamp", "exp": None}
        
        with unittest.mock.patch("src.security.jwt.decode", return_value=mock_decoded):
            # This should execute line 95: params.exp = decoded["exp"] (setting it to None)
            result = await verify_jwt("dummy_token")
            
            # Verify that params.exp was set to None (line 95 executed)
            assert result.exp is None
            assert result.t == "test_timestamp"

    @pytest.mark.asyncio
    async def test_verify_jwt_exp_condition_with_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt handles empty string exp field correctly (covers lines 95-96)."""
        import unittest.mock
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        
        # Mock decoded response with exp set to empty string
        mock_decoded = {"t": "test_timestamp", "exp": ""}
        
        with unittest.mock.patch("src.security.jwt.decode", return_value=mock_decoded):
            # This should execute:
            # Line 95: params.exp = decoded["exp"] (setting it to "")
            # Line 96: if (params.exp ...) - the condition should be False due to empty string being falsy
            result = await verify_jwt("dummy_token")
            
            # Verify that params.exp was set to empty string (line 95 executed)
            assert result.exp == ""
            # Verify that the empty string didn't trigger expiration logic (line 96 condition was falsy)
            assert result.t == "test_timestamp"

    @pytest.mark.asyncio
    async def test_verify_jwt_raises_jwt_expired_exception_for_expired_iso_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that verify_jwt raises AuthenticationException with 'JWT expired' message for expired ISO token (covers line 101)."""
        import unittest.mock
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        
        # Create an expired ISO timestamp 
        exp_time = datetime.datetime.now() - datetime.timedelta(hours=1)  # 1 hour ago
        exp_iso_string = exp_time.isoformat()
        
        # Mock decoded response with expired ISO string
        mock_decoded = {"t": "test_timestamp", "exp": exp_iso_string}
        
        with unittest.mock.patch("src.security.jwt.decode", return_value=mock_decoded):
            # This should execute:
            # Line 95: params.exp = decoded["exp"] 
            # Line 96-100: if condition evaluates to True (exp is truthy and parsed datetime < now)
            # Line 101: raise AuthenticationException("JWT expired")
            with pytest.raises(AuthenticationException, match="JWT expired"):
                await verify_jwt("dummy_token")

    @pytest.mark.asyncio
    async def test_auth_session_level_access_with_workspace_mismatch_raises_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that auth raises AuthenticationException when session JWT has mismatched workspace (covers lines 182-184)."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        monkeypatch.setattr("src.security.settings.AUTH.USE_AUTH", True)
        
        # Create JWT params with session and workspace scoped to different workspaces
        jwt_params_with_mismatched_workspace = JWTParams(
            t="test_timestamp", 
            s="session123",  # JWT is scoped to this session
            w="workspace_a"  # JWT is scoped to workspace_a
        )
        
        # Create and mock the JWT token
        token = create_jwt(jwt_params_with_mismatched_workspace)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Test case: session_name matches JWT session, but workspace_name doesn't match JWT workspace
        # This should trigger lines 181 (condition true), 182 (workspace check), 183 (raise exception)
        with pytest.raises(AuthenticationException, match="JWT not permissioned for this resource"):
            await auth(
                credentials=credentials,
                session_name="session123",  # Matches jwt_params.s
                workspace_name="workspace_b"  # Does NOT match jwt_params.w (workspace_a)
            )

    @pytest.mark.asyncio
    async def test_auth_session_level_access_with_matching_workspace_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that auth returns jwt_params when session JWT has matching workspace (covers line 184)."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        monkeypatch.setattr("src.security.settings.AUTH.USE_AUTH", True)
        
        # Create JWT params with session and workspace that match
        jwt_params_matching = JWTParams(
            t="test_timestamp", 
            s="session123",  # JWT is scoped to this session
            w="workspace_a"  # JWT is scoped to workspace_a
        )
        
        # Create and mock the JWT token
        token = create_jwt(jwt_params_matching)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Test case: session_name matches JWT session AND workspace_name matches JWT workspace
        # This should trigger lines 181 (condition true), 182 (workspace check passes), 184 (return jwt_params)
        result = await auth(
            credentials=credentials,
            session_name="session123",  # Matches jwt_params.s
            workspace_name="workspace_a"  # Matches jwt_params.w
        )
        
        # Verify the jwt_params are returned correctly
        assert result.s == "session123"
        assert result.w == "workspace_a"
        assert result.t == "test_timestamp"

    @pytest.mark.asyncio
    async def test_auth_peer_level_access_with_workspace_mismatch_raises_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that auth raises AuthenticationException when peer JWT has mismatched workspace (covers line 189)."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Set up the JWT secret
        monkeypatch.setattr("src.security.settings.AUTH.JWT_SECRET", "test-secret")
        monkeypatch.setattr("src.security.settings.AUTH.USE_AUTH", True)
        
        # Create JWT params with peer and workspace scoped to different workspaces
        jwt_params_with_mismatched_workspace = JWTParams(
            t="test_timestamp", 
            p="peer123",  # JWT is scoped to this peer
            w="workspace_a"  # JWT is scoped to workspace_a
        )
        
        # Create and mock the JWT token
        token = create_jwt(jwt_params_with_mismatched_workspace)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Test case: peer_name matches JWT peer, but workspace_name doesn't match JWT workspace
        # This should trigger lines 187 (condition true), 188 (workspace check), 189 (raise exception)
        with pytest.raises(AuthenticationException, match="JWT not permissioned for this resource"):
            await auth(
                credentials=credentials,
                peer_name="peer123",  # Matches jwt_params.p
                workspace_name="workspace_b"  # Does NOT match jwt_params.w (workspace_a)
            )
