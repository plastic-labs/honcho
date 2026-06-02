"""Auth scope tests — DEV-1736 regression coverage.

Prior to this fix `auth()` walked the route's declared scope first and fell
through to a workspace check, so a `{w, p}` token authorized any peer in `w`.
The contract now is: authorize by the token's narrowest claim, never widen.
"""

import jwt as pyjwt
import pytest
from fastapi.security import HTTPAuthorizationCredentials

from src.config import settings
from src.exceptions import AuthenticationException
from src.security import JWTParams, auth, create_jwt, verify_jwt


@pytest.fixture(autouse=True)
def _enable_auth(monkeypatch: pytest.MonkeyPatch):  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class TestVerifyJWTShape:
    def test_peer_token_without_workspace_rejected(self):
        token = pyjwt.encode({"p": "alice"}, b"test-secret", algorithm="HS256")
        with pytest.raises(AuthenticationException):
            verify_jwt(token)

    def test_session_token_without_workspace_rejected(self):
        token = pyjwt.encode({"s": "sess-1"}, b"test-secret", algorithm="HS256")
        with pytest.raises(AuthenticationException):
            verify_jwt(token)

    def test_workspace_only_token_ok(self):
        token = create_jwt(JWTParams(w="ws-a"))
        params = verify_jwt(token)
        assert params.w == "ws-a"

    def test_workspace_peer_token_ok(self):
        token = create_jwt(JWTParams(w="ws-a", p="alice"))
        params = verify_jwt(token)
        assert params.w == "ws-a"
        assert params.p == "alice"


class TestAuthPeerScope:
    """`{w: ws-a, p: alice}` may only act on alice in ws-a."""

    @pytest.mark.asyncio
    async def test_matches_own_peer(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        params = await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")
        assert params.p == "alice"

    @pytest.mark.asyncio
    async def test_denies_sibling_peer_same_workspace(self):
        """The original bug: peer-scoped token fell through to workspace auth."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a", peer_name="bob")

    @pytest.mark.asyncio
    async def test_denies_workspace_route_with_no_peer(self):
        """Peer-scoped token cannot use workspace-listing routes."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a")

    @pytest.mark.asyncio
    async def test_self_authorizing_route_receives_claims(self):
        """Body-scoped routes use require_auth() and compare claims in-handler."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        params = await auth(credentials=creds)
        assert params.w == "ws-a"
        assert params.p == "alice"

    @pytest.mark.asyncio
    async def test_denies_cross_workspace(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-b", peer_name="alice")


class TestAuthSessionScope:
    @pytest.mark.asyncio
    async def test_matches_own_session(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", s="sess-1")))
        params = await auth(
            credentials=creds, workspace_name="ws-a", session_name="sess-1"
        )
        assert params.s == "sess-1"

    @pytest.mark.asyncio
    async def test_denies_sibling_session_same_workspace(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", s="sess-1")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a", session_name="sess-2")

    @pytest.mark.asyncio
    async def test_denies_workspace_route_with_no_session(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", s="sess-1")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a")

    @pytest.mark.asyncio
    async def test_self_authorizing_route_receives_claims(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a", s="sess-1")))
        params = await auth(credentials=creds)
        assert params.w == "ws-a"
        assert params.s == "sess-1"


class TestAuthWorkspaceScope:
    @pytest.mark.asyncio
    async def test_matches_workspace(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        params = await auth(credentials=creds, workspace_name="ws-a")
        assert params.w == "ws-a"

    @pytest.mark.asyncio
    async def test_workspace_token_reaches_peer_route(self):
        """Workspace tokens still authorize narrower routes inside the workspace."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        params = await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")
        assert params.w == "ws-a"

    @pytest.mark.asyncio
    async def test_denies_cross_workspace(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-b")

    @pytest.mark.asyncio
    async def test_passes_self_authorizing_route(self):
        """Routes with no declared scope (e.g. POST /v3/workspaces) self-authorize
        on the token's `w`. The auth dependency must let workspace tokens through."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        params = await auth(credentials=creds)
        assert params.w == "ws-a"


class TestAuthAdminAndUnscoped:
    @pytest.mark.asyncio
    async def test_admin_passes_any_route(self):
        creds = _bearer(create_jwt(JWTParams(ad=True)))
        params = await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")
        assert params.ad is True

    @pytest.mark.asyncio
    async def test_non_admin_token_denied_on_admin_route(self):
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, admin=True)

    @pytest.mark.asyncio
    async def test_unscoped_token_on_self_authorizing_route(self):
        """A token with no scope claims and a route with no declared scope is the
        escape hatch for routes that introspect jwt_params themselves."""
        creds = _bearer(create_jwt(JWTParams()))
        params = await auth(credentials=creds)
        assert params.w is None
        assert params.p is None
        assert params.s is None

    @pytest.mark.asyncio
    async def test_unscoped_token_denied_on_scoped_route(self):
        creds = _bearer(create_jwt(JWTParams()))
        with pytest.raises(AuthenticationException):
            await auth(credentials=creds, workspace_name="ws-a")
