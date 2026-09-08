"""Auth scope tests — regression coverage.

Prior to this fix `auth()` walked the route's declared scope first and fell
through to a workspace check, so a `{w, p}` token authorized any peer in `w`.
The contract now is: authorize by the token's narrowest claim, never widen.
"""

from contextlib import asynccontextmanager

import jwt as pyjwt
import pytest
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from src.config import settings
from src.db import tenant_context
from src.exceptions import AuthenticationException, ValidationException
from src.security import JWTParams, auth, create_jwt, require_auth, verify_jwt


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


@asynccontextmanager
async def _fake_tracked_db(*_args: object, **_kwargs: object):
    """Stand-in for tracked_db; the membership query itself is monkeypatched."""
    yield None


def _patch_membership(monkeypatch: pytest.MonkeyPatch, *, is_member: bool):
    async def _is_peer_in_session(*_args: object, **_kwargs: object) -> bool:
        return is_member

    # Names are resolved via lazy imports inside auth(), so patch the source
    # modules rather than the security namespace.
    monkeypatch.setattr("src.dependencies.tracked_db", _fake_tracked_db)
    monkeypatch.setattr("src.crud.session.is_peer_in_session", _is_peer_in_session)


class TestAuthMemberRead:
    """Peer-scoped key gets read-only access to sessions it is a member of."""

    @pytest.mark.asyncio
    async def test_member_peer_allowed_on_read_route(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_membership(monkeypatch, is_member=True)
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        params = await auth(
            credentials=creds,
            workspace_name="ws-a",
            session_name="sess-1",
            allow_member_read=True,
        )
        assert params.p == "alice"

    @pytest.mark.asyncio
    async def test_non_member_peer_denied_on_read_route(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_membership(monkeypatch, is_member=False)
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(
                credentials=creds,
                workspace_name="ws-a",
                session_name="sess-1",
                allow_member_read=True,
            )

    @pytest.mark.asyncio
    async def test_member_peer_denied_on_write_route(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Write routes never set allow_member_read, so membership is irrelevant."""
        _patch_membership(monkeypatch, is_member=True)
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(
                credentials=creds,
                workspace_name="ws-a",
                session_name="sess-1",
                allow_member_read=False,
            )

    @pytest.mark.asyncio
    async def test_member_peer_denied_cross_workspace(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _patch_membership(monkeypatch, is_member=True)
        creds = _bearer(create_jwt(JWTParams(w="ws-a", p="alice")))
        with pytest.raises(AuthenticationException):
            await auth(
                credentials=creds,
                workspace_name="ws-b",
                session_name="sess-1",
                allow_member_read=True,
            )

    @pytest.mark.asyncio
    async def test_session_token_has_no_cross_scope_to_peer_routes(self):
        """A session key never reaches peer routes, even with allow_member_read."""
        creds = _bearer(create_jwt(JWTParams(w="ws-a", s="sess-1")))
        with pytest.raises(AuthenticationException):
            await auth(
                credentials=creds,
                workspace_name="ws-a",
                peer_name="alice",
                allow_member_read=True,
            )


class TestCreateKeyValidation:
    @pytest.mark.asyncio
    async def test_peer_key_without_workspace_rejected(self):
        from src.routers.keys import create_key

        with pytest.raises(ValidationException):
            await create_key(workspace_id=None, peer_id="alice", session_id=None)

    @pytest.mark.asyncio
    async def test_session_key_without_workspace_rejected(self):
        from src.routers.keys import create_key

        with pytest.raises(ValidationException):
            await create_key(workspace_id=None, peer_id=None, session_id="sess-1")

    @pytest.mark.asyncio
    async def test_peer_key_with_workspace_ok(self):
        from src.routers.keys import create_key

        result = await create_key(workspace_id="ws-a", peer_id="alice", session_id=None)
        assert "key" in result


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


class TestAuthTenantClaim:
    """A3/DEV-2478 — the `tn` tenant claim and the MULTI_TENANT tenant gate.

    Under MULTI_TENANT every token must carry a tenant; the gate is enforced before
    the admin short-circuit, so `ad` is admin-within-tenant, never cross-tenant. Flag
    off (single-tenant OSS) the gate is inert.
    """

    def test_verify_jwt_decodes_tn(self):
        params = verify_jwt(create_jwt(JWTParams(tn="tenant-a", w="ws-a")))
        assert params.tn == "tenant-a"

    def test_verify_jwt_normalizes_empty_tn_to_none(self):
        # A blank tn must not masquerade as a present claim (mirrors w/p/s).
        params = verify_jwt(create_jwt(JWTParams(tn="", w="ws-a")))
        assert params.tn is None

    @pytest.mark.asyncio
    async def test_multi_tenant_requires_tn(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))  # no tn
        with pytest.raises(AuthenticationException, match="tenant claim"):
            await auth(credentials=creds, workspace_name="ws-a")

    @pytest.mark.asyncio
    async def test_multi_tenant_admin_still_requires_tn(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # §3: a tenant-less admin token is NOT god-mode under MULTI_TENANT — the
        # gate runs before the admin short-circuit.
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        creds = _bearer(create_jwt(JWTParams(ad=True)))  # admin, no tn
        with pytest.raises(AuthenticationException, match="tenant claim"):
            await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")

    @pytest.mark.asyncio
    async def test_multi_tenant_token_with_tn_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        creds = _bearer(create_jwt(JWTParams(tn="tenant-a", w="ws-a")))
        params = await auth(credentials=creds, workspace_name="ws-a")
        assert params.tn == "tenant-a"

    @pytest.mark.asyncio
    async def test_multi_tenant_admin_with_tn_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Admin-within-tenant: with a tenant present, `ad` still reaches any route
        # (RLS confines it to the bound tenant downstream).
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        creds = _bearer(create_jwt(JWTParams(tn="tenant-a", ad=True)))
        params = await auth(credentials=creds, workspace_name="ws-a", peer_name="alice")
        assert params.ad is True
        assert params.tn == "tenant-a"

    @pytest.mark.asyncio
    async def test_flag_off_tenantless_token_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Single-tenant OSS default: no tn, no gate — byte-for-byte prior behavior.
        monkeypatch.setattr(settings, "MULTI_TENANT", False)
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        params = await auth(credentials=creds, workspace_name="ws-a")
        assert params.w == "ws-a"
        assert params.tn is None


class TestRequireAuthTenantBinding:
    """A3/DEV-2478 §4 — require_auth binds the resolved tenant into tenant_context
    for the duration of the request (a yield-dependency), so A2's checkout hook binds
    `app.tenant`; reset on teardown so nothing leaks to the next request.
    """

    @staticmethod
    def _drive(creds: HTTPAuthorizationCredentials):
        # A real (empty) Request; require_auth() with no scope never reads it anyway.
        request = Request(
            {"type": "http", "query_string": b"", "path_params": {}, "headers": []}
        )
        return require_auth()(request=request, credentials=creds)

    @pytest.mark.asyncio
    async def test_binds_during_request_and_resets_after(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        creds = _bearer(create_jwt(JWTParams(tn="tenant-a", w="ws-a")))
        assert tenant_context.get() is None
        agen = self._drive(creds)
        jwt_params = await agen.__anext__()
        assert jwt_params.tn == "tenant-a"
        assert tenant_context.get() == "tenant-a"  # bound for the request body
        await agen.aclose()  # teardown
        assert tenant_context.get() is None  # reset — no leak

    @pytest.mark.asyncio
    async def test_no_binding_when_token_has_no_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Flag off (or a tenant-less token): nothing is bound.
        monkeypatch.setattr(settings, "MULTI_TENANT", False)
        creds = _bearer(create_jwt(JWTParams(w="ws-a")))
        agen = self._drive(creds)
        await agen.__anext__()
        assert tenant_context.get() is None
        await agen.aclose()
        assert tenant_context.get() is None
