"""Tests for multi-tenant support — Tenant model, JWT scoping, tenant router."""

import pytest
from sqlalchemy import select

from src.models import Tenant, Workspace
from src.security import JWTParams, create_jwt, verify_jwt


class TestTenantModel:
    """Tenant ORM model tests."""

    async def test_create_tenant(self, db_session):
        """Create a tenant with required fields."""
        tenant = Tenant(name="test-tenant")
        db_session.add(tenant)
        await db_session.commit()
        await db_session.refresh(tenant)

        assert tenant.id is not None
        assert len(tenant.id) == 21  # nanoid
        assert tenant.name == "test-tenant"
        assert tenant.admin_jwt_secret is None
        assert tenant.h_metadata == {}
        assert tenant.configuration == {}

    async def test_tenant_unique_name(self, db_session):
        """Tenant names must be unique."""
        t1 = Tenant(name="unique-tenant")
        t2 = Tenant(name="unique-tenant")
        db_session.add(t1)
        await db_session.commit()

        db_session.add(t2)
        with pytest.raises(Exception):  # IntegrityError
            await db_session.commit()

    async def test_tenant_workspace_cascade(self, db_session):
        """Deleting a tenant cascades to workspaces."""
        tenant = Tenant(name="cascade-tenant")
        db_session.add(tenant)
        await db_session.flush()

        ws = Workspace(name="cascade-ws", tenant_id=tenant.id)
        db_session.add(ws)
        await db_session.commit()

        # Delete tenant
        await db_session.delete(tenant)
        await db_session.commit()

        # Workspace should be gone
        result = await db_session.scalar(
            select(Workspace).where(Workspace.name == "cascade-ws")
        )
        assert result is None

    async def test_workspace_without_tenant(self, db_session):
        """Workspaces work without tenant_id (backward compat)."""
        ws = Workspace(name="no-tenant-ws")
        db_session.add(ws)
        await db_session.commit()

        assert ws.tenant_id is None
        assert ws.tenant is None


class TestTenantJWT:
    """JWT tenant scoping tests."""

    def test_jwt_with_tenant_id(self):
        """JWT can carry a tenant_id."""
        params = JWTParams(t="", tid="test-tenant-id")
        token = create_jwt(params)
        decoded = verify_jwt(token)

        assert decoded.tid == "test-tenant-id"
        assert decoded.w is None
        assert decoded.ad is None

    def test_jwt_admin_with_tenant(self):
        """Admin JWT with tenant scoping."""
        params = JWTParams(t="", ad=True, tid="tenant-42")
        token = create_jwt(params)
        decoded = verify_jwt(token)

        assert decoded.ad is True
        assert decoded.tid == "tenant-42"

    def test_jwt_without_tenant(self):
        """JWT without tenant_id works (backward compat)."""
        params = JWTParams(t="", w="workspace-1")
        token = create_jwt(params)
        decoded = verify_jwt(token)

        assert decoded.w == "workspace-1"
        assert decoded.tid is None

    def test_jwt_full_scope(self):
        """JWT with all scopes."""
        params = JWTParams(
            t="",
            ad=True,
            tid="t-1",
            w="ws-1",
            p="peer-1",
            s="session-1",
        )
        token = create_jwt(params)
        decoded = verify_jwt(token)

        assert decoded.ad is True
        assert decoded.tid == "t-1"
        assert decoded.w == "ws-1"
        assert decoded.p == "peer-1"
        assert decoded.s == "session-1"


class TestTenantAPI:
    """Tenant CRUD endpoint tests."""

    async def test_create_tenant_admin_only(self, client):
        """Only admin JWTs can create tenants."""
        # Without auth — should fail
        response = await client.post("/v3/tenants", json={"name": "no-auth"})
        assert response.status_code in (401, 403)

    async def test_create_tenant_with_admin(self, admin_client):
        """Admin can create tenants."""
        response = await admin_client.post(
            "/v3/tenants", json={"name": "admin-tenant"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "admin-tenant"
        assert "id" in data

    async def test_create_duplicate_tenant(self, admin_client):
        """Creating a tenant with an existing name returns 409."""
        await admin_client.post("/v3/tenants", json={"name": "dup-tenant"})
        response = await admin_client.post(
            "/v3/tenants", json={"name": "dup-tenant"}
        )
        assert response.status_code == 409

    async def test_list_tenants_admin_only(self, client, admin_client):
        """Only admins can list tenants."""
        response = await client.post("/v3/tenants/list")
        assert response.status_code in (401, 403)

        response = await admin_client.post("/v3/tenants/list")
        assert response.status_code == 200

    async def test_get_tenant(self, admin_client):
        """Get a specific tenant by ID."""
        create_resp = await admin_client.post(
            "/v3/tenants", json={"name": "get-tenant"}
        )
        tenant_id = create_resp.json()["id"]

        response = await admin_client.get(f"/v3/tenants/{tenant_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "get-tenant"

    async def test_get_nonexistent_tenant(self, admin_client):
        """Getting a nonexistent tenant returns 404."""
        response = await admin_client.get("/v3/tenants/nonexistent-id")
        assert response.status_code == 404

    async def test_delete_tenant(self, admin_client):
        """Admin can delete a tenant."""
        create_resp = await admin_client.post(
            "/v3/tenants", json={"name": "del-tenant"}
        )
        tenant_id = create_resp.json()["id"]

        response = await admin_client.delete(f"/v3/tenants/{tenant_id}")
        assert response.status_code == 204

        # Verify gone
        response = await admin_client.get(f"/v3/tenants/{tenant_id}")
        assert response.status_code == 404

    async def test_workspace_with_tenant(self, admin_client):
        """Create a workspace assigned to a tenant."""
        tenant_resp = await admin_client.post(
            "/v3/tenants", json={"name": "ws-tenant"}
        )
        tenant_id = tenant_resp.json()["id"]

        response = await admin_client.post(
            "/v3/workspaces",
            json={"id": "tenant-ws", "tenant_id": tenant_id},
        )
        assert response.status_code in (200, 201)
        data = response.json()
        assert data["tenant_id"] == tenant_id
