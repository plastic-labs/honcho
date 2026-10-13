"""POST /v3/keys under MULTI_TENANT: a minted key carries the minting admin's tenant."""

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from src.config import settings
from src.security import JWTParams, auth, create_jwt, verify_jwt


def _admin_client(client: TestClient, token: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.mark.asyncio
async def test_multi_tenant_key_carries_the_callers_tenant_and_authenticates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Under MULTI_TENANT auth rejects a tenant-less token, so the minted key must
    carry the minting admin's tenant to be usable at all."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    admin = create_jwt(JWTParams(ad=True, tn="tenant-t"))
    _admin_client(client, admin, monkeypatch)

    response = client.post("/v3/keys", params={"workspace_id": "ws-t"})

    assert response.status_code == 200, response.text
    key = response.json()["key"]
    assert verify_jwt(key).tn == "tenant-t"
    params = await auth(
        credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=key),
        workspace_name="ws-t",
    )
    assert params.tn == "tenant-t"


def test_single_tenant_key_is_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Flag off nothing is bound, so the key has no tenant claim and is the same
    token the route minted before it read the tenant."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    _admin_client(client, create_jwt(JWTParams(ad=True)), monkeypatch)

    response = client.post("/v3/keys", params={"workspace_id": "ws-t"})

    assert response.status_code == 200, response.text
    assert response.json()["key"] == create_jwt(JWTParams(w="ws-t"))
