"""POST /v3/keys under MULTI_TENANT: a minted key carries the minting admin's tenant."""

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from src.config import settings
from src.security import JWTParams, auth, create_jwt, verify_jwt


def _as_admin(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, *, tn: str | None
) -> TestClient:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    client.headers["Authorization"] = f"Bearer {create_jwt(JWTParams(ad=True, tn=tn))}"
    return client


@pytest.mark.asyncio
async def test_multi_tenant_key_carries_the_callers_tenant_and_authenticates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Under MULTI_TENANT auth rejects a tenant-less token, so the minted key must
    carry the minting admin's tenant to be usable at all."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    _as_admin(client, monkeypatch, tn="tenant-t")

    response = client.post("/v3/keys", params={"workspace_id": "ws-t"})

    assert response.status_code == 200, response.text
    key = response.json()["key"]
    assert verify_jwt(key).tn == "tenant-t"
    params = await auth(
        credentials=HTTPAuthorizationCredentials(scheme="Bearer", credentials=key),
        workspace_name="ws-t",
    )
    assert params.tn == "tenant-t"


def test_single_tenant_key_carries_no_tenant(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Flag off the key carries no tenant claim, even when the minting admin's
    token has one."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    _as_admin(client, monkeypatch, tn="tenant-t")

    response = client.post("/v3/keys", params={"workspace_id": "ws-t"})

    assert response.status_code == 200, response.text
    assert response.json()["key"] == create_jwt(JWTParams(w="ws-t"))
