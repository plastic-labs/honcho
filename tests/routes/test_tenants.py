"""Route tests for the tenant registry API — see src/routers/tenants.py.

The tenants router sits ABOVE tenant scope: it has its own service-secret auth
plane (``require_tenant_api``) instead of ``require_auth``, and it talks to the
database through ``service_db`` (patched onto the per-test engine by conftest's
``mock_tracked_db``, like every other service_db import site).
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings

# A stable, non-empty service secret for the enabled cases. Constant-time
# comparison against this is the whole of the tenant API's auth check.
SECRET = "test-tenant-api-secret"
HEADER = "X-Tenant-Api-Key"


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> str:
    """Turn the tenant API on (MULTI_TENANT + a configured secret) and return it."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.TENANT_API, "SECRET", SECRET)
    return SECRET


def _create_body(
    tenant_id: str,
    tier: str = "pro",
    vector_correlation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "tenant_id": tenant_id,
        "tier": tier,
        "vector_correlation_id": vector_correlation_id,
    }


# ---------------------------------------------------------------------------
# Auth plane: disabled / unauthorized matrix
# ---------------------------------------------------------------------------


def test_disabled_when_multi_tenant_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Flag off is a hard 405 even with the correct secret in the header."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings.TENANT_API, "SECRET", SECRET)

    response = client.post(
        "/v3/tenants",
        json=_create_body(generate_nanoid()),
        headers={HEADER: SECRET},
    )
    assert response.status_code == 405, response.text


def test_disabled_when_secret_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """MULTI_TENANT on but no configured secret → still disabled (405)."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.TENANT_API, "SECRET", None)

    response = client.post(
        "/v3/tenants",
        json=_create_body(generate_nanoid()),
        headers={HEADER: SECRET},
    )
    assert response.status_code == 405, response.text


def test_missing_header_unauthorized(
    client: TestClient,
    enabled: str,  # pyright: ignore[reportUnusedParameter]
):
    """Enabled but no header → 401."""
    response = client.post("/v3/tenants", json=_create_body(generate_nanoid()))
    assert response.status_code == 401, response.text


def test_wrong_header_unauthorized(client: TestClient, enabled: str):
    """Enabled but wrong secret → 401."""
    response = client.post(
        "/v3/tenants",
        json=_create_body(generate_nanoid()),
        headers={HEADER: enabled + "-nope"},
    )
    assert response.status_code == 401, response.text


def test_non_ascii_header_bytes_unauthorized(
    client: TestClient,
    enabled: str,  # pyright: ignore[reportUnusedParameter]
):
    """A raw high-byte header value must 401, never 500.

    compare_digest rejects non-ASCII str with TypeError; the guard compares
    bytes so hostile raw bytes fail closed as a clean auth mismatch. httpx
    blocks non-ASCII header *strings* client-side, so send bytes directly.
    """
    response = client.get(
        f"/v3/tenants/{generate_nanoid()}", headers={HEADER.encode(): b"caf\xe9"}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Create: idempotency contract
# ---------------------------------------------------------------------------


def test_create_tenant(client: TestClient, enabled: str):
    """Happy path: a fresh tenant is created with 201 and echoes its fields."""
    tenant_id = generate_nanoid()
    response = client.post(
        "/v3/tenants",
        json=_create_body(tenant_id, tier="enterprise", vector_correlation_id="vec-1"),
        headers={HEADER: enabled},
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["tenant_id"] == tenant_id
    assert data["tier"] == "enterprise"
    assert data["vector_correlation_id"] == "vec-1"
    assert "created_at" in data


def test_create_tenant_idempotent_identical(client: TestClient, enabled: str):
    """Re-POSTing identical fields returns the existing row with 200, not 201."""
    tenant_id = generate_nanoid()
    body = _create_body(tenant_id, tier="pro", vector_correlation_id="vec-2")

    first = client.post("/v3/tenants", json=body, headers={HEADER: enabled})
    assert first.status_code == 201, first.text

    second = client.post("/v3/tenants", json=body, headers={HEADER: enabled})
    assert second.status_code == 200, second.text
    assert second.json()["tenant_id"] == tenant_id
    assert second.json() == first.json()


def test_create_tenant_conflict_different_tier(client: TestClient, enabled: str):
    """Same tenant_id with a different tier is a 409 — this API never mutates."""
    tenant_id = generate_nanoid()

    first = client.post(
        "/v3/tenants",
        json=_create_body(tenant_id, tier="pro"),
        headers={HEADER: enabled},
    )
    assert first.status_code == 201, first.text

    conflict = client.post(
        "/v3/tenants",
        json=_create_body(tenant_id, tier="enterprise"),
        headers={HEADER: enabled},
    )
    assert conflict.status_code == 409, conflict.text


def test_create_tenant_invalid_id(client: TestClient, enabled: str):
    """A tenant_id violating RESOURCE_NAME_PATTERN (a ':') is a 422."""
    response = client.post(
        "/v3/tenants",
        json=_create_body("bad:id"),
        headers={HEADER: enabled},
    )
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def test_get_tenant(client: TestClient, enabled: str):
    """GET returns the created tenant's fields with 200."""
    tenant_id = generate_nanoid()
    created = client.post(
        "/v3/tenants",
        json=_create_body(tenant_id, tier="pro", vector_correlation_id="vec-3"),
        headers={HEADER: enabled},
    )
    assert created.status_code == 201, created.text

    response = client.get(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["tenant_id"] == tenant_id
    assert data["tier"] == "pro"
    assert data["vector_correlation_id"] == "vec-3"


def test_get_tenant_missing(client: TestClient, enabled: str):
    """GET on an unknown tenant is a 404."""
    response = client.get(f"/v3/tenants/{generate_nanoid()}", headers={HEADER: enabled})
    assert response.status_code == 404, response.text


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_empty_tenant(client: TestClient, enabled: str):
    """An empty tenant deletes with 204 and is then gone (404 on GET)."""
    tenant_id = generate_nanoid()
    created = client.post(
        "/v3/tenants",
        json=_create_body(tenant_id),
        headers={HEADER: enabled},
    )
    assert created.status_code == 201, created.text

    deleted = client.delete(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled})
    assert deleted.status_code == 204, deleted.text

    gone = client.get(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled})
    assert gone.status_code == 404, gone.text


def test_delete_default_tenant_forbidden(client: TestClient, enabled: str):
    """The bootstrap 'default' tenant cannot be deleted → 422."""
    response = client.delete(
        f"/v3/tenants/{models.DEFAULT_TENANT_ID}", headers={HEADER: enabled}
    )
    assert response.status_code == 422, response.text


def test_delete_missing_tenant(client: TestClient, enabled: str):
    """DELETE on an unknown tenant is a 404."""
    response = client.delete(
        f"/v3/tenants/{generate_nanoid()}", headers={HEADER: enabled}
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_delete_non_empty_tenant_conflict(
    client: TestClient, enabled: str, db_session: AsyncSession
):
    """A tenant with a dependent row (a Workspace) refuses deletion with 409.

    The tenant-scoped tables FK ``tenants.tenant_id`` with no ON DELETE action,
    so Postgres refuses the delete and the CRUD layer surfaces it as a conflict.
    """
    tenant_id = generate_nanoid()
    db_session.add(models.Tenant(tenant_id=tenant_id, tier="pro"))
    # Workspace.name is unique within a tenant; id defaults to a 21-char nanoid.
    db_session.add(models.Workspace(name=generate_nanoid(), tenant_id=tenant_id))
    await db_session.commit()

    response = client.delete(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled})
    assert response.status_code == 409, response.text

    # The tenant survived the refused delete.
    still_there = client.get(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled})
    assert still_there.status_code == 200, still_there.text


# ---------------------------------------------------------------------------
# PATCH: the registry's one mutation door, and its allowlist
# ---------------------------------------------------------------------------


def _created(client: TestClient, enabled: str, tenant_id: str) -> dict[str, Any]:
    response = client.post(
        "/v3/tenants", json=_create_body(tenant_id), headers={HEADER: enabled}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_echoes_derivation_paused_false(client: TestClient, enabled: str):
    assert _created(client, enabled, generate_nanoid())["derivation_paused"] is False


def test_create_rejects_a_mutable_field_instead_of_dropping_it(
    client: TestClient, enabled: str
):
    """Seeding a pause is create -> PATCH; a pause passed to create must not vanish."""
    body = _create_body(generate_nanoid()) | {"derivation_paused": True}
    response = client.post("/v3/tenants", json=body, headers={HEADER: enabled})
    assert response.status_code == 422, response.text


def test_patch_pauses_resumes_and_is_idempotent(client: TestClient, enabled: str):
    tenant_id = generate_nanoid()
    _created(client, enabled, tenant_id)

    paused = client.patch(
        f"/v3/tenants/{tenant_id}",
        json={"derivation_paused": True},
        headers={HEADER: enabled},
    )
    assert paused.status_code == 200, paused.text
    assert paused.json()["derivation_paused"] is True
    assert paused.json()["tenant_id"] == tenant_id

    # Webhook-driven callers retry: re-asserting the held value is a 200, not a 409.
    again = client.patch(
        f"/v3/tenants/{tenant_id}",
        json={"derivation_paused": True},
        headers={HEADER: enabled},
    )
    assert again.status_code == 200, again.text
    assert again.json() == paused.json()

    # The read path sees it, and the create contract still holds.
    assert (
        client.get(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled}).json()[
            "derivation_paused"
        ]
        is True
    )
    retry_create = client.post(
        "/v3/tenants", json=_create_body(tenant_id), headers={HEADER: enabled}
    )
    assert retry_create.status_code == 200, retry_create.text

    resumed = client.patch(
        f"/v3/tenants/{tenant_id}",
        json={"derivation_paused": False},
        headers={HEADER: enabled},
    )
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["derivation_paused"] is False


def test_patch_unknown_tenant_is_404(client: TestClient, enabled: str):
    response = client.patch(
        f"/v3/tenants/{generate_nanoid()}",
        json={"derivation_paused": True},
        headers={HEADER: enabled},
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "body",
    [
        {"tier": "enterprise"},
        {"vector_correlation_id": "moved"},
        {"tenant_id": "someone-else"},
        {"created_at": "2026-01-01T00:00:00Z"},
        {"derivation_paused": True, "tier": "enterprise"},
        {},
    ],
    ids=["tier", "vector_correlation_id", "tenant_id", "created_at", "mixed", "empty"],
)
def test_patch_rejects_everything_outside_the_allowlist(
    client: TestClient, enabled: str, body: dict[str, Any]
):
    """The allowlist is the whole contract: one field today, a 422 for anything else.

    A mixed body is refused whole rather than partially applied — otherwise a
    client could learn that a pause "worked" while its tier change was dropped.
    """
    tenant_id = generate_nanoid()
    before = _created(client, enabled, tenant_id)

    response = client.patch(
        f"/v3/tenants/{tenant_id}", json=body, headers={HEADER: enabled}
    )
    assert response.status_code == 422, response.text

    after = client.get(f"/v3/tenants/{tenant_id}", headers={HEADER: enabled}).json()
    assert after == before


def test_patch_requires_the_service_secret(client: TestClient, enabled: str):
    response = client.patch(
        f"/v3/tenants/{generate_nanoid()}", json={"derivation_paused": True}
    )
    assert response.status_code == 401, response.text
