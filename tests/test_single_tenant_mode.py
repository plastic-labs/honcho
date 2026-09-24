"""Single-tenant mode is one switch.

``MULTI_TENANT`` is the mode. Every other tenant-related setting is a subordinate the
flag-on path consults and the flag-off path must ignore, so a self-hoster who copies
a value out of a shared-instance config gets unchanged single-tenant behavior, never
a half-state. Every test here runs flag-off with every subordinate set to a value
flag-on would act on, and asserts the same thing: nothing changed. The values are
set on purpose, because the claim under test is that they do not matter. A new
tenant-subordinate setting belongs in ``_SUBORDINATES``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.config import settings
from src.db import tenant_context
from src.dependencies import tracked_db as real_tracked_db
from src.models import (
    DEFAULT_TENANT_ID,
    _default_tenant_id,  # pyright: ignore[reportPrivateUsage]
)
from src.security import JWTParams, create_jwt, require_auth
from src.startup import validate_tenant_isolation
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key
from tests.conftest import untouchable_engine

_STRAY_SECRET = "copied-from-a-shared-instance-config"

Subordinate = Callable[[pytest.MonkeyPatch], None]


def _skip_rls_assert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT_SKIP_RLS_ASSERT", True)


def _service_connection_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        settings.DB,
        "SERVICE_CONNECTION_URI",
        "postgresql+psycopg://service:service@127.0.0.1:1/unreachable",
    )


def _transaction_pooler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.DB, "POOLER_MODE", "transaction")


def _tenant_api_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.TENANT_API, "SECRET", _STRAY_SECRET)


# Every setting the flag-on path consults. Extend this when a new one is added.
_SUBORDINATES: tuple[Subordinate, ...] = (
    _skip_rls_assert,
    _service_connection_uri,
    _transaction_pooler,
    _tenant_api_secret,
)


@pytest.fixture(autouse=True)
def _single_tenant_with_stray_subordinates(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag off, with every subordinate set to a value flag-on would act on."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    for apply in _SUBORDINATES:
        apply(monkeypatch)


@pytest.mark.asyncio
async def test_isolation_validator_is_a_no_op() -> None:
    # A transaction-mode pooler or an unreachable service role are boot refusals
    # flag-on; flag-off the validator returns before reading either, or the engine.
    await validate_tenant_isolation(untouchable_engine(), instance_type="api")


def test_tenant_api_stays_disabled(client: TestClient) -> None:
    # The stray secret alone must not open the above-tenant plane.
    response = client.post(
        "/v3/tenants",
        json={"tenant_id": "acme", "tier": "pro", "vector_correlation_id": None},
        headers={"X-Tenant-Api-Key": _STRAY_SECRET},
    )
    assert response.status_code == 405, response.text


@pytest.mark.asyncio
async def test_tenant_bearing_token_binds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A control plane that mints tenant claims for every tenant may also talk to a
    # flag-off instance. The claim survives on the params and is otherwise ignored —
    # rows keep stamping the default tenant.
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=create_jwt(JWTParams(tn="acme", w="ws-a"))
    )
    request = Request(
        {"type": "http", "query_string": b"", "path_params": {}, "headers": []}
    )
    dependency = require_auth()(request=request, credentials=credentials)
    params = await dependency.__anext__()
    try:
        assert params.tn == "acme"
        assert tenant_context.get() is None
        assert _default_tenant_id() == DEFAULT_TENANT_ID
    finally:
        await dependency.aclose()
    assert tenant_context.get() is None


def test_work_unit_keys_are_not_namespaced() -> None:
    # Even a stray ambient tenant does not leak into keys flag-off: the prefix is
    # gated on the flag, so flag-off keys stay parseable as tenant-less.
    token = tenant_context.set("acme")
    try:
        key = construct_work_unit_key(
            "ws",
            {"task_type": "representation", "session_name": "s", "observed": "alice"},
        )
    finally:
        tenant_context.reset(token)
    assert key == "representation:ws:s:alice"
    assert parse_work_unit_key(key).tenant_id is None


@pytest.mark.asyncio
async def test_tracked_db_needs_no_tenant() -> None:
    # The fail-closed guard is flag-on only; a tenant-less session is an ordinary one.
    assert tenant_context.get() is None
    async with real_tracked_db("single_tenant_probe", read_only=True) as db:
        assert (await db.execute(text("SELECT 1"))).scalar() == 1
