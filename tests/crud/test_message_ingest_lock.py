"""The ingest advisory lock's key: tenant-scoped under MULTI_TENANT, flag-off SQL unchanged."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.crud.message import _ingest_lock  # pyright: ignore[reportPrivateUsage]
from src.db import tenant_context


def test_flag_off_sql_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    stmt, params = _ingest_lock("default", "s1")
    assert str(stmt) == (
        "SELECT pg_advisory_xact_lock(hashtext(:workspace_name), hashtext(:session_name))"
    )
    assert params == {"workspace_name": "default", "session_name": "s1"}


def test_flag_on_keys_on_tenant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    keys: list[dict[str, str]] = []
    for tenant in ("tenant-a", "tenant-b"):
        token = tenant_context.set(tenant)
        try:
            stmt, params = _ingest_lock("default", "s1")
        finally:
            tenant_context.reset(token)
        assert ":tenant" in str(stmt)
        keys.append(params)
    assert keys[0]["tenant"] == "tenant-a"
    assert keys[1]["tenant"] == "tenant-b"


def test_flag_on_without_tenant_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="requires a tenant"):
            _ingest_lock("default", "s1")
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_flag_on_statement_executes(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set("tenant-a")
    try:
        stmt, params = _ingest_lock("default", "s1")
    finally:
        tenant_context.reset(token)
    await db_session.execute(stmt, params)
