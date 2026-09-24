"""Startup validator for the multi-tenant isolation binding."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.startup.embedding_validator import StartupValidationError
from src.startup.tenant_isolation_validator import (
    _assert_tenant_role_cannot_bypass_rls,  # pyright: ignore[reportPrivateUsage]
    validate_tenant_isolation,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests for the role assertion
# ---------------------------------------------------------------------------


def test_assert_tenant_role_passes_for_an_ordinary_role() -> None:
    _assert_tenant_role_cannot_bypass_rls("honcho_app", False, False)


def test_assert_tenant_role_raises_for_a_superuser() -> None:
    with pytest.raises(StartupValidationError, match="is a superuser") as excinfo:
        _assert_tenant_role_cannot_bypass_rls("postgres", True, False)
    message = str(excinfo.value)
    assert "'postgres'" in message
    assert "DB_CONNECTION_URI" in message
    assert "DB_SERVICE_CONNECTION_URI" in message


def test_assert_tenant_role_raises_for_a_bypassrls_role() -> None:
    with pytest.raises(StartupValidationError, match="has BYPASSRLS") as excinfo:
        _assert_tenant_role_cannot_bypass_rls("honcho_migrator", False, True)
    assert "'honcho_migrator'" in str(excinfo.value)


def test_assert_tenant_role_names_both_reasons_when_both_apply() -> None:
    with pytest.raises(StartupValidationError) as excinfo:
        _assert_tenant_role_cannot_bypass_rls("postgres", True, True)
    message = str(excinfo.value)
    assert "is a superuser" in message
    assert "has BYPASSRLS" in message


# ---------------------------------------------------------------------------
# Retry / fail-closed behavior for the role introspection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_role_introspection_fails_closed_when_it_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the retry budget exhausts, the validator crashes — uncertainty
    is not a green light to serve traffic."""
    call_count = 0

    async def always_raise(_engine: AsyncEngine) -> tuple[str, bool, bool]:
        nonlocal call_count
        call_count += 1
        raise OperationalError("SELECT 1", {}, Exception("DB unreachable"))

    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_tenant_role_once",
        always_raise,
    )
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._RETRY_BACKOFF_SECONDS", 0.0
    )
    # Reach the role-introspection call: RLS must already read as enforced.
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_rls_with_retry",
        AsyncMock(
            return_value={
                table: (True, True)
                for table in (
                    "workspaces",
                    "peers",
                    "sessions",
                    "messages",
                    "message_embeddings",
                    "collections",
                    "documents",
                    "document_sources",
                    "session_peers",
                    "webhook_endpoints",
                )
            }
        ),
    )
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings, "MULTI_TENANT_SKIP_RLS_ASSERT", False)
    monkeypatch.setattr(settings.DB, "POOLER_MODE", "session")

    with pytest.raises(StartupValidationError, match="could not validate"):
        await validate_tenant_isolation(AsyncMock())

    assert call_count == 3, "should exhaust the retry budget before failing"


# ---------------------------------------------------------------------------
# Full-flow validate_tenant_isolation: role ok / superuser / bypassrls / flag-off
#
# No test exercises the real validator against a real connection with
# MULTI_TENANT on (the local/CI DB role is itself the postgres superuser, and a
# plain migrated test DB has no RLS policies applied -- see
# tests/integration/test_rls_isolation.py for the one place RLS is actually
# applied to a throwaway DB). So these fake both introspection seams
# (_introspect_rls_with_retry, _introspect_tenant_role_with_retry) the same
# way test_role_introspection_fails_closed_when_it_keeps_failing does above.
# ---------------------------------------------------------------------------


def _enforced_rls() -> dict[str, tuple[bool, bool]]:
    return {
        table: (True, True)
        for table in (
            "workspaces",
            "peers",
            "sessions",
            "messages",
            "message_embeddings",
            "collections",
            "documents",
            "document_sources",
            "session_peers",
            "webhook_endpoints",
        )
    }


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_rls_with_retry",
        AsyncMock(return_value=_enforced_rls()),
    )
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings, "MULTI_TENANT_SKIP_RLS_ASSERT", False)
    monkeypatch.setattr(settings.DB, "POOLER_MODE", "session")
    monkeypatch.setattr(settings.DB, "SERVICE_CONNECTION_URI", "postgresql://service")


@pytest.mark.asyncio
async def test_validate_tenant_isolation_passes_when_role_cannot_bypass_rls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_tenant_role_with_retry",
        AsyncMock(return_value=("honcho_app", False, False)),
    )

    await validate_tenant_isolation(AsyncMock())


@pytest.mark.asyncio
async def test_validate_tenant_isolation_fails_closed_for_a_superuser_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_tenant_role_with_retry",
        AsyncMock(return_value=("postgres", True, False)),
    )

    with pytest.raises(StartupValidationError, match="is a superuser"):
        await validate_tenant_isolation(AsyncMock())


@pytest.mark.asyncio
async def test_validate_tenant_isolation_fails_closed_for_a_bypassrls_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_tenant_role_with_retry",
        AsyncMock(return_value=("honcho_migrator", False, True)),
    )

    with pytest.raises(StartupValidationError, match="has BYPASSRLS"):
        await validate_tenant_isolation(AsyncMock())


@pytest.mark.asyncio
async def test_validate_tenant_isolation_skips_the_role_check_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MULTI_TENANT off: the validator returns before introspecting anything,
    so a superuser/BYPASSRLS role (the self-host default) never trips it."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    rls_mock = AsyncMock(return_value=_enforced_rls())
    role_mock = AsyncMock(return_value=("postgres", True, True))
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_rls_with_retry", rls_mock
    )
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_tenant_role_with_retry",
        role_mock,
    )

    await validate_tenant_isolation(AsyncMock())

    rls_mock.assert_not_called()
    role_mock.assert_not_called()
