"""Startup tenant-isolation validator.

Flag-off it must be a pure no-op — self-host never pays for it and never sees it.
Flag-on it refuses the half-states where isolation looks enabled but cannot hold
(unsafe pooler, RLS not enforced, no service role) or cannot serve (auth off).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import AppSettings, settings
from src.startup import StartupValidationError, validate_tenant_isolation
from src.startup.tenant_isolation_validator import (
    _RLS_REQUIRED_TABLES,  # pyright: ignore[reportPrivateUsage]
)
from tests.startup import untouchable_engine

_NO_ENGINE = untouchable_engine()


def _flag_on_settings() -> AppSettings:
    """A copy of the live settings in a fully serviceable flag-on configuration.

    Tests flip exactly one thing back to make the validator refuse boot, so each
    failure is attributable to that one thing.
    """
    s = settings.model_copy(deep=True)
    s.MULTI_TENANT = True
    s.MULTI_TENANT_SKIP_RLS_ASSERT = False
    s.AUTH.USE_AUTH = True
    s.AUTH.JWT_SECRET = "test-secret"
    s.DB.POOLER_MODE = "session"
    s.DB.SCHEMA = "public"  # where conftest builds the migrated test schema
    s.DB.SERVICE_CONNECTION_URI = (
        "postgresql+psycopg://service:service@localhost:5432/postgres"
    )
    return s


async def _set_rls(engine: AsyncEngine, *, enforced: bool) -> None:
    """ENABLE+FORCE (or DISABLE+NO FORCE) RLS on the data tables, without policies.

    Enough for the validator's pg_class introspection; no policy is created, so the
    tables are default-deny for every role while enforced — restore before any test
    reads them.
    """
    enable, force = ("ENABLE", "FORCE") if enforced else ("DISABLE", "NO FORCE")
    async with engine.begin() as conn:
        for table in _RLS_REQUIRED_TABLES:
            await conn.execute(text(f"ALTER TABLE {table} {enable} ROW LEVEL SECURITY"))
            await conn.execute(text(f"ALTER TABLE {table} {force} ROW LEVEL SECURITY"))


# ---------------------------------------------------------------------------
# Flag off: the single-tenant default
# ---------------------------------------------------------------------------


def test_the_shipped_default_is_single_tenant() -> None:
    # Pinned on the class, not the process: a developer's .env cannot make this pass
    # or fail. The default a self-hoster inherits without configuring anything.
    assert AppSettings.model_fields["MULTI_TENANT"].default is False


@pytest.mark.asyncio
async def test_flag_off_is_a_no_op_that_never_touches_the_engine() -> None:
    s = settings.model_copy(deep=True)
    s.MULTI_TENANT = False
    # Every subordinate set to a value the flag-on path rejects, to prove none of
    # them is even read when the flag is off.
    s.AUTH.USE_AUTH = False
    s.DB.POOLER_MODE = "transaction"
    s.DB.SERVICE_CONNECTION_URI = None
    s.MULTI_TENANT_SKIP_RLS_ASSERT = False

    await validate_tenant_isolation(_NO_ENGINE, app_settings=s)


# ---------------------------------------------------------------------------
# Flag on: configuration-only refusals (no database needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_boot_when_auth_is_off() -> None:
    s = _flag_on_settings()
    s.AUTH.USE_AUTH = False
    with pytest.raises(StartupValidationError, match="AUTH_USE_AUTH is off"):
        await validate_tenant_isolation(_NO_ENGINE, app_settings=s)


@pytest.mark.asyncio
async def test_refuses_boot_under_a_transaction_mode_pooler() -> None:
    s = _flag_on_settings()
    s.DB.POOLER_MODE = "transaction"
    with pytest.raises(StartupValidationError, match="DB_POOLER_MODE='transaction'"):
        await validate_tenant_isolation(_NO_ENGINE, app_settings=s)


@pytest.mark.asyncio
async def test_skip_rls_assert_warns_and_returns_before_introspection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    s = _flag_on_settings()
    s.MULTI_TENANT_SKIP_RLS_ASSERT = True
    with caplog.at_level("WARNING"):
        await validate_tenant_isolation(_NO_ENGINE, app_settings=s)
    assert "MULTI_TENANT_SKIP_RLS_ASSERT is set" in caplog.text


# ---------------------------------------------------------------------------
# Flag on: against the real migrated test schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_boot_when_rls_is_not_enforced(db_engine: AsyncEngine) -> None:
    """The migrated schema carries no RLS (policies are provisioned out of band), which
    is exactly what a self-hoster who flips the flag has. Boot must refuse."""
    with pytest.raises(StartupValidationError, match=r"RLS is not enabled\+forced"):
        await validate_tenant_isolation(db_engine, app_settings=_flag_on_settings())


@pytest.mark.asyncio
async def test_requires_a_service_role_once_rls_is_enforced(
    db_engine: AsyncEngine,
) -> None:
    s = _flag_on_settings()
    await _set_rls(db_engine, enforced=True)
    try:
        s.DB.SERVICE_CONNECTION_URI = None
        with pytest.raises(
            StartupValidationError, match="DB_SERVICE_CONNECTION_URI is unset"
        ):
            await validate_tenant_isolation(db_engine, app_settings=s)

        # The fully configured flag-on state: auth on, safe pooler, RLS enforced,
        # service role set. The one combination that boots.
        s.DB.SERVICE_CONNECTION_URI = (
            "postgresql+psycopg://service:service@localhost:5432/postgres"
        )
        await validate_tenant_isolation(db_engine, app_settings=s)
    finally:
        await _set_rls(db_engine, enforced=False)
