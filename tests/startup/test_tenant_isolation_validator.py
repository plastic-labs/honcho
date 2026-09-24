"""Startup tenant-isolation validator.

Flag-off it must be a pure no-op — self-host never pays for it and never sees it.
Flag-on it refuses the half-states where isolation looks enabled but cannot hold
(unsafe pooler, RLS not enforced, no service role) or cannot serve (auth off on an
API instance; a deriver takes its tenant from the claimed work unit, not a JWT).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import AppSettings, settings
from src.startup import StartupValidationError, validate_tenant_isolation
from src.startup.tenant_isolation_validator import (
    _RLS_REQUIRED_TABLES,  # pyright: ignore[reportPrivateUsage]
)
from tests.conftest import untouchable_engine

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

    await validate_tenant_isolation(_NO_ENGINE, instance_type="api", app_settings=s)


# ---------------------------------------------------------------------------
# Flag on: configuration-only refusals (no database needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_boot_when_auth_is_off() -> None:
    s = _flag_on_settings()
    s.AUTH.USE_AUTH = False
    with pytest.raises(StartupValidationError, match="AUTH_USE_AUTH is off"):
        await validate_tenant_isolation(_NO_ENGINE, instance_type="api", app_settings=s)


@pytest.mark.asyncio
async def test_deriver_instance_skips_the_auth_check() -> None:
    # A deriver binds its tenant from the claimed work unit's key and verifies no
    # JWT, so it is not forced to carry the API's auth config. Everything else it
    # would be checked for is set so the validator returns before touching the DB.
    s = _flag_on_settings()
    s.AUTH.USE_AUTH = False
    s.MULTI_TENANT_SKIP_RLS_ASSERT = True
    await validate_tenant_isolation(_NO_ENGINE, instance_type="deriver", app_settings=s)


@pytest.mark.asyncio
async def test_refuses_boot_under_a_transaction_mode_pooler() -> None:
    s = _flag_on_settings()
    s.DB.POOLER_MODE = "transaction"
    with pytest.raises(StartupValidationError, match="DB_POOLER_MODE='transaction'"):
        await validate_tenant_isolation(_NO_ENGINE, instance_type="api", app_settings=s)


@pytest.mark.asyncio
async def test_skip_rls_assert_warns_and_returns_before_introspection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    s = _flag_on_settings()
    s.MULTI_TENANT_SKIP_RLS_ASSERT = True
    with caplog.at_level("WARNING"):
        await validate_tenant_isolation(_NO_ENGINE, instance_type="api", app_settings=s)
    assert "MULTI_TENANT_SKIP_RLS_ASSERT is set" in caplog.text


@pytest.mark.asyncio
async def test_fails_closed_when_introspection_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the retry budget exhausts, the validator crashes — uncertainty is not
    a green light to serve traffic."""
    call_count = 0

    async def always_raise(
        _engine: AsyncEngine, _schema: str
    ) -> dict[str, tuple[bool, bool]]:
        nonlocal call_count
        call_count += 1
        raise OperationalError("SELECT 1", {}, Exception("DB unreachable"))

    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._introspect_rls_once", always_raise
    )
    monkeypatch.setattr(
        "src.startup.tenant_isolation_validator._RETRY_BACKOFF_SECONDS", 0.0
    )

    with pytest.raises(StartupValidationError, match="could not validate"):
        await validate_tenant_isolation(
            _NO_ENGINE, instance_type="api", app_settings=_flag_on_settings()
        )

    assert call_count == 3, "should exhaust the retry budget before failing"


# ---------------------------------------------------------------------------
# Flag on: against the real migrated test schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_boot_when_rls_is_not_enforced(db_engine: AsyncEngine) -> None:
    """The migrated schema carries no RLS (policies are provisioned out of band), which
    is exactly what a self-hoster who flips the flag has. Boot must refuse."""
    with pytest.raises(StartupValidationError, match=r"RLS is not enabled\+forced"):
        await validate_tenant_isolation(
            db_engine, instance_type="api", app_settings=_flag_on_settings()
        )


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
            await validate_tenant_isolation(
                db_engine, instance_type="api", app_settings=s
            )

        # The fully configured flag-on state: auth on, safe pooler, RLS enforced,
        # service role set. The one combination that boots.
        s.DB.SERVICE_CONNECTION_URI = (
            "postgresql+psycopg://service:service@localhost:5432/postgres"
        )
        await validate_tenant_isolation(db_engine, instance_type="api", app_settings=s)
    finally:
        await _set_rls(db_engine, enforced=False)
