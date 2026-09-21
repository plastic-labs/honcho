"""The paused-tenant set: what the claim's exclusion seam reads, and how it fails.

Two policies under test, because they differ on purpose. The first load at
start fails closed (a process that cannot read the set does not claim); every
later refresh fails open on the last known good set (emptying it would un-pause
everybody). Flag-off, none of it runs and the seam reads None.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import derivation_pause, models
from src.config import settings
from src.crud.deriver import claim_excluded_tenant_ids
from src.derivation_pause import DerivationPauseRefresher
from src.startup import StartupValidationError
from src.telemetry import prometheus_metrics


@dataclass(frozen=True)
class Tenants:
    """Three tenant ids unique to one test: tenants rows outlive the per-test
    table clear (the default tenant must survive), so ids cannot repeat."""

    paused_a: str
    paused_b: str
    live: str


@pytest.fixture
def tenants() -> Tenants:
    suffix = generate_nanoid(size=8)
    # Named so the sorted set is (paused_a, paused_b) regardless of the suffix.
    return Tenants(
        paused_a=f"t-a-{suffix}", paused_b=f"t-b-{suffix}", live=f"t-live-{suffix}"
    )


@pytest.fixture
def multi_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)


async def _seed_tenants(db: AsyncSession, tenants: Tenants) -> None:
    db.add_all(
        [
            models.Tenant(
                tenant_id=tenants.paused_b, tier="shared", derivation_paused=True
            ),
            models.Tenant(tenant_id=tenants.live, tier="shared"),
            models.Tenant(
                tenant_id=tenants.paused_a, tier="shared", derivation_paused=True
            ),
        ]
    )
    await db.commit()


@pytest.mark.asyncio
async def test_refresh_publishes_exactly_the_paused_set_sorted(
    db_session: AsyncSession, tenants: Tenants
) -> None:
    await _seed_tenants(db_session, tenants)

    loaded = await derivation_pause.refresh(db_session)

    expected = (tenants.paused_a, tenants.paused_b)
    assert loaded == expected
    assert derivation_pause.excluded_tenant_ids() == expected
    # The seam the claim builder calls resolves to the same set.
    assert claim_excluded_tenant_ids() == expected
    assert derivation_pause.paused_tenant_count() == 2


@pytest.mark.asyncio
async def test_nobody_paused_reads_as_none_so_the_claim_sql_is_unchanged(
    db_session: AsyncSession, tenants: Tenants
) -> None:
    db_session.add(models.Tenant(tenant_id=tenants.live, tier="shared"))
    await db_session.commit()

    assert await derivation_pause.refresh(db_session) == ()
    assert derivation_pause.excluded_tenant_ids() is None
    assert claim_excluded_tenant_ids() is None


@pytest.mark.asyncio
async def test_unpausing_a_row_drops_it_on_the_next_refresh(
    db_session: AsyncSession, tenants: Tenants
) -> None:
    await _seed_tenants(db_session, tenants)
    await derivation_pause.refresh(db_session)

    tenant = await db_session.get(models.Tenant, tenants.paused_a)
    assert tenant is not None
    tenant.derivation_paused = False
    await db_session.commit()

    assert await derivation_pause.refresh(db_session) == (tenants.paused_b,)
    assert claim_excluded_tenant_ids() == (tenants.paused_b,)


def test_never_loaded_reads_as_none() -> None:
    assert derivation_pause.excluded_tenant_ids() is None
    assert derivation_pause.paused_tenant_count() == 0


@pytest.mark.asyncio
async def test_start_is_a_noop_flag_off_and_never_touches_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", False)

    async def _must_not_run() -> tuple[str, ...]:
        raise AssertionError("flag-off must not read the registry")

    monkeypatch.setattr(derivation_pause, "refresh_from_service_db", _must_not_run)
    refresher = DerivationPauseRefresher()

    await refresher.start()

    assert derivation_pause.excluded_tenant_ids() is None
    await refresher.shutdown()  # nothing to stop; must not raise


@pytest.mark.asyncio
@pytest.mark.usefixtures("multi_tenant")
async def test_first_load_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that cannot read the paused set refuses to start claiming, with
    the startup validators' error type so the boot refusal reads the same."""

    async def _unreadable() -> tuple[str, ...]:
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(derivation_pause, "refresh_from_service_db", _unreadable)
    refresher = DerivationPauseRefresher()

    with pytest.raises(StartupValidationError, match="registry unreadable"):
        await refresher.start()

    assert refresher._task is None  # pyright: ignore[reportPrivateUsage]
    assert derivation_pause.excluded_tenant_ids() is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("multi_tenant")
async def test_a_later_failure_keeps_the_last_known_good_set_and_counts(
    db_session: AsyncSession,
    tenants: Tenants,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail OPEN: a transient error must not un-pause anybody."""
    await _seed_tenants(db_session, tenants)
    await derivation_pause.refresh(db_session)
    expected = (tenants.paused_a, tenants.paused_b)
    assert claim_excluded_tenant_ids() == expected

    async def _blip() -> tuple[str, ...]:
        raise RuntimeError("database blip")

    monkeypatch.setattr(derivation_pause, "refresh_from_service_db", _blip)
    failures: list[None] = []
    monkeypatch.setattr(
        prometheus_metrics,
        "record_paused_tenants_refresh_failure",
        lambda: failures.append(None),
    )

    await DerivationPauseRefresher().refresh_once()

    assert claim_excluded_tenant_ids() == expected
    assert len(failures) == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("multi_tenant")
async def test_start_loads_then_shutdown_stops_the_timer(
    db_session: AsyncSession, tenants: Tenants
) -> None:
    await _seed_tenants(db_session, tenants)
    refresher = DerivationPauseRefresher()

    await refresher.start()
    try:
        assert claim_excluded_tenant_ids() == (tenants.paused_a, tenants.paused_b)
        assert refresher._task is not None  # pyright: ignore[reportPrivateUsage]
    finally:
        await refresher.shutdown()
    assert refresher._task is None  # pyright: ignore[reportPrivateUsage]
