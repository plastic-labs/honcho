"""Per-tenant fairness metrics: claims, claim wait and in-flight units by tenant.

Off unless DERIVER_TENANT_FAIRNESS_METRICS is on. The claim runs with no tenant
bound, so each series takes its tenant from the claimed unit's key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from prometheus_client import REGISTRY
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.crud.deriver import get_deriver_metrics
from src.deriver.queue_manager import QueueManager
from src.telemetry import prometheus_metrics
from tests.deriver.conftest import SeedWorkUnit

NAMESPACE = "fairness-test"


def unit_key(tenant_id: str, name: str) -> str:
    return f"{tenant_id}:representation:workspace:{name}:peer"


def sample(name: str, tenant_id: str) -> float | None:
    return REGISTRY.get_sample_value(
        name, {"namespace": NAMESPACE, "tenant_id": tenant_id}
    )


@pytest.fixture
def metrics_on(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Enable metrics under a test namespace, with the fairness flag as given."""

    def _configure(fairness: bool) -> None:
        monkeypatch.setattr(settings.METRICS, "ENABLED", True)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", NAMESPACE)
        monkeypatch.setattr(settings.DERIVER, "TENANT_FAIRNESS_METRICS", fairness)

    return _configure


async def seed_units(
    seed_work_unit: SeedWorkUnit, tenant_id: str, ages_seconds: Sequence[int]
) -> list[str]:
    keys: list[str] = []
    for index, age in enumerate(ages_seconds):
        key = unit_key(tenant_id, f"{tenant_id}-{index}")
        await seed_work_unit(
            key, tenant_id=tenant_id, token_counts=(1,), ages_seconds=(age,)
        )
        keys.append(key)
    return keys


@pytest.mark.asyncio
class TestClaimMetrics:
    async def test_each_claimed_unit_is_counted_under_its_tenant(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
        metrics_on: Callable[[bool], None],
    ) -> None:
        batch_gate_settings(max_age_seconds=60, workers=5)
        metrics_on(True)
        whale, minnow = f"t-{generate_nanoid(size=8)}", f"t-{generate_nanoid(size=8)}"
        await seed_units(seed_work_unit, whale, (600, 500, 400))
        await seed_units(seed_work_unit, minnow, (300,))

        claimed = await QueueManager().get_and_claim_work_units()

        assert len(claimed) == 4
        assert sample("deriver_claims_total", whale) == 3
        assert sample("deriver_claims_total", minnow) == 1
        assert sample("deriver_claim_wait_seconds_count", whale) == 3
        # The waits are the seeded ages, so their sum is at least 600+500+400.
        whale_wait = sample("deriver_claim_wait_seconds_sum", whale)
        assert whale_wait is not None and whale_wait >= 1500

    async def test_a_candidate_another_claimer_won_is_not_counted(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
        metrics_on: Callable[[bool], None],
    ) -> None:
        batch_gate_settings(max_age_seconds=60, workers=5)
        metrics_on(True)
        tenant = f"t-{generate_nanoid(size=8)}"
        kept, lost = await seed_units(seed_work_unit, tenant, (600, 500))
        original = QueueManager.claim_work_units

        async def lose_one(
            self: QueueManager, db: AsyncSession, keys: Sequence[str]
        ) -> dict[str, str]:
            return await original(self, db, [key for key in keys if key != lost])

        with patch.object(QueueManager, "claim_work_units", lose_one):
            claimed = await QueueManager().get_and_claim_work_units()

        assert set(claimed) == {kept}
        assert sample("deriver_claims_total", tenant) == 1

    async def test_flag_off_records_nothing(
        self,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
        metrics_on: Callable[[bool], None],
    ) -> None:
        batch_gate_settings(max_age_seconds=60, workers=5)
        metrics_on(False)
        tenant = f"t-{generate_nanoid(size=8)}"
        await seed_units(seed_work_unit, tenant, (600,))

        claimed = await QueueManager().get_and_claim_work_units()

        assert len(claimed) == 1
        assert sample("deriver_claims_total", tenant) is None
        assert sample("deriver_claim_wait_seconds_count", tenant) is None


@pytest.mark.asyncio
class TestInflightUnits:
    async def test_the_poll_counts_live_claims_by_tenant_only_under_the_flag(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,
        batch_gate_settings: Callable[..., None],
        metrics_on: Callable[[bool], None],
    ) -> None:
        batch_gate_settings(max_age_seconds=60, workers=5)
        metrics_on(True)
        first, second = f"t-{generate_nanoid(size=8)}", f"t-{generate_nanoid(size=8)}"
        await seed_units(seed_work_unit, first, (600, 500))
        await seed_units(seed_work_unit, second, (400,))
        await QueueManager().get_and_claim_work_units()

        stats = await get_deriver_metrics(db_session)
        assert stats.inflight_units_by_tenant is not None
        assert stats.inflight_units_by_tenant[first] == 2
        assert stats.inflight_units_by_tenant[second] == 1

        metrics_on(False)
        assert (await get_deriver_metrics(db_session)).inflight_units_by_tenant is None

    async def test_a_tenant_whose_work_finished_loses_its_series(
        self, metrics_on: Callable[[bool], None]
    ) -> None:
        metrics_on(True)
        busy, done = f"t-{generate_nanoid(size=8)}", f"t-{generate_nanoid(size=8)}"

        prometheus_metrics.set_deriver_metrics(
            inflight_units_by_tenant={busy: 3, done: 1}
        )
        assert sample("deriver_inflight_units", done) == 1

        prometheus_metrics.set_deriver_metrics(inflight_units_by_tenant={busy: 2})
        assert sample("deriver_inflight_units", busy) == 2
        assert sample("deriver_inflight_units", done) is None

    async def test_flag_off_leaves_the_gauge_untouched(
        self, metrics_on: Callable[[bool], None]
    ) -> None:
        metrics_on(True)
        tenant = f"t-{generate_nanoid(size=8)}"
        prometheus_metrics.set_deriver_metrics(inflight_units_by_tenant={tenant: 1})

        # A pass with the flag off passes None: no rebuild, so nothing is cleared.
        prometheus_metrics.set_deriver_metrics(inflight_units_by_tenant=None)
        assert sample("deriver_inflight_units", tenant) == 1
