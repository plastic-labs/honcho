"""Behavior of the deriver's fair scheduler: round-robin claiming across tenants.

The claim ranks each tenant's *eligible* work units oldest-first inside a
subquery and orders the outer claim by that rank, so a single poll takes every
tenant's oldest unit before it takes anybody's second. These tests pin the order
that produces and the properties that fall out of it: a flooding tenant cannot
starve a quiet one, the tenant-less reconciler lane is a bucket in the rotation
rather than an exception to it, and a deployment where every row is tenant-less
degenerates to plain oldest-first. They also pin the seams around that ordering
— eligibility placement (a claimed unit must not shadow its tenant's next one),
the billing-pause hook, the queue's lane CHECK, the pool-derived worker cap that
sets the claim's limit, and the enqueue stamp that fills the column the ranking
partitions on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import pytest
from sqlalchemy import ColumnElement
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.deriver import claim_excluded_tenants_clause
from src.deriver import queue_manager as queue_manager_module
from src.deriver.enqueue import _stamp_tenant_id
from src.deriver.queue_manager import QueueManager
from tests.deriver.test_work_unit_backlog import (  # noqa: F401 -- re-exported fixtures
    SeedWorkUnit,
    _read_backlog_keys,
    batch_gate_settings,
    seed_work_unit,
)

# The tenant ids used as backlog partitions. They are opaque strings on a
# no-FK attribution column, so a test may mint whatever reads clearly.
WHALE_TENANT = "t-whale"
FIRST_MINNOW_TENANT = "t-minnow-a"
SECOND_MINNOW_TENANT = "t-minnow-b"
PAUSED_TENANT = "t-paused"
LIVE_TENANT = "t-live"

# Every seeded unit is backdated past this flush window, so the batch gate is
# live (as it is in production) without any test depending on token counts.
AGE_FLUSH_SECONDS = 60


def representation_key(tenant_id: str | None, unit_name: str) -> str:
    """A representation work unit key, tenant-namespaced exactly as MULTI_TENANT writes it."""
    base_key = f"representation:workspace:{unit_name}:peer"
    return base_key if tenant_id is None else f"{tenant_id}:{base_key}"


@pytest.fixture
def recorded_claim_order(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The exact order the claim query offered work units in, one claim after another.

    ``get_and_claim_work_units`` returns a mapping, and a mapping cannot pin the
    ORDER BY that the whole fairness ranking exists to produce. ``claim_work_units``
    receives the candidate keys in query order, so wrapping it records the real
    ordering the ranked SELECT emitted — no mirrored query that could drift from
    the one under test.
    """
    offered_keys: list[str] = []
    original_claim_work_units = QueueManager.claim_work_units

    async def recording_claim_work_units(
        self: QueueManager, db: AsyncSession, work_unit_keys: Sequence[str]
    ) -> dict[str, str]:
        offered_keys.extend(work_unit_keys)
        return await original_claim_work_units(self, db, work_unit_keys)

    monkeypatch.setattr(QueueManager, "claim_work_units", recording_claim_work_units)
    return offered_keys


@pytest.fixture
def pool_derivation_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Pin every input to the pool-derived worker cap so no deployed default leaks in."""

    def _configure(
        *,
        configured_workers: int,
        workers_per_connection: float,
        pool_size: int,
        max_overflow: int,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "WORKERS", configured_workers)
        monkeypatch.setattr(
            settings.DERIVER, "WORKERS_PER_POOL_CONNECTION", workers_per_connection
        )
        monkeypatch.setattr(settings.DB, "POOL_SIZE", pool_size)
        monkeypatch.setattr(settings.DB, "MAX_OVERFLOW", max_overflow)

    return _configure


@pytest.mark.asyncio
class TestRoundRobinInterleavesTenants:
    """One poll rotates through the tenants before it revisits any of them."""

    async def test_every_tenant_is_offered_its_oldest_unit_before_any_tenant_repeats(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        recorded_claim_order: list[str],
    ) -> None:
        """A claim walks rank 1 for every tenant, then rank 2, oldest-first inside each round.

        The whale owns the four oldest units in the queue, so a plain oldest-first
        claim would hand it the first four slots. The ranking makes each round a
        rotation instead: the minnows' oldest units land ahead of the whale's second.
        """
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=7
        )
        whale_keys = [
            representation_key(WHALE_TENANT, f"whale-{index}") for index in range(4)
        ]
        first_minnow_keys = [
            representation_key(FIRST_MINNOW_TENANT, f"minnow-a-{index}")
            for index in range(2)
        ]
        second_minnow_key = representation_key(SECOND_MINNOW_TENANT, "minnow-b-0")

        for work_unit_key, age_seconds in zip(
            whale_keys, (600, 500, 400, 300), strict=True
        ):
            await seed_work_unit(
                work_unit_key,
                tenant_id=WHALE_TENANT,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )
        for work_unit_key, age_seconds in zip(
            first_minnow_keys, (250, 150), strict=True
        ):
            await seed_work_unit(
                work_unit_key,
                tenant_id=FIRST_MINNOW_TENANT,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )
        await seed_work_unit(
            second_minnow_key,
            tenant_id=SECOND_MINNOW_TENANT,
            token_counts=(1,),
            ages_seconds=(200,),
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert recorded_claim_order == [
            # Round 1: every tenant's oldest, ordered among themselves by age.
            whale_keys[0],
            first_minnow_keys[0],
            second_minnow_key,
            # Round 2: only the tenants that still have work.
            whale_keys[1],
            first_minnow_keys[1],
            # Rounds 3 and 4: the whale alone.
            whale_keys[2],
            whale_keys[3],
        ]
        assert recorded_claim_order.index(
            second_minnow_key
        ) < recorded_claim_order.index(whale_keys[1])
        # The contrast the ranking exists to create: ordering these same rows by
        # age alone spends the first four slots on the whale.
        assert recorded_claim_order != [
            *whale_keys,
            first_minnow_keys[0],
            second_minnow_key,
            first_minnow_keys[1],
        ]
        assert set(claimed_work_units) == {
            *whale_keys,
            *first_minnow_keys,
            second_minnow_key,
        }


@pytest.mark.asyncio
class TestAWhaleCannotStarveAMinnow:
    """Depth in one tenant buys it one slot per round, never the whole round."""

    async def test_a_flooding_whale_never_pushes_a_minnow_past_the_first_round(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        recorded_claim_order: list[str],
    ) -> None:
        """Twenty units all older than the minnow's one still leave it second in line."""
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=21
        )
        whale_keys = [
            representation_key(WHALE_TENANT, f"flood-{index:02d}")
            for index in range(20)
        ]
        for index, work_unit_key in enumerate(whale_keys):
            await seed_work_unit(
                work_unit_key,
                tenant_id=WHALE_TENANT,
                token_counts=(1,),
                ages_seconds=(1000 - index,),
            )
        minnow_key = representation_key(FIRST_MINNOW_TENANT, "quiet-0")
        await seed_work_unit(
            minnow_key,
            tenant_id=FIRST_MINNOW_TENANT,
            token_counts=(1,),
            ages_seconds=(100,),
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        tenant_count = 2
        assert recorded_claim_order.index(minnow_key) < tenant_count
        assert recorded_claim_order[:2] == [whale_keys[0], minnow_key]
        assert minnow_key in claimed_work_units

    async def test_a_minnow_shares_a_two_slot_claim_with_a_flooding_whale(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
    ) -> None:
        """Scarcity is where fairness matters: two slots, two tenants, one each."""
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=2
        )
        whale_keys = [
            representation_key(WHALE_TENANT, f"flood-{index:02d}")
            for index in range(20)
        ]
        for index, work_unit_key in enumerate(whale_keys):
            await seed_work_unit(
                work_unit_key,
                tenant_id=WHALE_TENANT,
                token_counts=(1,),
                ages_seconds=(1000 - index,),
            )
        minnow_key = representation_key(FIRST_MINNOW_TENANT, "quiet-0")
        await seed_work_unit(
            minnow_key,
            tenant_id=FIRST_MINNOW_TENANT,
            token_counts=(1,),
            ages_seconds=(100,),
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert set(claimed_work_units) == {whale_keys[0], minnow_key}


@pytest.mark.asyncio
class TestTheTenantlessBucketParticipates:
    """NULL tenant_id groups as one partition, so the reconciler lane rotates too."""

    async def test_a_reconciler_unit_claims_in_the_first_round_beside_the_tenants(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        recorded_claim_order: list[str],
    ) -> None:
        """The youngest unit in the queue still claims first round because it is its own bucket.

        Both tenants hold units older than the reconciler's. Oldest-first would
        spend all three slots on them; the NULL bucket's rank-1 row takes the third.
        """
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=3
        )
        alpha_keys = [
            representation_key(FIRST_MINNOW_TENANT, f"alpha-{index}")
            for index in range(2)
        ]
        beta_keys = [
            representation_key(SECOND_MINNOW_TENANT, f"beta-{index}")
            for index in range(2)
        ]
        for work_unit_key, age_seconds in zip(alpha_keys, (500, 450), strict=True):
            await seed_work_unit(
                work_unit_key,
                tenant_id=FIRST_MINNOW_TENANT,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )
        for work_unit_key, age_seconds in zip(beta_keys, (400, 350), strict=True):
            await seed_work_unit(
                work_unit_key,
                tenant_id=SECOND_MINNOW_TENANT,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )
        reconciler_key = "reconciler:sync_vectors"
        await seed_work_unit(
            reconciler_key,
            task_type="reconciler",
            tenant_id=None,
            token_counts=(0,),
            ages_seconds=(100,),
            with_messages=False,
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert recorded_claim_order == [alpha_keys[0], beta_keys[0], reconciler_key]
        assert set(claimed_work_units) == {alpha_keys[0], beta_keys[0], reconciler_key}


@pytest.mark.asyncio
class TestFlagOffDegeneratesToOldestFirst:
    """With every row tenant-less there is one partition, and rank order is age order."""

    async def test_successive_single_unit_claims_walk_the_backlog_oldest_first(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
    ) -> None:
        """A self-hosted deployment sees exactly the pre-fairness ordering."""
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=1
        )
        keys_by_age = [
            representation_key(None, "oldest-workspace"),
            representation_key(None, "older-workspace"),
            representation_key(None, "newer-workspace"),
            representation_key(None, "newest-workspace"),
        ]
        for work_unit_key, age_seconds in zip(
            keys_by_age, (400, 300, 200, 100), strict=True
        ):
            await seed_work_unit(
                work_unit_key,
                tenant_id=None,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )

        queue_manager = QueueManager()
        claims = [await queue_manager.get_and_claim_work_units() for _ in keys_by_age]

        assert [set(claim) for claim in claims] == [{key} for key in keys_by_age]

    async def test_one_wide_claim_returns_the_whole_backlog_oldest_first(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        recorded_claim_order: list[str],
    ) -> None:
        """However many logical tenants the keys imply, one NULL partition ranks them by age."""
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=4
        )
        keys_by_age = [
            representation_key(None, "oldest-workspace"),
            representation_key(None, "older-workspace"),
            representation_key(None, "newer-workspace"),
            representation_key(None, "newest-workspace"),
        ]
        for work_unit_key, age_seconds in zip(
            keys_by_age, (400, 300, 200, 100), strict=True
        ):
            await seed_work_unit(
                work_unit_key,
                tenant_id=None,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert recorded_claim_order == keys_by_age
        assert set(claimed_work_units) == set(keys_by_age)


@pytest.mark.asyncio
class TestAClaimedUnitDoesNotShadowItsTenantsNext:
    """Eligibility is filtered inside the ranking subquery, not after it."""

    async def test_a_tenants_second_unit_ranks_first_once_its_oldest_is_claimed(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        recorded_claim_order: list[str],
    ) -> None:
        """Ranking after the eligibility filter would demote this unit to round two.

        The claimed unit is the tenant's oldest. If the claim ranked every backlog
        row and only then dropped claimed ones, this tenant's remaining unit would
        keep rank 2 and lose the first slot to the other tenant's much younger
        rank-1 row — the whole tenant would fall a round behind for as long as one
        of its units was in flight.
        """
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=2
        )
        claimed_key = representation_key(WHALE_TENANT, "in-flight")
        shadowed_key = representation_key(WHALE_TENANT, "behind-the-claim")
        other_tenant_key = representation_key(LIVE_TENANT, "unrelated")
        await seed_work_unit(
            claimed_key,
            tenant_id=WHALE_TENANT,
            token_counts=(1,),
            ages_seconds=(900,),
        )
        await seed_work_unit(
            shadowed_key,
            tenant_id=WHALE_TENANT,
            token_counts=(1,),
            ages_seconds=(800,),
        )
        await seed_work_unit(
            other_tenant_key,
            tenant_id=LIVE_TENANT,
            token_counts=(1,),
            ages_seconds=(100,),
        )
        db_session.add(
            models.ActiveQueueSession(work_unit_key=claimed_key, tenant_id=WHALE_TENANT)
        )
        await db_session.commit()

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert recorded_claim_order == [shadowed_key, other_tenant_key]
        assert set(claimed_work_units) == {shadowed_key, other_tenant_key}


@pytest.mark.asyncio
class TestThePauseSeam:
    """The exclusion hook a billing pause will hang off, and what it does to a claim."""

    async def test_the_seam_excludes_nothing_today(
        self,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
    ) -> None:
        """With no paused-tenant source wired in, every bucket claims as usual."""
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=5
        )
        paused_key = representation_key(PAUSED_TENANT, "would-be-paused")
        live_key = representation_key(LIVE_TENANT, "always-live")
        reconciler_key = "reconciler:sync_vectors"
        await seed_work_unit(
            paused_key,
            tenant_id=PAUSED_TENANT,
            token_counts=(1,),
            ages_seconds=(300,),
        )
        await seed_work_unit(
            live_key, tenant_id=LIVE_TENANT, token_counts=(1,), ages_seconds=(200,)
        )
        await seed_work_unit(
            reconciler_key,
            task_type="reconciler",
            tenant_id=None,
            token_counts=(0,),
            ages_seconds=(100,),
            with_messages=False,
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert claim_excluded_tenants_clause() is None
        assert set(claimed_work_units) == {paused_key, live_key, reconciler_key}

    async def test_an_exclusion_clause_skips_the_paused_tenant_and_nobody_else(
        self,
        db_session: AsyncSession,
        seed_work_unit: SeedWorkUnit,  # noqa: F811
        batch_gate_settings: Callable[..., None],  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A clause wired into the seam filters before ranking, leaving the rest untouched.

        The paused tenant's units stay in the backlog — they are skipped, not
        drained — so unpausing needs nothing but dropping the clause again.
        """
        batch_gate_settings(
            target_tokens=512, max_age_seconds=AGE_FLUSH_SECONDS, workers=10
        )

        def exclude_the_paused_tenant() -> ColumnElement[bool]:
            return models.WorkUnitBacklog.tenant_id.is_(None) | (
                models.WorkUnitBacklog.tenant_id != PAUSED_TENANT
            )

        monkeypatch.setattr(
            queue_manager_module,
            "claim_excluded_tenants_clause",
            exclude_the_paused_tenant,
        )
        paused_keys = [
            representation_key(PAUSED_TENANT, f"paused-{index}") for index in range(2)
        ]
        live_key = representation_key(LIVE_TENANT, "still-running")
        reconciler_key = "reconciler:sync_vectors"
        for work_unit_key, age_seconds in zip(paused_keys, (900, 800), strict=True):
            await seed_work_unit(
                work_unit_key,
                tenant_id=PAUSED_TENANT,
                token_counts=(1,),
                ages_seconds=(age_seconds,),
            )
        await seed_work_unit(
            live_key, tenant_id=LIVE_TENANT, token_counts=(1,), ages_seconds=(200,)
        )
        await seed_work_unit(
            reconciler_key,
            task_type="reconciler",
            tenant_id=None,
            token_counts=(0,),
            ages_seconds=(100,),
            with_messages=False,
        )

        claimed_work_units = await QueueManager().get_and_claim_work_units()

        assert set(claimed_work_units) == {live_key, reconciler_key}
        assert await _read_backlog_keys(db_session) == {
            *paused_keys,
            live_key,
            reconciler_key,
        }


@pytest.mark.asyncio
class TestTheQueueLaneCheck:
    """A NULL workspace_name means the reconciler lane, and nothing else does."""

    async def test_a_reconciler_item_carrying_a_workspace_is_rejected(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """The reconciler is cross-tenant housekeeping; a workspace on it is a bug."""
        workspace, _peer = sample_data
        db_session.add(
            models.QueueItem(
                work_unit_key="reconciler:sync_vectors",
                task_type="reconciler",
                payload={},
                processed=False,
                workspace_name=workspace.name,
            )
        )

        with pytest.raises(IntegrityError):
            await db_session.flush()
        await db_session.rollback()

    async def test_a_representation_item_without_a_workspace_is_rejected(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Every tenant-scoped task type belongs to a workspace and must say which."""
        db_session.add(
            models.QueueItem(
                work_unit_key=representation_key(LIVE_TENANT, "no-workspace"),
                task_type="representation",
                payload={},
                processed=False,
                workspace_name=None,
                tenant_id=LIVE_TENANT,
            )
        )

        with pytest.raises(IntegrityError):
            await db_session.flush()
        await db_session.rollback()

    async def test_both_valid_lane_shapes_insert(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """The constraint is an equivalence, so it admits exactly two shapes."""
        workspace, _peer = sample_data
        db_session.add_all(
            [
                models.QueueItem(
                    work_unit_key="reconciler:sync_vectors",
                    task_type="reconciler",
                    payload={},
                    processed=False,
                    workspace_name=None,
                ),
                models.QueueItem(
                    work_unit_key=representation_key(LIVE_TENANT, "with-workspace"),
                    task_type="representation",
                    payload={},
                    processed=False,
                    workspace_name=workspace.name,
                    tenant_id=LIVE_TENANT,
                ),
            ]
        )

        await db_session.commit()

        assert await _read_backlog_keys(db_session) == {
            "reconciler:sync_vectors",
            representation_key(LIVE_TENANT, "with-workspace"),
        }


@pytest.mark.asyncio
class TestThePoolDerivedWorkerCap:
    """Effective concurrency is the configured worker count capped by pool headroom."""

    async def test_a_roomy_pool_leaves_the_configured_worker_count_alone(
        self, pool_derivation_settings: Callable[..., None]
    ) -> None:
        """Derivation only ever lowers, so headroom above the config changes nothing."""
        pool_derivation_settings(
            configured_workers=8,
            workers_per_connection=4.0,
            pool_size=10,
            max_overflow=20,
        )

        assert QueueManager().workers == 8

    async def test_a_tight_pool_lowers_the_worker_count_to_the_derived_cap(
        self, pool_derivation_settings: Callable[..., None]
    ) -> None:
        """0.5 workers per connection over 8 connections is 4 workers, not the configured 64."""
        pool_derivation_settings(
            configured_workers=64,
            workers_per_connection=0.5,
            pool_size=6,
            max_overflow=2,
        )

        assert QueueManager().workers == 4

    async def test_a_cap_that_floors_to_zero_still_leaves_one_worker(
        self, pool_derivation_settings: Callable[..., None]
    ) -> None:
        """A deriver with no workers would poll forever and process nothing."""
        pool_derivation_settings(
            configured_workers=8,
            workers_per_connection=0.25,
            pool_size=1,
            max_overflow=0,
        )

        assert QueueManager().workers == 1

    async def test_the_derived_cap_is_published_as_a_gauge(
        self,
        pool_derivation_settings: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The binding constraint is visible in metrics, not only in pool-timeout errors."""
        published_caps: list[int] = []
        monkeypatch.setattr(
            queue_manager_module.prometheus_metrics,
            "set_deriver_effective_worker_cap",
            published_caps.append,
        )
        pool_derivation_settings(
            configured_workers=64,
            workers_per_connection=0.5,
            pool_size=6,
            max_overflow=2,
        )

        queue_manager = QueueManager()

        assert published_caps == [queue_manager.workers] == [4]

    async def test_a_lowered_cap_warns_at_boot(
        self,
        pool_derivation_settings: Callable[..., None],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Silently running fewer workers than configured is the failure this avoids."""
        pool_derivation_settings(
            configured_workers=64,
            workers_per_connection=0.5,
            pool_size=6,
            max_overflow=2,
        )

        with caplog.at_level(logging.WARNING, logger=queue_manager_module.__name__):
            QueueManager()

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.name == queue_manager_module.__name__
        ]
        assert len(warnings) == 1
        assert "pool-derived cap 4" in warnings[0]

    async def test_an_unbinding_cap_stays_quiet(
        self,
        pool_derivation_settings: Callable[..., None],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The warning marks a real constraint, so the ordinary case must not raise it."""
        pool_derivation_settings(
            configured_workers=8,
            workers_per_connection=4.0,
            pool_size=10,
            max_overflow=20,
        )

        with caplog.at_level(logging.WARNING, logger=queue_manager_module.__name__):
            QueueManager()

        assert not [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.name == queue_manager_module.__name__
        ]


@pytest.mark.asyncio
class TestTheEnqueueTenantStamp:
    """The column the ranking partitions on is derived from the key at every insert."""

    async def test_an_unprefixed_key_is_rejected_under_multi_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record that reached the insert without a tenant would land in the NULL bucket."""
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        records = [
            {
                "work_unit_key": "representation:workspace:session:peer",
                "task_type": "representation",
            }
        ]

        with pytest.raises(ValueError, match="carries no tenant"):
            _stamp_tenant_id(records)

    async def test_a_prefixed_key_stamps_its_tenant_under_multi_tenant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Column and key prefix are the same value by construction, not by a second lookup."""
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        records = [
            {
                "work_unit_key": representation_key(LIVE_TENANT, "session"),
                "task_type": "representation",
            }
        ]

        stamped_records = _stamp_tenant_id(records)

        assert stamped_records[0]["tenant_id"] == LIVE_TENANT

    async def test_an_unprefixed_key_stamps_null_with_the_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Self-hosted deployments have one bucket, and NULL is what names it."""
        monkeypatch.setattr(settings, "MULTI_TENANT", False)
        records = [
            {
                "work_unit_key": "representation:workspace:session:peer",
                "task_type": "representation",
            }
        ]

        stamped_records = _stamp_tenant_id(records)

        assert stamped_records[0]["tenant_id"] is None
