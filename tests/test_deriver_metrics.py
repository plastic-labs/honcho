"""Tests for the outstanding-work value, the poller and the JSON route."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src import schemas
from src.backlog import (
    DeriverMetricsPoller,
    DeriverMetricsSnapshot,
    DreamDuePoller,
    active_work_seconds,
    outstanding_work_seconds,
)
from src.config import settings
from src.dreamer.dream_due import DueDream
from src.models import DEFAULT_TENANT_ID
from src.routers import deriver_metrics


def _due_dreams(count: int) -> list[DueDream]:
    return [
        DueDream(
            workspace_name=f"workspace-{i}",
            observer=f"peer-{i}",
            observed=f"peer-{i}",
            dream_type=schemas.DreamType.OMNI,
            session_name=f"session-{i}",
            tenant_id=DEFAULT_TENANT_ID,
        )
        for i in range(count)
    ]


class TestScaleSignal:
    def test_nothing_outstanding_reads_zero(self):
        assert outstanding_work_seconds(schemas.DeriverMetrics(), dreams_due=0) == 0.0

    def test_claimable_work_reports_the_active_value(self):
        stats = schemas.DeriverMetrics(eligible_work_units=1)

        assert outstanding_work_seconds(stats, dreams_due=0) == active_work_seconds()

    def test_work_in_flight_still_reports_the_active_value(self):
        """A row claimed a moment ago has a small age and would read as idle."""
        stats = schemas.DeriverMetrics(
            claimed_work_units=1, pending_items=1, oldest_pending_age_seconds=2.0
        )

        assert outstanding_work_seconds(stats, dreams_due=0) == active_work_seconds()

    def test_waiting_batch_reports_its_real_age(self):
        """The real age is what tells a caller how close the flush is."""
        stats = schemas.DeriverMetrics(
            pending_items=3, oldest_pending_age_seconds=1234.0
        )

        assert outstanding_work_seconds(stats, dreams_due=0) == 1234.0

    def test_embeddings_due_an_attempt_report_the_active_value(self):
        stats = schemas.DeriverMetrics(embeddings_pending=5, embeddings_pending_due=5)

        assert outstanding_work_seconds(stats, dreams_due=0) == active_work_seconds()

    def test_embeddings_inside_their_retry_wait_do_not(self):
        """Otherwise one permanently failing row holds the value up for hours."""
        stats = schemas.DeriverMetrics(embeddings_pending=5)

        assert outstanding_work_seconds(stats, dreams_due=0) == 0.0

    def test_a_due_dream_reports_the_active_value(self):
        assert (
            outstanding_work_seconds(schemas.DeriverMetrics(), dreams_due=1)
            == active_work_seconds()
        )

    def test_active_value_is_positive(self):
        assert active_work_seconds() > 0


class _FakeLease:
    """Stands in for AdvisoryLease: holds whatever the test says."""

    def __init__(self, *, held: bool) -> None:
        self.held: bool = held
        self.released: bool = False

    async def ensure(self) -> bool:
        return self.held

    async def release(self) -> None:
        self.released = True
        self.held = False


def _dream_poller(*, lease: _FakeLease | None) -> DreamDuePoller:
    poller = DreamDuePoller()
    poller._lease = lease  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return poller


@pytest.mark.asyncio
class TestPoller:
    async def test_refresh_publishes_a_snapshot(self):
        stats = schemas.DeriverMetrics(eligible_work_units=2, pending_items=4)
        poller = DeriverMetricsPoller()
        poller.dream_due_poller._due_dreams = _due_dreams(3)  # pyright: ignore[reportPrivateUsage]

        with patch(
            "src.backlog.crud.get_deriver_metrics",
            AsyncMock(return_value=stats),
        ):
            await poller.refresh()

        snapshot = poller.snapshot
        assert snapshot.measured_at is not None
        assert snapshot.stats.eligible_work_units == 2
        assert snapshot.dreams_due == 3
        assert snapshot.signal_seconds == active_work_seconds()

    async def test_refresh_never_runs_the_dream_query(self):
        """The dream poll runs on its own task, so a slow one can't stall this pass."""
        poller = DeriverMetricsPoller()
        dream_query = AsyncMock(return_value=_due_dreams(1))

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=schemas.DeriverMetrics()),
            ),
            patch("src.backlog.list_due_dreams", dream_query),
        ):
            await poller.refresh()
            await poller.refresh()

        assert dream_query.await_count == 0

    async def test_metrics_keep_advancing_while_a_dream_poll_is_stuck(self):
        """A dream poll that hangs must not freeze deriver_metrics_last_success (F25)."""
        poller = DeriverMetricsPoller()
        dream_poller = _dream_poller(lease=None)
        poller.dream_due_poller = dream_poller
        release = asyncio.Event()

        async def stuck_poll(*_args: Any, **_kwargs: Any) -> list[DueDream]:
            await release.wait()
            return []

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=schemas.DeriverMetrics()),
            ),
            patch("src.backlog.list_due_dreams", stuck_poll),
        ):
            stuck = asyncio.create_task(dream_poller.poll())
            await asyncio.sleep(0)
            await poller.refresh()
            first = poller.snapshot.measured_at
            await asyncio.sleep(0.01)
            await poller.refresh()
            assert not stuck.done()
            release.set()
            await stuck

        assert first is not None
        assert poller.snapshot.measured_at is not None
        assert poller.snapshot.measured_at > first

    async def test_a_failed_pass_leaves_the_previous_snapshot_alone(self):
        """A half-finished pass must never be published as a measurement."""
        stats = schemas.DeriverMetrics(eligible_work_units=1)
        poller = DeriverMetricsPoller()

        with patch(
            "src.backlog.crud.get_deriver_metrics",
            AsyncMock(return_value=stats),
        ):
            await poller.refresh()

        first = poller.snapshot

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(side_effect=RuntimeError("db down")),
            ),
            pytest.raises(RuntimeError),
        ):
            await poller.refresh()

        assert poller.snapshot is first


@pytest.mark.asyncio
class TestDreamDuePoller:
    async def test_the_lease_holder_polls_with_a_sweep_slot_and_enqueues(self):
        due = _due_dreams(2)
        poller = _dream_poller(lease=_FakeLease(held=True))
        query = AsyncMock(return_value=due)
        enqueue = AsyncMock()

        with (
            patch("src.backlog.list_due_dreams", query),
            patch("src.backlog.enqueue_dream", enqueue),
        ):
            await poller.poll()

        assert poller.dreams_due == 2
        assert query.await_args is not None
        assert query.await_args.kwargs["sweep_slot"] is not None
        assert enqueue.await_count == 2
        first_call = enqueue.await_args_list[0]
        assert first_call.args == (due[0].workspace_name,)
        assert first_call.kwargs["observer"] == due[0].observer
        assert first_call.kwargs["observed"] == due[0].observed
        assert first_call.kwargs["dream_type"] == due[0].dream_type
        assert first_call.kwargs["session_name"] == due[0].session_name

    async def test_a_replica_without_the_lease_polls_nothing_and_reports_zero(self):
        poller = _dream_poller(lease=_FakeLease(held=False))
        poller._due_dreams = _due_dreams(3)  # pyright: ignore[reportPrivateUsage]
        query = AsyncMock(return_value=_due_dreams(1))
        enqueue = AsyncMock()

        with (
            patch("src.backlog.list_due_dreams", query),
            patch("src.backlog.enqueue_dream", enqueue),
        ):
            await poller.poll()

        assert query.await_count == 0
        assert enqueue.await_count == 0
        assert poller.dreams_due == 0

    async def test_deriver_scheduler_measures_without_a_sweep_and_enqueues_nothing(
        self,
    ):
        poller = _dream_poller(lease=None)
        query = AsyncMock(return_value=_due_dreams(1))
        enqueue = AsyncMock()

        with (
            patch("src.backlog.list_due_dreams", query),
            patch("src.backlog.enqueue_dream", enqueue),
        ):
            await poller.poll()

        assert enqueue.await_count == 0
        assert poller.dreams_due == 1
        assert query.await_args is not None
        assert query.await_args.kwargs["sweep_slot"] is None

    async def test_the_lease_is_only_built_for_the_api_scheduler(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.DERIVER, "SCHEDULER", "deriver")
        assert DreamDuePoller()._lease is None  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(settings.DERIVER, "SCHEDULER", "api")
        assert DreamDuePoller()._lease is not None  # pyright: ignore[reportPrivateUsage]

    async def test_one_failed_enqueue_does_not_stop_the_rest(self):
        poller = _dream_poller(lease=_FakeLease(held=True))
        enqueue = AsyncMock(side_effect=[RuntimeError("db down"), None])

        with (
            patch(
                "src.backlog.list_due_dreams", AsyncMock(return_value=_due_dreams(2))
            ),
            patch("src.backlog.enqueue_dream", enqueue),
        ):
            await poller.poll()

        assert enqueue.await_count == 2

    async def test_shutdown_releases_the_lease(self):
        lease = _FakeLease(held=True)
        poller = _dream_poller(lease=lease)

        await poller.shutdown()

        assert lease.released


@pytest.mark.asyncio
class TestDeriverMetricsRoute:
    async def test_serves_the_cached_snapshot(self):
        poller = DeriverMetricsPoller()
        poller._snapshot = DeriverMetricsSnapshot(  # pyright: ignore[reportPrivateUsage]
            signal_seconds=1800.0,
            dreams_due=1,
            stats=schemas.DeriverMetrics(eligible_work_units=2, pending_items=5),
            measured_at=time.time(),
        )
        deriver_metrics.set_deriver_metrics_poller(poller)
        try:
            body = await deriver_metrics.get_deriver_metrics_response()
        finally:
            deriver_metrics.set_deriver_metrics_poller(None)

        assert body["outstanding_work_seconds"] == 1800.0
        assert body["eligible_work_units"] == 2
        assert body["pending_items"] == 5
        assert body["dreams_due"] == 1

    async def test_errors_before_the_first_pass(self):
        """A 503 tells the caller there is no measurement; a 0 would be a lie."""
        deriver_metrics.set_deriver_metrics_poller(DeriverMetricsPoller())
        try:
            with pytest.raises(HTTPException) as excinfo:
                await deriver_metrics.get_deriver_metrics_response()
        finally:
            deriver_metrics.set_deriver_metrics_poller(None)

        assert excinfo.value.status_code == 503

    async def test_serves_an_old_snapshot_with_its_age(self):
        """The caller decides what is too old, from measurement_age_seconds."""
        poller = DeriverMetricsPoller()
        poller._snapshot = DeriverMetricsSnapshot(  # pyright: ignore[reportPrivateUsage]
            signal_seconds=7.0,
            measured_at=time.time() - 3600,
        )
        deriver_metrics.set_deriver_metrics_poller(poller)
        try:
            body = await deriver_metrics.get_deriver_metrics_response()
        finally:
            deriver_metrics.set_deriver_metrics_poller(None)

        assert body["outstanding_work_seconds"] == 7.0
        assert body["measurement_age_seconds"] >= 3600

    async def test_errors_when_no_poller_is_registered(self):
        deriver_metrics.set_deriver_metrics_poller(None)

        with pytest.raises(HTTPException) as excinfo:
            await deriver_metrics.get_deriver_metrics_response()

        assert excinfo.value.status_code == 503


@pytest.mark.asyncio
class TestLifespanScheduler:
    async def _run_lifespan(self) -> AsyncMock:
        from src import main as main_module

        scheduler = AsyncMock()

        with (
            patch.object(main_module, "initialize_telemetry_async", AsyncMock()),
            patch.object(main_module, "register_db_pool_collector"),
            patch.object(main_module, "register_db_query_instrumentation"),
            patch.object(main_module, "register_db_connection_instrumentation"),
            patch.object(main_module, "validate_embedding_schema", AsyncMock()),
            patch.object(main_module, "validate_queue_item_batches", AsyncMock()),
            patch.object(main_module, "init_cache", AsyncMock()),
            patch.object(main_module, "close_cache", AsyncMock()),
            patch.object(main_module, "shutdown_telemetry", AsyncMock()),
            patch.object(main_module, "engine", AsyncMock()),
            patch.object(main_module, "DeriverMetricsPoller", return_value=AsyncMock()),
            patch.object(main_module, "ReconcilerScheduler", return_value=scheduler),
            patch.object(main_module, "set_reconciler_scheduler"),
            patch(
                "src.vector_store.close_external_vector_store",
                AsyncMock(),
            ),
        ):
            async with main_module.lifespan(main_module.app):
                pass

        return scheduler

    async def test_api_scheduler_runs_the_reconciler(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.DERIVER, "SCHEDULER", "api")

        scheduler = await self._run_lifespan()

        assert scheduler.start.await_count == 1
        assert scheduler.shutdown.await_count == 1

    async def test_deriver_scheduler_leaves_the_reconciler_off(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings.DERIVER, "SCHEDULER", "deriver")

        scheduler = await self._run_lifespan()

        assert scheduler.start.await_count == 0
        assert scheduler.shutdown.await_count == 0
