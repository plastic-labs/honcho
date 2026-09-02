"""Tests for the outstanding-work value, the poller and the JSON route."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src import schemas
from src.backlog import (
    DeriverMetricsPoller,
    DeriverMetricsSnapshot,
    active_work_seconds,
    outstanding_work_seconds,
)
from src.routers import deriver_metrics


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


@pytest.mark.asyncio
class TestPoller:
    async def test_refresh_publishes_a_snapshot(self):
        stats = schemas.DeriverMetrics(eligible_work_units=2, pending_items=4)
        poller = DeriverMetricsPoller()

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=stats),
            ),
            patch("src.backlog.count_due_dreams", AsyncMock(return_value=3)),
        ):
            await poller.refresh()

        snapshot = poller.snapshot
        assert snapshot.measured_at is not None
        assert snapshot.stats.eligible_work_units == 2
        assert snapshot.dreams_due == 3
        assert snapshot.signal_seconds == active_work_seconds()

    async def test_dream_query_runs_on_its_own_spacing(self):
        """The dream query is the expensive one, so it must not run every pass."""
        stats = schemas.DeriverMetrics()
        poller = DeriverMetricsPoller()
        dream_count = AsyncMock(return_value=1)

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=stats),
            ),
            patch("src.backlog.count_due_dreams", dream_count),
        ):
            await poller.refresh()
            await poller.refresh()

        assert dream_count.await_count == 1
        assert poller.snapshot.dreams_due == 1

    async def test_a_failed_dream_query_is_retried_on_the_next_pass(self):
        """Advancing the deadline first would republish the old count for a whole interval."""
        stats = schemas.DeriverMetrics()
        poller = DeriverMetricsPoller()
        dream_count = AsyncMock(side_effect=[RuntimeError("db down"), 4])

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=stats),
            ),
            patch("src.backlog.count_due_dreams", dream_count),
        ):
            with pytest.raises(RuntimeError):
                await poller.refresh()
            await poller.refresh()

        assert dream_count.await_count == 2
        assert poller.snapshot.dreams_due == 4

    async def test_a_failed_pass_leaves_the_previous_snapshot_alone(self):
        """A half-finished pass must never be published as a measurement."""
        stats = schemas.DeriverMetrics(eligible_work_units=1)
        poller = DeriverMetricsPoller()

        with (
            patch(
                "src.backlog.crud.get_deriver_metrics",
                AsyncMock(return_value=stats),
            ),
            patch("src.backlog.count_due_dreams", AsyncMock(return_value=0)),
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
