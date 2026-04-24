"""Integration tests for the dream completion write.

Finding 3 (code-level) relocates `last_dream_at` from enqueue time to
dream-completion time (in `process_dream`). These tests exercise the real
Postgres JSONB merge via `tracked_db` to verify the write lands in the
collection's internal_metadata on successful dreams — and critically,
does NOT land on failures or exceptions.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.enqueue import enqueue_dream
from src.dreamer.dream_scheduler import (
    DreamScheduler,
    check_and_schedule_dream,
    set_dream_scheduler,
)
from src.dreamer.orchestrator import DreamResult, process_dream
from src.schemas import (
    DreamType,
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.queue_payload import DreamPayload


@pytest_asyncio.fixture
async def seeded_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> models.Collection:
    """Create a Collection with an empty dream metadata dict."""
    workspace, peer = sample_data
    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata={},
    )
    db_session.add(collection)
    await db_session.commit()
    await db_session.refresh(collection)
    return collection


def _make_dream_result() -> DreamResult:
    """Build a minimal non-null DreamResult for happy-path tests."""
    return DreamResult(
        run_id="test_run_01",
        specialists_run=["deduction", "induction"],
        deduction_success=True,
        induction_success=True,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=3,
        total_duration_ms=1234.5,
        input_tokens=100,
        output_tokens=50,
    )


async def _get_dream_metadata(
    db_session: AsyncSession, collection: models.Collection
) -> dict[str, Any]:
    """Re-fetch collection and return its internal_metadata['dream'] dict (or {})."""
    await db_session.refresh(collection)
    stmt = select(models.Collection).where(models.Collection.id == collection.id)
    refreshed = (await db_session.execute(stmt)).scalar_one()
    dream_meta: dict[str, Any] = refreshed.internal_metadata.get("dream", {})
    return dream_meta


class TestLastDreamAtCompletionWrite:
    """Regression tests for Finding 3: `last_dream_at` written at completion."""

    @pytest.mark.asyncio
    async def test_happy_path_writes_last_dream_at(
        self,
        db_session: AsyncSession,
        seeded_collection: models.Collection,
    ):
        """Non-null DreamResult → `last_dream_at` is set in internal_metadata."""
        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
        )

        with patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(return_value=_make_dream_result()),
        ):
            await process_dream(payload, seeded_collection.workspace_name)

        dream_meta = await _get_dream_metadata(db_session, seeded_collection)
        assert (
            "last_dream_at" in dream_meta
        ), "process_dream must write last_dream_at when run_dream returns a result"
        # Must be a tz-aware UTC ISO timestamp. A naive datetime.now().isoformat()
        # would pass a loose "T in string" check but corrupt the 8h guard math
        # against tz-aware now() comparisons downstream.
        parsed = datetime.fromisoformat(dream_meta["last_dream_at"])
        assert (
            parsed.tzinfo is not None
        ), f"last_dream_at must be timezone-aware, got {dream_meta['last_dream_at']!r}"
        assert parsed.utcoffset() == timedelta(
            0
        ), f"last_dream_at must be UTC, got offset {parsed.utcoffset()}"

    @pytest.mark.asyncio
    async def test_failure_path_leaves_last_dream_at_null(
        self,
        db_session: AsyncSession,
        seeded_collection: models.Collection,
    ):
        """run_dream returns None → `last_dream_at` stays absent.

        Lenient success criteria: the guard only advances on completion of a
        non-null DreamResult. Failed runs (None return) must not count.
        """
        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
        )

        with patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(return_value=None),
        ):
            await process_dream(payload, seeded_collection.workspace_name)

        dream_meta = await _get_dream_metadata(db_session, seeded_collection)
        assert "last_dream_at" not in dream_meta, (
            "last_dream_at must NOT be written when run_dream returns None "
            "(failed dream). The guard should not falsely advance."
        )

    @pytest.mark.asyncio
    async def test_completion_writes_guard_pair_atomically(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Completion writes last_dream_at AND last_dream_document_count together.

        Both guard fields advance only on successful consolidation, recomputed
        inside the row-locked RMW block so the pair stays coherent. Baseline
        reflects the actual explicit-doc count at completion, not a stale
        enqueue-time snapshot.
        """
        workspace, peer = sample_data
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)
        for i in range(7):
            db_session.add(
                models.Document(
                    content=f"explicit {i}",
                    level="explicit",
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                )
            )
        await db_session.commit()
        await db_session.refresh(collection)

        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer=collection.observer,
            observed=collection.observed,
        )

        with patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(return_value=_make_dream_result()),
        ):
            await process_dream(payload, collection.workspace_name)

        dream_meta = await _get_dream_metadata(db_session, collection)
        assert "last_dream_at" in dream_meta, "last_dream_at must be written"
        assert dream_meta.get("last_dream_document_count") == 7, (
            "last_dream_document_count must equal the current explicit-doc count "
            "at completion time; both guard fields advance together."
        )

    @pytest.mark.asyncio
    async def test_exception_path_leaves_last_dream_at_null(
        self,
        db_session: AsyncSession,
        seeded_collection: models.Collection,
    ):
        """run_dream raises → `last_dream_at` stays absent.

        `process_dream` catches exceptions (logs + marks task processed without
        re-raising) so the queue worker doesn't get stuck retrying. The guard
        write must not happen in the exception path — it's inside the
        `if result is not None` block, which never executes if an exception
        bypassed the assignment.
        """
        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer=seeded_collection.observer,
            observed=seeded_collection.observed,
        )

        with patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(side_effect=RuntimeError("simulated specialist crash")),
        ):
            # process_dream swallows exceptions internally; no re-raise expected
            await process_dream(payload, seeded_collection.workspace_name)

        dream_meta = await _get_dream_metadata(db_session, seeded_collection)
        assert "last_dream_at" not in dream_meta, (
            "last_dream_at must NOT be written when run_dream raises. "
            "process_dream swallows the exception but the guard write must "
            "not occur."
        )


class TestEnqueueDreamLeavesMetadataAlone:
    """enqueue_dream must not touch collection.internal_metadata["dream"].

    After the Loop 4 fix, the guard fields advance only on successful
    completion in process_dream. enqueue_dream should preserve whatever
    metadata is already on the collection (e.g. a prior completion's
    timestamp and baseline) and add nothing of its own.
    """

    @pytest.mark.asyncio
    async def test_enqueue_does_not_modify_dream_metadata(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        workspace, peer = sample_data
        prior_metadata = {
            "dream": {
                "last_dream_at": "2026-04-17T12:00:00+00:00",
                "last_dream_document_count": 99,
            }
        }
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata=prior_metadata,
        )
        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)

        await enqueue_dream(
            workspace_name=workspace.name,
            observer=collection.observer,
            observed=collection.observed,
            dream_type=DreamType.OMNI,
            session_name=None,
        )

        dream_meta = await _get_dream_metadata(db_session, collection)
        assert dream_meta == prior_metadata["dream"], (
            "enqueue_dream must leave dream metadata untouched; the guard fields "
            "advance only at completion."
        )


class TestExecuteDreamSessionFilter:
    """Regression test for the session lookup asymmetry in execute_dream.

    The session_name lookup filters to `level == "explicit"`, symmetric with
    check_and_schedule_dream's count query. Otherwise a derived doc could win
    ORDER BY created_at DESC and the dream would be scoped to a session that
    wasn't in the triggering document cohort.
    """

    @pytest.mark.asyncio
    async def test_session_name_picked_from_latest_explicit_doc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Latest explicit session wins even when a newer deductive doc exists.

        Seeds:
        - Session A (older): one explicit-level Document
        - Session B (newer): one deductive-level Document (dreamer output)

        Without the explicit filter on the session lookup, the newer deductive
        doc's session_name (B) would be returned. With the filter, A is
        returned — matching the explicit-only count query in
        check_and_schedule_dream.
        """
        workspace, peer = sample_data

        # Pre-create the collection so crud.get_collection inside enqueue_dream
        # finds something (process_dream's baseline write is not exercised here;
        # we're only asserting the kwargs passed to enqueue_dream).
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)

        # Two sessions: A (older), B (newer). Insert A first so its created_at
        # is strictly earlier than B's.
        session_a = models.Session(name="session_a", workspace_name=workspace.name)
        db_session.add(session_a)
        await db_session.commit()
        await db_session.refresh(session_a)

        session_b = models.Session(name="session_b", workspace_name=workspace.name)
        db_session.add(session_b)
        await db_session.commit()
        await db_session.refresh(session_b)

        # Older explicit doc in session A.
        explicit_doc = models.Document(
            content="explicit observation",
            level="explicit",
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            session_name=session_a.name,
        )
        db_session.add(explicit_doc)
        await db_session.commit()

        # Newer deductive doc in session B. Without the explicit filter on the
        # session lookup, this doc's session_name (B) would win on ORDER BY
        # created_at DESC — even though the count query ignores it.
        deductive_doc = models.Document(
            content="deductive observation",
            level="deductive",
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            session_name=session_b.name,
        )
        db_session.add(deductive_doc)
        await db_session.commit()

        captured_kwargs: dict[str, Any] = {}

        async def capture_enqueue_dream(
            workspace_name: str,
            *,
            observer: str,
            observed: str,
            dream_type: Any,
            session_name: str,
        ) -> None:
            captured_kwargs.update(
                {
                    "workspace_name": workspace_name,
                    "observer": observer,
                    "observed": observed,
                    "dream_type": dream_type,
                    "session_name": session_name,
                }
            )

        # Fresh scheduler instance; ENABLED patched so execute_dream runs.
        DreamScheduler.reset_singleton()
        scheduler = DreamScheduler()
        set_dream_scheduler(scheduler)

        try:
            with (
                patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED", True),
                patch(
                    "src.deriver.enqueue.enqueue_dream",
                    side_effect=capture_enqueue_dream,
                ),
                patch(
                    "src.utils.config_helpers.get_configuration",
                    return_value=ResolvedConfiguration(
                        reasoning=ResolvedReasoningConfiguration(enabled=True),
                        peer_card=ResolvedPeerCardConfiguration(use=True, create=True),
                        summary=ResolvedSummaryConfiguration(
                            enabled=True,
                            messages_per_short_summary=10,
                            messages_per_long_summary=20,
                        ),
                        dream=ResolvedDreamConfiguration(enabled=True),
                    ),
                ),
            ):
                await scheduler.execute_dream(
                    workspace.name,
                    DreamType.OMNI,
                    observer=peer.name,
                    observed=peer.name,
                )
        finally:
            DreamScheduler.reset_singleton()

        assert captured_kwargs, (
            "enqueue_dream must be called — execute_dream returned early, "
            "likely because the session lookup returned no rows (check that "
            "the explicit filter matches at least one doc in the fixture)."
        )
        assert captured_kwargs["session_name"] == session_a.name, (
            f"Session lookup must filter to level=='explicit' to match the "
            f"baseline count query. Got session_name="
            f"{captured_kwargs['session_name']!r}, expected {session_a.name!r} "
            f"(the older session with the only explicit doc). Picking "
            f"{session_b.name!r} means the session came from a derived doc "
            f"that the count query ignores — the dream would be scoped to a "
            f"session that wasn't in the triggering cohort."
        )


class TestGuardPairCoherence:
    """Loop 4 coherence tests for the invariant preserved by the atomic pair
    write and the in-flight stampede defense.

    Invariant: From the moment a dream is scheduled until it completes or
    fails, no second dream may be enqueued for the same
    (workspace, observer, observed) — and the baseline count advances only
    when consolidation actually happened.
    """

    @pytest_asyncio.fixture
    async def _scheduler(self):
        DreamScheduler.reset_singleton()
        scheduler = DreamScheduler()
        set_dream_scheduler(scheduler)
        with (
            patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED", True),
            patch("src.dreamer.dream_scheduler.settings.DREAM.DOCUMENT_THRESHOLD", 50),
            patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED_TYPES", ["omni"]),
        ):
            yield scheduler
        DreamScheduler.reset_singleton()

    @pytest.mark.asyncio
    async def test_pending_queue_item_blocks_second_schedule(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        _scheduler: DreamScheduler,
    ):
        """In-flight window: pending QueueItem must block a second schedule.

        Walks the stampede timeline: enqueue fires a dream, more explicit
        docs arrive past the threshold again, but check_and_schedule_dream
        sees the pending queue row and returns False — no second QueueItem.
        """
        workspace, peer = sample_data
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)
        for i in range(50):
            db_session.add(
                models.Document(
                    content=f"explicit {i}",
                    level="explicit",
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                )
            )
        await db_session.commit()
        await db_session.refresh(collection)

        await enqueue_dream(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            dream_type=DreamType.OMNI,
            session_name=None,
        )

        pending_q = select(models.QueueItem).where(
            models.QueueItem.task_type == "dream",
            models.QueueItem.processed == False,  # noqa: E712
            models.QueueItem.workspace_name == workspace.name,
        )
        pending_rows = (await db_session.execute(pending_q)).scalars().all()
        assert len(pending_rows) == 1, (
            "enqueue_dream must insert exactly one pending dream QueueItem "
            "(baseline for the stampede test)."
        )

        for i in range(50, 100):
            db_session.add(
                models.Document(
                    content=f"explicit {i}",
                    level="explicit",
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                )
            )
        await db_session.commit()
        await db_session.refresh(collection)

        scheduled = await check_and_schedule_dream(db_session, collection)

        assert scheduled is False, (
            "check_and_schedule_dream must return False while a dream is "
            "pending in the queue — the in-flight window must not admit a "
            "second schedule regardless of how many explicit docs arrive."
        )
        pending_rows_after = (await db_session.execute(pending_q)).scalars().all()
        assert len(pending_rows_after) == 1, (
            "No second QueueItem may be inserted while the first is pending. "
            f"Found {len(pending_rows_after)} pending rows."
        )

    @pytest.mark.asyncio
    async def test_silent_failure_allows_retry_on_same_corpus(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        _scheduler: DreamScheduler,
    ):
        """Failed dream (run_dream returns None) leaves both guard fields
        untouched, so check_and_schedule_dream re-schedules on the same
        corpus instead of silently consuming the baseline.
        """
        workspace, peer = sample_data
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)
        for i in range(50):
            db_session.add(
                models.Document(
                    content=f"explicit {i}",
                    level="explicit",
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                )
            )
        await db_session.commit()
        await db_session.refresh(collection)

        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer=peer.name,
            observed=peer.name,
        )
        with patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(return_value=None),
        ):
            await process_dream(payload, workspace.name)

        dream_meta = await _get_dream_metadata(db_session, collection)
        assert dream_meta.get("last_dream_document_count", 0) == 0, (
            "Failed dream must not advance last_dream_document_count; "
            "pre-Loop-4 the baseline was consumed at enqueue time and a "
            "silent failure would lock out retries on the same corpus."
        )
        assert (
            "last_dream_at" not in dream_meta
        ), "Failed dream must not advance last_dream_at either."

        with patch.object(
            _scheduler, "schedule_dream", new_callable=AsyncMock
        ) as mock_schedule:
            scheduled = await check_and_schedule_dream(db_session, collection)

        assert scheduled is True, (
            "After a silent failure both guards should still allow the "
            "same-corpus retry — 50 explicit docs ≥ threshold, no prior "
            "last_dream_at, no pending queue item."
        )
        assert mock_schedule.called, "schedule_dream must be invoked on the retry path."
