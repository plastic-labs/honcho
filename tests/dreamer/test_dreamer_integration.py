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
from src.dreamer.dream_scheduler import DreamScheduler, set_dream_scheduler
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
    async def test_completion_preserves_last_dream_document_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Completion write must NOT drop sibling keys in the `dream` sub-object.

        `update_collection_internal_metadata` uses a top-level JSONB `||` merge,
        which replaces the whole `"dream"` key. Without a read-modify-write, the
        completion write (`last_dream_at`) would wipe `last_dream_document_count`
        that enqueue set — causing the next `check_and_schedule_dream` to read 0
        as the baseline and let a fresh dream trigger after the 8h guard expires
        even without any new explicit documents.
        """
        workspace, peer = sample_data
        # Pre-seed collection as if enqueue_dream already wrote the baseline.
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={"dream": {"last_dream_document_count": 42}},
        )
        db_session.add(collection)
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
        assert dream_meta.get("last_dream_document_count") == 42, (
            "last_dream_document_count from enqueue must be preserved across "
            "the completion write — top-level JSONB || would drop it without "
            "the read-modify-write in process_dream."
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


class TestEnqueueDreamPreservesSiblings:
    """Regression test for the symmetric sibling-drop bug in enqueue_dream.

    `update_collection_internal_metadata` uses a top-level JSONB `||` merge,
    so writing `{"dream": {"last_dream_document_count": N}}` without a prior
    read-modify-write would replace the whole `"dream"` subkey and drop
    `last_dream_at` (written by process_dream on the prior completion).

    Symmetric to the orchestrator fix in c8fe40a (and the
    test_completion_preserves_last_dream_document_count test above).
    """

    @pytest.mark.asyncio
    async def test_enqueue_preserves_last_dream_at(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """enqueue_dream must merge into existing dream metadata, not replace it.

        Pre-seeds `last_dream_at` (as if a prior dream completed), calls
        `enqueue_dream`, and asserts both:
        1. `last_dream_document_count` is written (the new baseline), and
        2. `last_dream_at` is preserved (the sibling key from the prior completion).
        """
        workspace, peer = sample_data
        prior_timestamp = "2026-04-17T12:00:00+00:00"
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={"dream": {"last_dream_at": prior_timestamp}},
        )
        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)

        await enqueue_dream(
            workspace_name=workspace.name,
            observer=collection.observer,
            observed=collection.observed,
            dream_type=DreamType.OMNI,
            document_count=77,
            session_name=None,
        )

        dream_meta = await _get_dream_metadata(db_session, collection)
        assert (
            dream_meta.get("last_dream_document_count") == 77
        ), "enqueue_dream must write last_dream_document_count as the new baseline"
        assert dream_meta.get("last_dream_at") == prior_timestamp, (
            "last_dream_at from a prior completion must be preserved across "
            "the enqueue write — top-level JSONB || would drop it without the "
            "read-modify-write in enqueue_dream."
        )


class TestExecuteDreamSessionFilter:
    """Regression test for the session lookup asymmetry in execute_dream.

    The baseline count query in execute_dream filters to `level == "explicit"`
    (symmetric with check_and_schedule_dream). The session_name lookup must
    filter the same way — otherwise the session is picked from a derived doc
    and can disagree with the document set the count is measuring, producing
    a dream scoped to a session that wasn't even in the triggering document
    cohort.
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
        returned — matching the explicit-only count query immediately below.
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

        # Capture kwargs passed to enqueue_dream (we don't want to actually
        # run it — just verify which session_name execute_dream picks).
        captured_kwargs: dict[str, Any] = {}

        async def capture_enqueue_dream(
            workspace_name: str,
            *,
            observer: str,
            observed: str,
            dream_type: Any,
            document_count: int,
            session_name: str,
        ) -> None:
            captured_kwargs.update(
                {
                    "workspace_name": workspace_name,
                    "observer": observer,
                    "observed": observed,
                    "dream_type": dream_type,
                    "document_count": document_count,
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
        # Sanity: the count must reflect explicit-only too (baseline symmetry).
        assert (
            captured_kwargs["document_count"] == 1
        ), "document_count must reflect explicit-only (1), not total (2)."
