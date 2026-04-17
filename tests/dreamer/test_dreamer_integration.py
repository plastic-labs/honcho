"""Integration tests for the dream completion write.

Finding 3 (code-level) relocates `last_dream_at` from enqueue time to
dream-completion time (in `process_dream`). These tests exercise the real
Postgres JSONB merge via `tracked_db` to verify the write lands in the
collection's internal_metadata on successful dreams — and critically,
does NOT land on failures or exceptions.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dreamer.orchestrator import DreamResult, process_dream
from src.schemas import DreamType
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
        # ISO format sanity check: contains 'T' separator and either 'Z' or '+'
        assert "T" in dream_meta["last_dream_at"]

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
