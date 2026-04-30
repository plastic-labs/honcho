"""Regression tests for `enqueue_dream` metadata write shape.

Loop 4 (PR #573): `enqueue_dream` no longer touches collection.internal_metadata
at all. Both guard fields (last_dream_at and last_dream_document_count) are
written atomically in `process_dream` on successful completion — this preserves
the invariant that the baseline advances only when consolidation actually
happened, and prevents the in-flight stampede from false-advancing a guard.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src import schemas
from src.deriver.enqueue import enqueue_dream


class TestEnqueueDreamMetadataShape:
    @pytest.mark.asyncio
    async def test_enqueue_does_not_touch_collection_metadata(self):
        """`enqueue_dream` must not call update_collection_internal_metadata."""
        with (
            patch(
                "src.deriver.enqueue.crud.update_collection_internal_metadata",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "src.deriver.enqueue.crud.get_collection",
                new_callable=AsyncMock,
            ) as mock_get_collection,
            patch(
                "src.deriver.enqueue.tracked_db",
            ) as mock_db_ctx,
        ):
            mock_session = AsyncMock()
            mock_session.scalar = AsyncMock(return_value=False)
            mock_session.execute = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_db_ctx.return_value.__aenter__.return_value = mock_session

            await enqueue_dream(
                workspace_name="test_workspace",
                observer="alice",
                observed="bob",
                dream_type=schemas.DreamType.OMNI,
                session_name=None,
            )

            assert not mock_update.called, (
                "enqueue_dream must not write to collection.internal_metadata; "
                "guard fields advance atomically in process_dream on success."
            )
            assert not mock_get_collection.called, (
                "enqueue_dream must not need to load the collection — it no "
                "longer touches dream metadata."
            )
            assert (
                mock_session.execute.called
            ), "enqueue_dream must still insert the QueueItem row."
