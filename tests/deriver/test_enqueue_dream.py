"""Regression tests for `enqueue_dream` metadata write shape.

Finding 3 (code-level) moves the `last_dream_at` timestamp write from
enqueue time to dream-completion time (in `process_dream`). These tests
verify that `enqueue_dream` no longer writes `last_dream_at` and still
writes `last_dream_document_count`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import models, schemas
from src.deriver.enqueue import enqueue_dream


class TestEnqueueDreamMetadataShape:
    """Regression tests for Finding 3 code-level: `last_dream_at` relocation."""

    @pytest.mark.asyncio
    async def test_update_data_omits_last_dream_at(self):
        """`enqueue_dream` must NOT write `last_dream_at`.

        Moved to completion (in `process_dream`) so duplicate enqueues can't
        reset the 8-hour guard and failed dreams don't falsely advance it.
        """
        # Mock a Collection with empty dream metadata so the read-modify-write
        # in enqueue_dream has something to merge into.
        mock_collection = MagicMock(spec=models.Collection)
        mock_collection.internal_metadata = {}

        with (
            patch(
                "src.deriver.enqueue.crud.update_collection_internal_metadata",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "src.deriver.enqueue.crud.get_collection",
                new_callable=AsyncMock,
                return_value=mock_collection,
            ),
            # Short-circuit the dedup / insert paths so we only exercise the
            # metadata update call. db_session.scalar returns False for both
            # the in-progress and pending checks; db_session.execute is a no-op.
            patch(
                "src.deriver.enqueue.tracked_db",
            ) as mock_db_ctx,
        ):
            mock_session = AsyncMock()
            mock_session.scalar = AsyncMock(return_value=False)
            mock_session.execute = AsyncMock()
            mock_db_ctx.return_value.__aenter__.return_value = mock_session

            await enqueue_dream(
                workspace_name="test_workspace",
                observer="alice",
                observed="bob",
                dream_type=schemas.DreamType.OMNI,
                document_count=42,
                session_name=None,
            )

            assert (
                mock_update.called
            ), "update_collection_internal_metadata must be called"
            call_kwargs = mock_update.call_args.kwargs
            update_data = call_kwargs["update_data"]
            dream_metadata = update_data["dream"]

            assert (
                "last_dream_document_count" in dream_metadata
            ), "last_dream_document_count should still be written at enqueue"
            assert dream_metadata["last_dream_document_count"] == 42

            assert "last_dream_at" not in dream_metadata, (
                "last_dream_at must NOT be written at enqueue — it now writes "
                "at dream completion in process_dream (orchestrator.py)."
            )
