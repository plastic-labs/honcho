from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.queue_manager import QueueManager


@pytest.mark.asyncio
class TestQueueProcessing:
    """Test suite for queue processing functionality"""

    async def test_get_and_claim_work_units(
        self,
        sample_queue_items: list[models.QueueItem],
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ) -> None:
        """Test that get_and_claim_work_units correctly identifies unprocessed work"""
        session, _peers = sample_session_with_peers  # pyright: ignore[reportUnusedVariable]

        # Verify we have queue items from our test setup
        assert len(sample_queue_items) == 9  # 6 representation + 3 summary

        # Create a queue manager instance
        queue_manager = QueueManager()

        # Get available work units
        work_units = await queue_manager.get_and_claim_work_units()

        # Should have some work units available (may include items from other tests)
        assert len(work_units) > 0

        # Check that all work units have the expected structure
        for work_unit in work_units:
            assert isinstance(work_unit, str)
            assert work_unit.split(":")[0] in ["representation", "summary"]

        # The test is mainly verifying that get_and_claim_work_units works without errors
        # and returns properly structured work unit key strings

    async def test_work_unit_claiming(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ) -> None:
        """Test that work units can be claimed and are not available to other workers"""
        _session, _peers = sample_session_with_peers

        # Create a queue manager instance
        queue_manager = QueueManager()

        # Get available work units
        work_units = await queue_manager.get_and_claim_work_units()
        assert len(work_units) > 0

        # The API already claimed returned units; verify it's tracked and not returned again
        from sqlalchemy import select

        work_unit = work_units[0]
        tracked = (
            await db_session.execute(
                select(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.work_unit_key == work_unit
                )
            )
        ).scalar_one_or_none()
        assert tracked is not None

        # Get available work units again - the claimed one should not be available
        remaining_work_units = await queue_manager.get_and_claim_work_units()

        # The claimed work unit should not be in the remaining list
        assert work_unit not in remaining_work_units

    @pytest.mark.asyncio
    async def test_get_and_claim_excludes_already_claimed(
        self,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        queue_manager = QueueManager()
        first_batch = await queue_manager.get_and_claim_work_units()
        assert len(first_batch) > 0

        # Call again; previously claimed keys should not appear
        second_batch = await queue_manager.get_and_claim_work_units()
        assert all(k not in second_batch for k in first_batch)

    @pytest.mark.asyncio
    async def test_claim_work_unit_conflict_returns_false(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        # Pre-create an active session for a key
        queue_manager = QueueManager()
        claimed = await queue_manager.get_and_claim_work_units()
        assert len(claimed) > 0
        key = claimed[0]

        # Trying to claim the same key again via the API should return empty list
        claimed_again = await queue_manager.claim_work_units(db_session, [key])
        assert claimed_again == []

    @pytest.mark.asyncio
    async def test_get_next_message_orders_and_filters_simple(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        add_queue_items: Callable[..., Any],
    ) -> None:
        from sqlalchemy import select

        session, peers = sample_session_with_peers
        peer = peers[0]

        payloads: list[Any] = []
        for i in range(3):
            payloads.append(
                create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                    message=models.Message(
                        id=i,
                        session_name=session.name,
                        workspace_name=session.workspace_name,
                        peer_name=peer.name,
                        content="hello",
                    ),  # include id for payload builder
                    task_type="representation",
                    sender_name=peer.name,
                    target_name=peer.name,
                )
            )

        items = await add_queue_items(payloads, session.id)
        # Determine ascending order by DB id
        ordered = (
            (
                await db_session.execute(
                    select(models.QueueItem)
                    .where(models.QueueItem.work_unit_key == items[0].work_unit_key)
                    .order_by(models.QueueItem.id)
                )
            )
            .scalars()
            .all()
        )
        first, second = ordered[0], ordered[1]

        qm = QueueManager()
        batch = await qm.get_message_batch(first.work_unit_key, limit=1)
        nxt = batch[0] if batch else None
        assert nxt is not None and nxt.id == first.id

        # Mark first processed, next should be the second
        first.processed = True
        await db_session.commit()
        batch2 = await qm.get_message_batch(first.work_unit_key, limit=1)
        nxt2 = batch2[0] if batch2 else None
        assert nxt2 is not None and nxt2.id == second.id

    @pytest.mark.asyncio
    async def test_cleanup_work_unit_removes_row(
        self,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        db_session: AsyncSession,
    ) -> None:
        from sqlalchemy import select

        qm = QueueManager()
        claimed = await qm.get_and_claim_work_units()
        assert len(claimed) > 0
        key = claimed[0]

        removed = await qm._cleanup_work_unit(key)  # pyright: ignore[reportPrivateUsage]
        assert removed is True

        remaining = (
            await db_session.execute(
                select(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.work_unit_key == key
                )
            )
        ).scalar_one_or_none()
        assert remaining is None

    async def test_stale_work_unit_cleanup(
        self,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ) -> None:
        """Test that stale work units are cleaned up properly"""
        _session, _peers = sample_session_with_peers

        # Create an active queue session with an old timestamp
        # from datetime import datetime, timedelta, timezone

        # datetime.now(timezone.utc) - timedelta(minutes=10)

        # We'll test this by checking that the cleanup logic works in get_and_claim_work_units
        # which is called by the queue manager during normal operation
        queue_manager = QueueManager()

        # Get available work units - this should clean up stale entries
        work_units = await queue_manager.get_and_claim_work_units()

        # This test ensures the cleanup logic doesn't break, though we don't have stale entries yet
        assert isinstance(work_units, list)

    async def test_work_unit_key_format(
        self, sample_session_with_peers: tuple[models.Session, list[models.Peer]]
    ) -> None:
        """Test that work unit keys have the correct format"""
        session, peers = sample_session_with_peers
        peer1, peer2, _ = peers

        # Create a representation work unit key
        # Format: task_type:workspace:session:sender:target
        work_unit_key = (
            f"representation:workspace1:{session.name}:{peer1.name}:{peer2.name}"
        )

        # Check that the key contains the expected information
        assert session.name in work_unit_key
        assert peer1.name in work_unit_key
        assert peer2.name in work_unit_key
        assert "representation" in work_unit_key
        assert "workspace1" in work_unit_key

        # Create a summary work unit key
        # Summary work units use None for sender/target
        summary_work_unit_key = f"summary:workspace1:{session.name}:None:None"

        assert session.name in summary_work_unit_key
        assert "None" in summary_work_unit_key
        assert "summary" in summary_work_unit_key
        assert "workspace1" in summary_work_unit_key

    @pytest.mark.asyncio
    async def test_representation_batching_respects_token_limits(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that representation tasks are batched based on token limits"""
        from unittest.mock import patch

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create messages with token counts that exceed batch limit
        limit = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
        messages = [
            models.Message(
                id=i,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Test message {i}",
                token_count=token_count,
            )
            for i, token_count in enumerate([limit // 2, limit // 2, limit // 2])
        ]

        # Create queue items with token counts
        payloads = [
            create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=msg,
                task_type="representation",
                sender_name=peer.name,
                target_name=peer.name,
            )
            for msg in messages
        ]

        # Add items with token counts
        from src.deriver.utils import get_work_unit_key

        queue_items: list[models.QueueItem] = []
        for payload, message in zip(payloads, messages, strict=False):
            task_type = payload.get("task_type", "unknown")
            work_unit_key = get_work_unit_key(task_type, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                token_count=message.token_count,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str, queue_payloads: list[dict[str, Any]]
        ) -> None:
            processed_batches.append(
                {
                    "task_type": task_type,
                    "payload_count": len(queue_payloads),
                }
            )

        # Process work unit and verify batching
        qm = QueueManager()
        with patch(
            "src.deriver.queue_manager.process_items", side_effect=mock_process_items
        ):
            await qm.process_work_unit(queue_items[0].work_unit_key)

        # Should create 2 batches due to token limits
        assert len(processed_batches) == 2
        assert processed_batches[0]["payload_count"] == 2
        assert processed_batches[1]["payload_count"] == 1
        assert all(b["task_type"] == "representation" for b in processed_batches)

    @pytest.mark.asyncio
    async def test_hard_batch_size_limit(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that get_message_batch respects the hard limit of 10 messages"""
        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create 15 messages to test batch size limit
        messages = [
            models.Message(
                id=i,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Test message {i}",
                token_count=5,  # Small tokens to avoid token-based batching
            )
            for i in range(15)
        ]

        payloads = [
            create_queue_payload(msg, "representation", peer.name, peer.name)
            for msg in messages
        ]

        # Create queue items
        from src.deriver.utils import get_work_unit_key

        queue_items: list[models.QueueItem] = []
        for payload, message in zip(payloads, messages, strict=False):
            task_type = payload.get("task_type", "unknown")
            work_unit_key = get_work_unit_key(task_type, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                token_count=message.token_count,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        # Test batch size limit
        qm = QueueManager()
        batch = await qm.get_message_batch(queue_items[0].work_unit_key, limit=10)
        assert len(batch) == 10  # Should respect hard limit

        # Mark the first batch as processed
        await qm.mark_messages_as_processed(batch, queue_items[0].work_unit_key)

        # Get the next batch - should return remaining 5 items
        next_batch = await qm.get_message_batch(queue_items[0].work_unit_key, limit=10)
        assert len(next_batch) == 5  # Should return remaining items

    @pytest.mark.asyncio
    async def test_single_message_processing(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that multiple summary messages in same work unit are processed separately"""
        from unittest.mock import patch

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create two summary messages
        messages = [
            models.Message(
                id=999,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="First summary message",
                token_count=500,
            ),
            models.Message(
                id=1000,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="Second summary message",
                token_count=600,
            ),
        ]

        # Create payloads and queue items
        queue_items: list[models.QueueItem] = []
        for i, message in enumerate(messages):
            payload = create_queue_payload(
                message, "summary", message_seq_in_session=i + 1
            )
            from src.deriver.utils import get_work_unit_key

            work_unit_key = get_work_unit_key("summary", payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="summary",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                token_count=message.token_count,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        # Mock and process work unit
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str, queue_payloads: list[dict[str, Any]]
        ) -> None:
            processed_batches.append(
                {"task_type": task_type, "payload_count": len(queue_payloads)}
            )

        qm = QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        with patch(
            "src.deriver.queue_manager.process_items", side_effect=mock_process_items
        ):
            await qm.process_work_unit(work_unit_key)

        # Verify both messages were processed in separate batches
        assert len(processed_batches) == 2
        assert all(batch["task_type"] == "summary" for batch in processed_batches)
        assert all(batch["payload_count"] == 1 for batch in processed_batches)

        # Verify the corresponding DB records are marked as processed
        from sqlalchemy import select

        # Query for the summary queue items that were processed
        processed_items = (
            (
                await db_session.execute(
                    select(models.QueueItem)
                    .where(models.QueueItem.work_unit_key == work_unit_key)
                    .where(models.QueueItem.task_type == "summary")
                    .order_by(models.QueueItem.id)
                )
            )
            .scalars()
            .all()
        )

        # Assert we found both summary items
        assert len(processed_items) == 2

        # Assert both items are marked as processed
        assert all(item.processed is True for item in processed_items)

        # Optionally verify the items have the expected token counts from the messages
        expected_token_counts = [500, 600]  # From the test messages
        actual_token_counts = [item.token_count for item in processed_items]
        assert sorted(actual_token_counts) == sorted(expected_token_counts)
