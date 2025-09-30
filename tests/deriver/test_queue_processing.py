from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.queue_manager import QueueManager, WorkerOwnership
from src.deriver.utils import get_work_unit_key


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

        work_unit = next(iter(work_units.keys()))
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
        key = list(claimed.keys())[0]

        # Trying to claim the same key again via the API should return empty dict
        claimed_again = await queue_manager.claim_work_units(db_session, [key])
        assert claimed_again == {}

    @pytest.mark.asyncio
    async def test_get_next_message_orders_and_filters_simple(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        add_queue_items: Callable[..., Any],
    ) -> None:
        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create and save messages to the database first
        messages: list[models.Message] = []
        for _ in range(3):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="hello",
                token_count=10,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        payloads: list[Any] = []
        for message in messages:
            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=message,
                task_type="representation",
                sender_name=peer.name,
                target_name=peer.name,
            )
            payloads.append(payload)

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

        aqs = models.ActiveQueueSession(
            work_unit_key=first.work_unit_key,
        )
        db_session.add(aqs)
        await db_session.commit()
        await db_session.refresh(aqs)

        batch = await qm.get_message_batch(
            task_type="representation",
            work_unit_key=first.work_unit_key,
            aqs_id=aqs.id,
        )
        nxt = batch[0] if batch else None
        assert nxt is not None and nxt.id == first.id

        # Mark first processed, next should be the second
        first.processed = True
        await db_session.commit()
        batch2 = await qm.get_message_batch(
            task_type="representation",
            work_unit_key=first.work_unit_key,
            aqs_id=aqs.id,
        )
        nxt2 = batch2[0] if batch2 else None
        assert nxt2 is not None and nxt2.id == second.id

    @pytest.mark.asyncio
    async def test_cleanup_work_unit_removes_row(
        self,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
        db_session: AsyncSession,
    ) -> None:
        qm = QueueManager()
        claimed = await qm.get_and_claim_work_units()
        assert len(claimed) > 0
        key = list(claimed.keys())[0]
        aqs_id = claimed[key]

        removed = await qm._cleanup_work_unit(aqs_id, key)  # pyright: ignore[reportPrivateUsage]
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
        assert isinstance(work_units, dict)

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

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create messages with token counts that exceed batch limit
        limit = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
        token_counts = [limit // 2, limit // 2, limit // 2]

        # Create and save messages to the database first
        messages: list[models.Message] = []
        for i, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Test message {i}",
                token_count=token_count,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

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

        queue_items: list[models.QueueItem] = []
        for payload in payloads:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = get_work_unit_key(task_type, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str,
            queue_payloads: list[dict[str, Any]],
            sender_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
            target_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            processed_batches.append(
                {
                    "task_type": task_type,
                    "payload_count": len(queue_payloads),
                }
            )

        # Process work unit and verify batching
        qm = QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        worker_id = "test_worker"

        # Manually claim and assign ownership
        claimed_units = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed_units[work_unit_key]
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=aqs_id
        )

        with patch(
            "src.deriver.queue_manager.process_items",
            side_effect=mock_process_items,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Should create 2 batches due to token limits
        assert len(processed_batches) == 2
        assert processed_batches[0]["payload_count"] == 2
        assert processed_batches[1]["payload_count"] == 1
        assert all(b["task_type"] == "representation" for b in processed_batches)

    @pytest.mark.asyncio
    async def test_token_batching_filters_by_work_unit(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that messages are batched chronologically across all peers, then filtered by work unit"""

        session, peers = sample_session_with_peers
        alice, bob, steve = peers[0], peers[1], peers[2]

        # Create messages from different peers with specific token counts
        # Message sequence: alice(250), bob(400), steve(300), alice(500), bob(500), alice(40), steve(100)
        # Token limit: 2000 - should batch messages 1-6 (1990 tokens)
        messages_data = [
            (alice, 250),
            (bob, 400),
            (steve, 300),
            (alice, 500),
            (bob, 500),
            (alice, 40),
            (steve, 100),
        ]

        messages: list[models.Message] = []
        for peer, token_count in messages_data:
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message from {peer.name}",
                token_count=token_count,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        # Create queue items for alice's work unit (representation)

        alice_queue_items: list[models.QueueItem] = []
        bob_queue_items: list[models.QueueItem] = []
        steve_queue_items: list[models.QueueItem] = []

        for i, message in enumerate(messages):
            peer = messages_data[i][0]
            target = alice  # All observing alice for simplicity

            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=message,
                task_type="representation",
                sender_name=peer.name,
                target_name=target.name,
            )
            work_unit_key = get_work_unit_key("representation", payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)

            # Track items by peer
            if peer == alice:
                alice_queue_items.append(queue_item)
            elif peer == bob:
                bob_queue_items.append(queue_item)
            else:
                steve_queue_items.append(queue_item)

        await db_session.commit()
        for item in alice_queue_items + bob_queue_items + steve_queue_items:
            await db_session.refresh(item)

        qm = QueueManager()

        # Mock the token limit to 2000 for this test
        with patch.object(settings.DERIVER, "REPRESENTATION_BATCH_MAX_TOKENS", 2000):
            # Test alice's work unit
            alice_work_unit_key = alice_queue_items[0].work_unit_key
            alice_aqs = models.ActiveQueueSession(work_unit_key=alice_work_unit_key)
            db_session.add(alice_aqs)
            await db_session.commit()
            await db_session.refresh(alice_aqs)

            alice_batch = await qm.get_message_batch(
                task_type="representation",
                work_unit_key=alice_work_unit_key,
                aqs_id=alice_aqs.id,
            )

            # All work units should get ALL 6 messages in the chronological batch (messages 1-6)
            # Message 7 is outside the token limit batch
            assert len(alice_batch) == 6
            alice_message_ids = {msg.payload["message_id"] for msg in alice_batch}
            expected_batch_ids = {
                messages[0].id,  # alice(250)
                messages[1].id,  # bob(400)
                messages[2].id,  # steve(300)
                messages[3].id,  # alice(500)
                messages[4].id,  # bob(500)
                messages[5].id,  # alice(40)
            }
            assert alice_message_ids == expected_batch_ids

            # Test bob's work unit - should get same batch
            bob_work_unit_key = bob_queue_items[0].work_unit_key
            bob_aqs = models.ActiveQueueSession(work_unit_key=bob_work_unit_key)
            db_session.add(bob_aqs)
            await db_session.commit()
            await db_session.refresh(bob_aqs)

            bob_batch = await qm.get_message_batch(
                task_type="representation",
                work_unit_key=bob_work_unit_key,
                aqs_id=bob_aqs.id,
            )

            # Bob should also get all 6 messages for context
            assert len(bob_batch) == 6
            bob_message_ids = {msg.payload["message_id"] for msg in bob_batch}
            assert bob_message_ids == expected_batch_ids

            # Test steve's work unit - should get same batch
            steve_work_unit_key = steve_queue_items[0].work_unit_key
            steve_aqs = models.ActiveQueueSession(work_unit_key=steve_work_unit_key)
            db_session.add(steve_aqs)
            await db_session.commit()
            await db_session.refresh(steve_aqs)

            steve_batch = await qm.get_message_batch(
                task_type="representation",
                work_unit_key=steve_work_unit_key,
                aqs_id=steve_aqs.id,
            )

            # Steve should also get all 6 messages for context
            assert len(steve_batch) == 6
            steve_message_ids = {msg.payload["message_id"] for msg in steve_batch}
            assert steve_message_ids == expected_batch_ids

    @pytest.mark.asyncio
    async def test_work_unit_returns_first_message_outside_token_limit_batch(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that work unit returns at least one message even if outside token-limited batch"""

        session, peers = sample_session_with_peers
        bob, steve, alice = peers[0], peers[1], peers[2]

        # Create messages: bob(800), steve(800), alice(100), alice(200)
        # Token limit: 1500 - should allow messages 1-2 (1600 tokens, first always included)
        # Alice's messages (3-4) are outside the chronological batch
        # BUT fallback logic should return alice's first message to prevent starvation
        messages_data = [
            (bob, 800),
            (steve, 800),
            (alice, 100),
            (alice, 200),
        ]

        messages: list[models.Message] = []
        for peer, token_count in messages_data:
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message from {peer.name}",
                token_count=token_count,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        # Create queue items

        alice_queue_items: list[models.QueueItem] = []
        bob_queue_items: list[models.QueueItem] = []
        steve_queue_items: list[models.QueueItem] = []

        for i, message in enumerate(messages):
            peer = messages_data[i][0]
            target = alice  # All observing alice

            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=message,
                task_type="representation",
                sender_name=peer.name,
                target_name=target.name,
            )
            work_unit_key = get_work_unit_key("representation", payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)

            if peer == alice:
                alice_queue_items.append(queue_item)
            elif peer == bob:
                bob_queue_items.append(queue_item)
            elif peer == steve:
                steve_queue_items.append(queue_item)

        await db_session.commit()
        for item in alice_queue_items + bob_queue_items + steve_queue_items:
            await db_session.refresh(item)

        qm = QueueManager()

        # Mock the token limit to 1500 for this test
        with patch.object(settings.DERIVER, "REPRESENTATION_BATCH_MAX_TOKENS", 1500):
            # Test alice's work unit
            # Messages: bob(800), steve(800), alice(100), alice(200)
            # Alice's first message appears at cumulative 1700, which exceeds the 1500 limit
            # Therefore, Alice's batch should be EMPTY - will be retried later when other
            # messages are processed and Alice's messages move closer to the front
            if alice_queue_items:
                alice_work_unit_key = alice_queue_items[0].work_unit_key
                alice_aqs = models.ActiveQueueSession(work_unit_key=alice_work_unit_key)
                db_session.add(alice_aqs)
                await db_session.commit()
                await db_session.refresh(alice_aqs)

                alice_batch = await qm.get_message_batch(
                    task_type="representation",
                    work_unit_key=alice_work_unit_key,
                    aqs_id=alice_aqs.id,
                )

                assert (
                    len(alice_batch) == 0
                )  # Empty batch - Alice is beyond token limit

            # Test bob's work unit
            # Bob's batch should include: bob(800) only
            # - bob's message is under the limit and it's his first message
            # - adding steve would exceed the limit and we already have Bob's message, so stop
            if bob_queue_items:
                bob_work_unit_key = bob_queue_items[0].work_unit_key
                bob_aqs = models.ActiveQueueSession(work_unit_key=bob_work_unit_key)
                db_session.add(bob_aqs)
                await db_session.commit()
                await db_session.refresh(bob_aqs)

                bob_batch = await qm.get_message_batch(
                    task_type="representation",
                    work_unit_key=bob_work_unit_key,
                    aqs_id=bob_aqs.id,
                )

                assert len(bob_batch) == 1
                assert bob_batch[0].payload["message_id"] == messages[0].id  # bob only

            # Test steve's work unit
            # Steve's first message appears at cumulative 1600, which exceeds the 1500 limit
            # Therefore, Steve's batch should be EMPTY - will be retried later
            if steve_queue_items:
                steve_work_unit_key = steve_queue_items[0].work_unit_key
                steve_aqs = models.ActiveQueueSession(work_unit_key=steve_work_unit_key)
                db_session.add(steve_aqs)
                await db_session.commit()
                await db_session.refresh(steve_aqs)

                steve_batch = await qm.get_message_batch(
                    task_type="representation",
                    work_unit_key=steve_work_unit_key,
                    aqs_id=steve_aqs.id,
                )

                assert (
                    len(steve_batch) == 0
                )  # Empty batch - Steve is beyond token limit

    @pytest.mark.asyncio
    async def test_single_message_processing(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that multiple summary messages in same work unit are processed separately"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create two summary messages
        token_counts = [500, 600]
        messages = [
            models.Message(
                id=999,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="First summary message",
            ),
            models.Message(
                id=1000,
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="Second summary message",
            ),
        ]

        # Create payloads and queue items
        queue_items: list[models.QueueItem] = []
        for i, message in enumerate(messages):
            payload = create_queue_payload(
                message, "summary", message_seq_in_session=i + 1
            )
            payload["token_count"] = token_counts[i]

            work_unit_key = get_work_unit_key("summary", payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="summary",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        # Mock and process work unit
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str,
            queue_payloads: list[dict[str, Any]],
            sender_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
            target_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            processed_batches.append(
                {"task_type": task_type, "payload_count": len(queue_payloads)}
            )

        qm = QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        worker_id = "test_worker"

        # Manually claim and assign ownership
        claimed_units = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed_units[work_unit_key]
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=aqs_id
        )

        with patch(
            "src.deriver.queue_manager.process_items",
            side_effect=mock_process_items,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Verify both messages were processed in separate batches
        assert len(processed_batches) == 2
        assert all(batch["task_type"] == "summary" for batch in processed_batches)
        assert all(batch["payload_count"] == 1 for batch in processed_batches)

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
        actual_token_counts = [
            item.payload.get("token_count") or 0 for item in processed_items
        ]
        assert sorted(actual_token_counts) == sorted(expected_token_counts)

    @pytest.mark.asyncio
    async def test_first_message_exceeds_token_limit_still_included(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that if the first message exceeds BATCH_MAX_TOKENS, it's still included alone"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create messages where first message exceeds the batch limit
        limit = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
        token_counts = [limit + 1000, 100, 200]  # First message way over limit

        # Create and save messages to the database first
        messages: list[models.Message] = []
        for i, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Test message {i}",
                token_count=token_count,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        payloads = [
            create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=msg,
                task_type="representation",
                sender_name=peer.name,
                target_name=peer.name,
            )
            for msg in messages
        ]

        # Add items to queue

        queue_items: list[models.QueueItem] = []
        for payload in payloads:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = get_work_unit_key(task_type, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str,
            queue_payloads: list[dict[str, Any]],
            sender_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
            target_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            processed_batches.append(
                {
                    "task_type": task_type,
                    "payload_count": len(queue_payloads),
                }
            )

        qm = QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        worker_id = "test_worker"

        # Manually claim and assign ownership
        claimed_units = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed_units[work_unit_key]
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=aqs_id
        )

        with patch(
            "src.deriver.queue_manager.process_items",
            side_effect=mock_process_items,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Should create 2 batches: first large message alone, then second and third together
        assert len(processed_batches) == 2
        assert (
            processed_batches[0]["payload_count"] == 1
        )  # First message (over limit) alone
        assert processed_batches[1]["payload_count"] == 2  # Second and third messages
        assert all(b["task_type"] == "representation" for b in processed_batches)

    @pytest.mark.asyncio
    async def test_message_exactly_at_token_limit(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test boundary condition when cumulative sum exactly equals limit"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create messages that test the exact boundary
        limit = settings.DERIVER.REPRESENTATION_BATCH_MAX_TOKENS
        token_counts = [
            limit // 2,
            limit // 2,
            1,
        ]  # First two exactly at limit, third exceeds

        # Create and save messages to the database first
        messages: list[models.Message] = []
        for i, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Test message {i}",
                token_count=token_count,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        payloads = [
            create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=msg,
                task_type="representation",
                sender_name=peer.name,
                target_name=peer.name,
            )
            for msg in messages
        ]

        # Add items to queue

        queue_items: list[models.QueueItem] = []
        for payload in payloads:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = get_work_unit_key(task_type, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_items(
            task_type: str,
            queue_payloads: list[dict[str, Any]],
            sender_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
            target_name: str | None = None,  # pyright: ignore[reportUnusedParameter]
        ) -> None:
            processed_batches.append(
                {
                    "task_type": task_type,
                    "payload_count": len(queue_payloads),
                }
            )

        qm = QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        worker_id = "test_worker"

        # Manually claim and assign ownership
        claimed_units = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed_units[work_unit_key]
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=aqs_id
        )

        with patch(
            "src.deriver.queue_manager.process_items",
            side_effect=mock_process_items,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Should create 2 batches: first two messages together (exactly at limit), third alone
        assert len(processed_batches) == 2
        assert (
            processed_batches[0]["payload_count"] == 2
        )  # First two messages (exactly at limit)
        assert (
            processed_batches[1]["payload_count"] == 1
        )  # Third message (exceeds limit)
        assert all(b["task_type"] == "representation" for b in processed_batches)
