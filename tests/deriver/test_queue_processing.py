import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.consumer import process_item
from src.deriver.queue_manager import QueueManager, WorkerOwnership
from src.utils.queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY, SummaryPayload
from src.utils.work_unit import construct_work_unit_key


@pytest.mark.asyncio
class TestQueueProcessing:
    """Test suite for queue processing functionality"""

    async def _add_representation_work_unit(
        self,
        *,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        token_counts: list[int],
        created_ats: list[datetime] | None = None,
    ) -> tuple[str, list[models.QueueItem]]:
        session, peers = sample_session_with_peers
        peer = peers[0]

        messages: list[models.Message] = []
        for index, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message {index}",
                token_count=token_count,
                seq_in_session=index + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        work_unit_key = ""
        queue_items: list[models.QueueItem] = []
        for index, message in enumerate(messages):
            payload = create_queue_payload(
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            work_unit_key = work_unit_key or construct_work_unit_key(
                session.workspace_name, payload
            )

            queue_item_kwargs: dict[str, Any] = {}
            if created_ats:
                queue_item_kwargs["created_at"] = created_ats[index]

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
                **queue_item_kwargs,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for queue_item in queue_items:
            await db_session.refresh(queue_item)

        return work_unit_key, queue_items

    async def test_get_and_claim_work_units(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    ) -> None:
        """Test that get_and_claim_work_units correctly identifies unprocessed work"""
        session, _peers = sample_session_with_peers

        # Verify we have queue items from our test setup
        assert len(sample_queue_items) == 9  # 6 representation + 3 summary
        expected_work_units = {item.work_unit_key for item in sample_queue_items}

        # Create a queue manager instance
        queue_manager = QueueManager()

        # Get available work units
        work_units = await queue_manager.get_and_claim_work_units()

        # Should return claimed work units from this test's seeded queue data
        assert len(work_units) > 0
        assert set(work_units).issubset(expected_work_units)

        # Check that all work units have the expected structure
        for work_unit in work_units:
            assert isinstance(work_unit, str)
            assert work_unit.split(":")[0] in ["representation", "summary"]
            assert f":{session.workspace_name}:" in work_unit

        tracked_keys = (
            await db_session.execute(
                select(models.ActiveQueueSession.work_unit_key).where(
                    models.ActiveQueueSession.work_unit_key.in_(list(work_units.keys()))
                )
            )
        ).scalars()
        assert set(tracked_keys) == set(work_units.keys())

    async def test_work_unit_claiming(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001 # pyright: ignore[reportUnusedParameter]
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
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001 # pyright: ignore[reportUnusedParameter]
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
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001 # pyright: ignore[reportUnusedParameter]
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
        for i in range(3):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="hello",
                token_count=10,
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        payloads: list[tuple[dict[str, Any], int]] = []
        for message in messages:
            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            payloads.append((payload, message.id))

        items = await add_queue_items(payloads, session.id, session.workspace_name)
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

        batch = await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key=first.work_unit_key,
            aqs_id=aqs.id,
        )
        items_to_process = batch.items_to_process
        nxt = items_to_process[0] if items_to_process else None
        assert nxt is not None and nxt.id == first.id

        # Mark first processed, next should be the second
        first.processed = True
        await db_session.commit()
        batch2 = await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key=first.work_unit_key,
            aqs_id=aqs.id,
        )
        items_to_process2 = batch2.items_to_process
        nxt2 = items_to_process2[0] if items_to_process2 else None
        assert nxt2 is not None and nxt2.id == second.id

    @pytest.mark.asyncio
    async def test_cleanup_work_unit_removes_row(
        self,
        sample_queue_items: list[models.QueueItem],  # noqa: ARG001 # pyright: ignore[reportUnusedParameter]
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
        limit = settings.DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
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
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create queue items with token counts
        payload_entries = [
            (
                create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                    message=msg,
                    task_type="representation",
                    observed=peer.name,
                    observer=peer.name,
                ),
                msg,
            )
            for msg in messages
        ]

        queue_items: list[models.QueueItem] = []
        for payload, message in payload_entries:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_representation_batch(
            messages: list[models.Message],
            _message_level_configuration: Any,
            *,
            observed: str | None = None,  # pyright: ignore[reportUnusedParameter]
            observers: list[str] | None = None,  # pyright: ignore[reportUnusedParameter]
            queue_item_message_ids: list[int] | None = None,  # pyright: ignore[reportUnusedParameter]
            **_extra: Any,  # added hit_batch_token_cap / was_flush_enabled / batch_max_tokens
        ) -> None:
            processed_batches.append(
                {
                    "task_type": "representation",
                    "payload_count": len(messages),
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
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=mock_process_representation_batch,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Should create 2 batches due to token limits
        assert len(processed_batches) == 2
        assert processed_batches[0]["payload_count"] == 2
        assert processed_batches[1]["payload_count"] == 1
        assert all(b["task_type"] == "representation" for b in processed_batches)

    @pytest.mark.asyncio
    async def test_hit_batch_token_cap_reflects_post_filter_batch(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: `hit_batch_token_cap` must reflect the actually-returned
        batch (post config-filter), not the pre-filter superset. Previously
        the flag used pre-filter `messages_context[-1].id`, which inflated
        the range queried for cap detection and produced false positives
        when the config-filter trimmed the trailing item from the batch.
        """
        from src.deriver import queue_manager as qm_module

        session, peers = sample_session_with_peers
        peer = peers[0]
        cap = settings.DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS

        # M1 + M2 sum to exactly the cap; M3 pushes over it. After SQL,
        # messages_context = [M1, M2]; the cap is genuinely binding *on the
        # pre-filter batch*. items_to_process = [QI(M1), QI(M2)]. We then
        # simulate a config-filter trim that keeps only QI(M1). The
        # actually-returned batch is [M1] alone — sum=400 < cap — so the
        # cap-flag must report False.
        token_counts = [400, cap - 400, 300]
        messages: list[models.Message] = []
        for i, tc in enumerate(token_counts):
            m = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"cap-test message {i}",
                token_count=tc,
                seq_in_session=i + 1,
            )
            db_session.add(m)
            messages.append(m)
        await db_session.commit()
        for m in messages:
            await db_session.refresh(m)

        queue_items: list[models.QueueItem] = []
        for m in messages:
            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=m,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)
            qi = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=m.id,
            )
            db_session.add(qi)
            queue_items.append(qi)
        await db_session.commit()
        for qi in queue_items:
            await db_session.refresh(qi)

        # Trim items_to_process down to the first queue item — mimics
        # `_resolve_batch_configuration` cutting at a configuration boundary.
        real_resolve = qm_module._resolve_batch_configuration  # pyright: ignore[reportPrivateUsage]

        def fake_resolve(
            items: list[models.QueueItem],
        ) -> tuple[list[models.QueueItem], Any]:
            _kept, cfg = real_resolve(items)
            return (items[:1] if items else []), cfg

        monkeypatch.setattr(qm_module, "_resolve_batch_configuration", fake_resolve)

        qm = qm_module.QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        claimed = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed[work_unit_key]
        await db_session.commit()

        result = await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key=work_unit_key,
            aqs_id=aqs_id,
        )

        # Returned batch is [M1] alone (config-filter trimmed M2). The cap
        # wasn't binding on this batch — sum=400 < cap. Pre-fix code reported
        # True (false positive) because it used pre-filter max_kept_id=M2.
        assert len(result.messages_context) == 1
        assert result.messages_context[0].id == messages[0].id
        assert result.hit_batch_token_cap is False

    @pytest.mark.asyncio
    async def test_hit_batch_token_cap_fires_when_trailing_context_trimmed(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Regression: when SQL kept trailing NON-QUEUE context past the last
        queued item, and the config filter trims that context away, the cap
        check must still recognize that the SQL cap clamped queue work.

        Pre-fix used `messages_context[-1].id == sql_max_kept_id` which goes
        False whenever trailing context is dropped — producing a false
        negative for the very case the cap-hit flag exists to report.
        Post-fix keys on the queue-item boundary, which is unaffected by
        trailing-context trimming.
        """
        from src.deriver import queue_manager as qm_module

        session, peers = sample_session_with_peers
        peer_a = peers[0]
        peer_b = peers[1] if len(peers) > 1 else peers[0]
        cap = settings.DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS

        # Layout: 4 messages, ordered.
        #   M1 (peer_a, queue, 200)
        #   M2 (peer_a, queue, 200)
        #   M3 (peer_b, NON-queue context, cap - 300)
        #   M4 (peer_a, queue, 400)
        # Cumulative tokens: M1=200, M2=400, M3=cap+100, M4=cap+500.
        # SQL keeps M1+M2 (cumulative <= cap), excludes M3 onwards (over cap).
        # Wait — we want SQL to keep through M3 (trailing context) but exclude
        # M4 (queue). Adjust so M3 fits but M4 doesn't.
        token_counts: list[tuple[models.Peer, int]] = [
            (peer_a, 200),  # M1 — queue
            (peer_a, 200),  # M2 — queue
            (peer_b, cap - 700),  # M3 — non-queue context; cumulative = cap-300
            (peer_a, 400),  # M4 — queue; cumulative cap+100 > cap → excluded
        ]
        messages: list[models.Message] = []
        for i, (msg_peer, tc) in enumerate(token_counts):
            m = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=msg_peer.name,
                content=f"cap-test message {i}",
                token_count=tc,
                seq_in_session=i + 1,
            )
            db_session.add(m)
            messages.append(m)
        await db_session.commit()
        for m in messages:
            await db_session.refresh(m)

        # Queue items for M1, M2, M4 only — M3 is non-queue context (peer_b).
        # observed = peer_a, so the deriver's representation work unit covers
        # peer_a messages; peer_b is treated as conversational context.
        queue_items: list[models.QueueItem] = []
        for m in [messages[0], messages[1], messages[3]]:
            payload = create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                message=m,
                task_type="representation",
                observed=peer_a.name,
                observer=peer_a.name,
            )
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)
            qi = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=m.id,
            )
            db_session.add(qi)
            queue_items.append(qi)
        await db_session.commit()
        for qi in queue_items:
            await db_session.refresh(qi)

        qm = qm_module.QueueManager()
        work_unit_key = queue_items[0].work_unit_key
        claimed = await qm.claim_work_units(db_session, [work_unit_key])
        aqs_id = claimed[work_unit_key]
        await db_session.commit()

        result = await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key=work_unit_key,
            aqs_id=aqs_id,
        )

        # Cap fired: SQL stopped at M3 (cap budget exhausted), excluding the
        # queue item M4. Config filter doesn't touch queue items here. The
        # flag must report True.
        assert result.hit_batch_token_cap is True

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
        for i, (peer, token_count) in enumerate(messages_data):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message from {peer.name}",
                token_count=token_count,
                seq_in_session=i + 1,
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
                observed=peer.name,
                observer=target.name,
            )
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
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
        with patch.object(
            settings.DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 2000
        ):
            # Test alice's work unit
            alice_work_unit_key = alice_queue_items[0].work_unit_key
            alice_aqs = models.ActiveQueueSession(work_unit_key=alice_work_unit_key)
            db_session.add(alice_aqs)
            await db_session.commit()
            await db_session.refresh(alice_aqs)

            alice_batch = await qm.get_queue_item_batch(
                task_type="representation",
                work_unit_key=alice_work_unit_key,
                aqs_id=alice_aqs.id,
            )
            alice_messages = alice_batch.messages_context
            alice_items = alice_batch.items_to_process

            assert len(alice_messages) == 6
            alice_message_ids: set[int] = {m.id for m in alice_messages}
            expected_batch_ids = {
                messages[0].id,  # alice(250)
                messages[1].id,  # bob(400)
                messages[2].id,  # steve(300)
                messages[3].id,  # alice(500)
                messages[4].id,  # bob(500)
                messages[5].id,  # alice(40)
            }
            assert alice_message_ids == expected_batch_ids

            # Ensure items are only for alice
            assert all(qi.payload.get("observed") == alice.name for qi in alice_items)

            # Test bob's work unit - now includes preceding message for context
            bob_work_unit_key = bob_queue_items[0].work_unit_key
            bob_aqs = models.ActiveQueueSession(work_unit_key=bob_work_unit_key)
            db_session.add(bob_aqs)
            await db_session.commit()
            await db_session.refresh(bob_aqs)

            bob_batch = await qm.get_queue_item_batch(
                task_type="representation",
                work_unit_key=bob_work_unit_key,
                aqs_id=bob_aqs.id,
            )
            bob_messages = bob_batch.messages_context
            bob_items = bob_batch.items_to_process

            # Bob should get 5 messages (1..5) - includes preceding alice message for context
            assert len(bob_messages) == 5
            bob_message_ids: set[int] = {m.id for m in bob_messages}
            expected_bob_ids = {
                messages[0].id,  # alice(250) - preceding context
                messages[1].id,  # bob(400)
                messages[2].id,  # steve(300)
                messages[3].id,  # alice(500)
                messages[4].id,  # bob(500)
            }
            assert bob_message_ids == expected_bob_ids
            # Ensure items are only for bob
            assert all(qi.payload.get("observed") == bob.name for qi in bob_items)

            # Test steve's work unit - now includes preceding message for context
            steve_work_unit_key = steve_queue_items[0].work_unit_key
            steve_aqs = models.ActiveQueueSession(work_unit_key=steve_work_unit_key)
            db_session.add(steve_aqs)
            await db_session.commit()
            await db_session.refresh(steve_aqs)

            steve_batch = await qm.get_queue_item_batch(
                task_type="representation",
                work_unit_key=steve_work_unit_key,
                aqs_id=steve_aqs.id,
            )
            steve_messages = steve_batch.messages_context
            steve_items = steve_batch.items_to_process

            # Steve should get 6 messages (2..7) - includes preceding bob message for context
            assert len(steve_messages) == 6
            steve_message_ids: set[int] = {m.id for m in steve_messages}
            expected_steve_ids = {
                messages[1].id,  # bob(400) - preceding context
                messages[2].id,  # steve(300)
                messages[3].id,  # alice(500)
                messages[4].id,  # bob(500)
                messages[5].id,  # alice(40)
                messages[6].id,  # steve(100)
            }
            assert steve_message_ids == expected_steve_ids
            # Ensure items are only for steve
            assert all(qi.payload.get("observed") == steve.name for qi in steve_items)

    @pytest.mark.asyncio
    async def test_per_work_unit_anchoring_with_token_limits(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that per-work-unit anchoring processes each sender's messages independently within token limits"""

        session, peers = sample_session_with_peers
        bob, steve, alice = peers[0], peers[1], peers[2]

        # Create messages: bob(800), steve(800), alice(100), alice(200)
        # Token limit: 1500
        # With per-work-unit anchoring:
        # - Bob's work unit: starts at message 1, includes only bob(800)
        # - Steve's work unit: starts at message 2, includes only steve(800)
        # - Alice's work unit: starts at message 3, includes alice(100) + alice(200)
        messages_data = [
            (bob, 800),
            (steve, 800),
            (alice, 100),
            (alice, 200),
        ]

        messages: list[models.Message] = []
        for i, (peer, token_count) in enumerate(messages_data):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message from {peer.name}",
                token_count=token_count,
                seq_in_session=i + 1,
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
                observed=peer.name,
                observer=target.name,
            )
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
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
        with patch.object(
            settings.DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 1500
        ):
            # Test alice's work unit
            # With per-work-unit anchoring + preceding context:
            # Alice starts at message 3, includes preceding message 2 (steve) for context
            # Alice's batch: steve(800) + alice(100) + alice(200) = 1100 tokens, under 1500 limit
            if alice_queue_items:
                alice_work_unit_key = alice_queue_items[0].work_unit_key
                alice_aqs = models.ActiveQueueSession(work_unit_key=alice_work_unit_key)
                db_session.add(alice_aqs)
                await db_session.commit()
                await db_session.refresh(alice_aqs)

                alice_batch2 = await qm.get_queue_item_batch(
                    task_type="representation",
                    work_unit_key=alice_work_unit_key,
                    aqs_id=alice_aqs.id,
                )
                alice_messages2 = alice_batch2.messages_context

                # Includes preceding steve message for context -> [2,3,4]
                assert len(alice_messages2) == 3
                assert [m.id for m in alice_messages2] == [
                    messages[1].id,  # steve - preceding context
                    messages[2].id,  # alice
                    messages[3].id,  # alice
                ]

            # Test bob's work unit
            # Bob starts at message 1, no preceding message available
            # Bob's batch: bob(800) only, under 1500 limit
            if bob_queue_items:
                bob_work_unit_key = bob_queue_items[0].work_unit_key
                bob_aqs = models.ActiveQueueSession(work_unit_key=bob_work_unit_key)
                db_session.add(bob_aqs)
                await db_session.commit()
                await db_session.refresh(bob_aqs)

                bob_batch2 = await qm.get_queue_item_batch(
                    task_type="representation",
                    work_unit_key=bob_work_unit_key,
                    aqs_id=bob_aqs.id,
                )
                bob_messages2 = bob_batch2.messages_context

                assert len(bob_messages2) == 1
                assert bob_messages2[0].id == messages[0].id  # bob only

            # Test steve's work unit
            # Steve starts at message 2, includes preceding message 1 (bob) for context
            # Steve's batch: bob(800) + steve(800) = 1600 tokens, exceeds 1500 limit
            # So should only get steve's message
            if steve_queue_items:
                steve_work_unit_key = steve_queue_items[0].work_unit_key
                steve_aqs = models.ActiveQueueSession(work_unit_key=steve_work_unit_key)
                db_session.add(steve_aqs)
                await db_session.commit()
                await db_session.refresh(steve_aqs)

                steve_batch2 = await qm.get_queue_item_batch(
                    task_type="representation",
                    work_unit_key=steve_work_unit_key,
                    aqs_id=steve_aqs.id,
                )
                steve_messages2 = steve_batch2.messages_context

                # Includes preceding bob message for context -> [1,2]
                assert len(steve_messages2) == 2
                assert [m.id for m in steve_messages2] == [
                    messages[0].id,  # bob - preceding context
                    messages[1].id,  # steve
                ]

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
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="First summary message",
                public_id=generate_nanoid(),
                seq_in_session=1,
            ),
            models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content="Second summary message",
                public_id=generate_nanoid(),
                seq_in_session=2,
            ),
        ]

        # Save messages to database first
        for message in messages:
            db_session.add(message)
        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create payloads and queue items
        queue_items: list[models.QueueItem] = []
        for i, message in enumerate(messages):
            payload = create_queue_payload(
                message, "summary", message_seq_in_session=i + 1
            )
            payload["token_count"] = token_counts[i]

            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="summary",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        # Mock and process work unit
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_item(
            queue_item: models.QueueItem,
        ) -> None:
            processed_batches.append(
                {"task_type": queue_item.task_type, "payload_count": 1}
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
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_item",
            side_effect=mock_process_item,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # Verify both messages were processed in separate batches
        assert len(processed_batches) == 2
        assert all(batch["task_type"] == "summary" for batch in processed_batches)
        assert all(batch["payload_count"] == 1 for batch in processed_batches)

        # Expire cached objects so we see updates made by tracked_db sessions
        db_session.expire_all()

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
        limit = settings.DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
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
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        payload_entries = [
            (
                create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                    message=msg,
                    task_type="representation",
                    observed=peer.name,
                    observer=peer.name,
                ),
                msg,
            )
            for msg in messages
        ]

        # Add items to queue
        queue_items: list[models.QueueItem] = []
        for payload, message in payload_entries:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_representation_batch(
            messages: list[models.Message],
            _message_level_configuration: Any,
            *,
            observed: str | None = None,  # pyright: ignore[reportUnusedParameter]
            observers: list[str] | None = None,  # pyright: ignore[reportUnusedParameter]
            queue_item_message_ids: list[int] | None = None,  # pyright: ignore[reportUnusedParameter]
            **_extra: Any,  # added hit_batch_token_cap / was_flush_enabled / batch_max_tokens
        ) -> None:
            processed_batches.append(
                {
                    "task_type": "representation",
                    "payload_count": len(messages),
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
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=mock_process_representation_batch,
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
        limit = settings.DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
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
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()

        # Refresh to get the actual IDs
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        payload_entries = [
            (
                create_queue_payload(  # type: ignore[reportUnknownArgumentType]
                    message=msg,
                    task_type="representation",
                    observed=peer.name,
                    observer=peer.name,
                ),
                msg,
            )
            for msg in messages
        ]

        # Add items to queue
        queue_items: list[models.QueueItem] = []
        for payload, message in payload_entries:
            task_type = payload.get("task_type", "unknown")
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()
        for item in queue_items:
            await db_session.refresh(item)

        # Mock process_items to capture batches
        processed_batches: list[dict[str, Any]] = []

        async def mock_process_representation_batch(
            messages: list[models.Message],
            _message_level_configuration: Any,
            *,
            observed: str | None = None,  # pyright: ignore[reportUnusedParameter]
            observers: list[str] | None = None,  # pyright: ignore[reportUnusedParameter]
            queue_item_message_ids: list[int] | None = None,  # pyright: ignore[reportUnusedParameter]
            **_extra: Any,  # added hit_batch_token_cap / was_flush_enabled / batch_max_tokens
        ) -> None:
            processed_batches.append(
                {
                    "task_type": "representation",
                    "payload_count": len(messages),
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
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=mock_process_representation_batch,
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

    @pytest.mark.asyncio
    async def test_forced_batching_waits_for_threshold(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that representation work units below token threshold are not claimed.

        The token-threshold gate in QueueManager.get_and_claim_work_units is
        skipped entirely when DERIVER_FLUSH_ENABLED is True, so this test
        forces it False regardless of what the process env has set (benches
        commonly enable flush mode for immediate processing).
        """
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create messages with tokens BELOW the threshold
        limit = settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS
        token_counts = [100, 100, 100]  # Total 300, way below 4096

        messages: list[models.Message] = []
        for i, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Short message {i}",
                token_count=token_count,
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        queue_items: list[models.QueueItem] = []
        for message in messages:
            payload = create_queue_payload(
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        qm = QueueManager()

        # Try to claim work units - should NOT return the representation work unit
        claimed = await qm.get_and_claim_work_units()

        # The representation work unit should NOT be claimed (tokens below threshold)
        rep_work_unit_key = queue_items[0].work_unit_key
        assert rep_work_unit_key not in claimed

        # Now add more messages to exceed the threshold
        more_messages: list[models.Message] = []
        for i in range(10):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Longer message {i}",
                token_count=limit // 2,  # Each message is half the limit
                seq_in_session=len(messages) + i + 1,
            )
            db_session.add(message)
            more_messages.append(message)

        await db_session.commit()
        for message in more_messages:
            await db_session.refresh(message)

        # Add queue items for new messages
        for message in more_messages:
            payload = create_queue_payload(
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=rep_work_unit_key,  # Same work unit
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)

        await db_session.commit()

        # Now the work unit should be claimable (tokens exceed threshold)
        claimed2 = await qm.get_and_claim_work_units()
        assert rep_work_unit_key in claimed2

    @pytest.mark.asyncio
    async def test_age_flush_waits_for_fresh_sub_threshold_items(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)
        monkeypatch.setattr(
            settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800
        )

        work_unit_key, _queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100, 100],
        )

        claimed = await QueueManager().get_and_claim_work_units()

        assert work_unit_key not in claimed

    @pytest.mark.asyncio
    async def test_age_flush_claims_old_sub_threshold_items_and_fetches_tail(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)
        monkeypatch.setattr(
            settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800
        )
        old_timestamp = datetime.now(UTC) - timedelta(hours=2)

        work_unit_key, queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100, 100],
            created_ats=[old_timestamp, old_timestamp, old_timestamp],
        )

        qm = QueueManager()
        claimed = await qm.get_and_claim_work_units()

        assert work_unit_key in claimed
        batch = await qm.get_queue_item_batch(
            task_type="representation",
            work_unit_key=work_unit_key,
            aqs_id=claimed[work_unit_key],
        )
        assert [item.id for item in batch.items_to_process] == [
            item.id for item in queue_items
        ]

    @pytest.mark.asyncio
    async def test_age_flush_zero_preserves_legacy_wait_for_old_items(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)
        monkeypatch.setattr(settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 0)
        old_timestamp = datetime.now(UTC) - timedelta(hours=2)

        work_unit_key, _queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100],
            created_ats=[old_timestamp, old_timestamp],
        )

        claimed = await QueueManager().get_and_claim_work_units()

        assert work_unit_key not in claimed

    @pytest.mark.asyncio
    async def test_flush_enabled_bypasses_age_and_token_thresholds(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", True)
        monkeypatch.setattr(
            settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800
        )

        work_unit_key, _queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100],
        )

        claimed = await QueueManager().get_and_claim_work_units()

        assert work_unit_key in claimed

    @pytest.mark.asyncio
    async def test_age_flush_uses_oldest_unprocessed_item(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)
        monkeypatch.setattr(
            settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800
        )
        now = datetime.now(UTC)

        work_unit_key, _queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100],
            created_ats=[now - timedelta(hours=2), now],
        )

        claimed = await QueueManager().get_and_claim_work_units()

        assert work_unit_key in claimed

    @pytest.mark.asyncio
    async def test_age_flush_ignores_old_processed_items(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "FLUSH_ENABLED", False)
        monkeypatch.setattr(
            settings.DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800
        )
        now = datetime.now(UTC)

        work_unit_key, queue_items = await self._add_representation_work_unit(
            db_session=db_session,
            sample_session_with_peers=sample_session_with_peers,
            create_queue_payload=create_queue_payload,
            token_counts=[100, 100],
            created_ats=[now - timedelta(hours=2), now],
        )
        queue_items[0].processed = True
        await db_session.commit()

        claimed = await QueueManager().get_and_claim_work_units()

        assert work_unit_key not in claimed

    @pytest.mark.asyncio
    async def test_forced_batching_single_large_message(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that a single message >= threshold is immediately claimable"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        limit = settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS

        # Create a single message that exceeds the threshold
        message = models.Message(
            session_name=session.name,
            workspace_name=session.workspace_name,
            peer_name=peer.name,
            content="A very long message",
            token_count=limit + 1000,  # Exceeds threshold
            seq_in_session=1,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)

        # Create queue item
        payload = create_queue_payload(
            message=message,
            task_type="representation",
            observed=peer.name,
            observer=peer.name,
        )
        work_unit_key = construct_work_unit_key(session.workspace_name, payload)

        queue_item = models.QueueItem(
            session_id=session.id,
            task_type="representation",
            work_unit_key=work_unit_key,
            payload=payload,
            processed=False,
            workspace_name=session.workspace_name,
            message_id=message.id,
        )
        db_session.add(queue_item)
        await db_session.commit()

        qm = QueueManager()

        # Single large message should be immediately claimable
        claimed = await qm.get_and_claim_work_units()
        assert work_unit_key in claimed

    @pytest.mark.asyncio
    async def test_forced_batching_bypassed_for_summary(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that summary tasks are processed immediately regardless of tokens"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        # Create a message with very few tokens
        message = models.Message(
            session_name=session.name,
            workspace_name=session.workspace_name,
            peer_name=peer.name,
            content="Short",
            token_count=10,  # Way below threshold
            seq_in_session=1,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)

        # Create a SUMMARY queue item (not representation)
        payload = create_queue_payload(
            message=message,
            task_type="summary",
            message_seq_in_session=1,
        )
        work_unit_key = construct_work_unit_key(session.workspace_name, payload)

        queue_item = models.QueueItem(
            session_id=session.id,
            task_type="summary",
            work_unit_key=work_unit_key,
            payload=payload,
            processed=False,
            workspace_name=session.workspace_name,
            message_id=message.id,
        )
        db_session.add(queue_item)
        await db_session.commit()

        qm = QueueManager()

        # Summary should be claimable regardless of token count
        claimed = await qm.get_and_claim_work_units()
        assert work_unit_key in claimed
        assert work_unit_key.startswith("summary:")

    @pytest.mark.asyncio
    async def test_forced_batching_exact_threshold(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """Test that work units exactly at the threshold are claimable"""

        session, peers = sample_session_with_peers
        peer = peers[0]

        limit = settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS

        # Create messages that sum to exactly the threshold
        token_counts = [limit // 2, limit // 2]

        messages: list[models.Message] = []
        for i, token_count in enumerate(token_counts):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message {i}",
                token_count=token_count,
                seq_in_session=i + 1,
            )
            db_session.add(message)
            messages.append(message)

        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        # Create queue items
        work_unit_key = None
        for message in messages:
            payload = create_queue_payload(
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            if work_unit_key is None:
                work_unit_key = construct_work_unit_key(session.workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)

        await db_session.commit()

        qm = QueueManager()

        # Work unit exactly at threshold should be claimable
        claimed = await qm.get_and_claim_work_units()
        assert work_unit_key is not None
        assert work_unit_key in claimed


class TestPollingJitter:
    """Polling jitter: desynchronize poll loops without changing the schedule."""

    def test_jitter_stays_within_ratio_bounds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.5)
        qm = QueueManager()
        samples = [qm._jitter(10.0) for _ in range(1000)]  # pyright: ignore[reportPrivateUsage]
        assert all(5.0 <= s <= 15.0 for s in samples)
        # A 0.5 ratio over 1000 samples should produce real spread, not a constant.
        assert max(samples) - min(samples) > 1.0

    def test_jitter_ratio_zero_is_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.0)
        qm = QueueManager()
        assert all(qm._jitter(7.5) == 7.5 for _ in range(50))  # pyright: ignore[reportPrivateUsage]

    def test_advance_jitters_return_but_keeps_deterministic_schedule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "POLLING_JITTER_RATIO", 0.5)
        monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_ENABLED", True)
        monkeypatch.setattr(settings.DERIVER, "POLLING_BACKOFF_MULTIPLIER", 2.0)
        monkeypatch.setattr(settings.DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)
        monkeypatch.setattr(
            settings.DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 30.0
        )
        qm = QueueManager()

        # The underlying schedule advances 1 -> 2 -> 4 -> ... -> 30 deterministically;
        # each returned sleep is jittered within [0.5x, 1.5x] of the pre-advance step.
        expected_schedule = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
        for step in expected_schedule:
            returned = qm._advance_poll_interval()  # pyright: ignore[reportPrivateUsage]
            assert 0.5 * step <= returned <= 1.5 * step

    @pytest.mark.asyncio
    async def test_startup_jitter_disabled_returns_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 0.0)
        qm = QueueManager()
        # Window 0.0 must not sleep at all.
        await qm._sleep_startup_jitter()  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_startup_jitter_interrupted_by_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings.DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 300.0)
        qm = QueueManager()
        qm.shutdown_event.set()
        # A shutdown already signalled must short-circuit the (long) jitter sleep.
        await asyncio.wait_for(qm._sleep_startup_jitter(), timeout=1.0)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
class TestQueueRetry:
    """Bounded retry of transient errors in process_work_unit (DEV-1975).

    A transient failure (deadlock, lost connection, provider transport) must
    leave the batch's queue items unprocessed and release the work unit for
    re-claim, up to MAX_RETRYABLE_ATTEMPTS per work unit; terminal failures
    keep today's burn-one-item behavior.
    """

    async def _seed_work_unit(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        n_messages: int = 1,
    ) -> tuple[QueueManager, str, str, list[models.QueueItem]]:
        """Seed a claimed representation work unit owned by a test worker."""
        session, peers = sample_session_with_peers
        peer = peers[0]

        messages: list[models.Message] = []
        for index in range(n_messages):
            message = models.Message(
                session_name=session.name,
                workspace_name=session.workspace_name,
                peer_name=peer.name,
                content=f"Message {index}",
                token_count=10,
                seq_in_session=index + 1,
            )
            db_session.add(message)
            messages.append(message)
        await db_session.commit()
        for message in messages:
            await db_session.refresh(message)

        queue_items: list[models.QueueItem] = []
        work_unit_key = ""
        for message in messages:
            payload = create_queue_payload(
                message=message,
                task_type="representation",
                observed=peer.name,
                observer=peer.name,
            )
            work_unit_key = work_unit_key or construct_work_unit_key(
                session.workspace_name, payload
            )
            queue_item = models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)
        await db_session.commit()
        for queue_item in queue_items:
            await db_session.refresh(queue_item)

        qm = QueueManager()
        worker_id = "test_worker"
        claimed_units = await qm.claim_work_units(db_session, [work_unit_key])
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=claimed_units[work_unit_key]
        )
        await db_session.commit()
        return qm, work_unit_key, worker_id, queue_items

    @staticmethod
    def _retryable_error() -> OperationalError:
        class FakePGError(Exception):
            sqlstate: str = "40P01"

        return OperationalError("UPDATE documents", {}, FakePGError())

    async def _fetch_items(
        self, db_session: AsyncSession, work_unit_key: str
    ) -> list[models.QueueItem]:
        db_session.expire_all()
        return list(
            (
                await db_session.execute(
                    select(models.QueueItem)
                    .where(models.QueueItem.work_unit_key == work_unit_key)
                    .order_by(models.QueueItem.id)
                )
            )
            .scalars()
            .all()
        )

    async def _aqs_rows(self, db_session: AsyncSession, work_unit_key: str) -> int:
        return len(
            (
                await db_session.execute(
                    select(models.ActiveQueueSession).where(
                        models.ActiveQueueSession.work_unit_key == work_unit_key
                    )
                )
            )
            .scalars()
            .all()
        )

    async def _retry_attempts_on_items(
        self, db_session: AsyncSession, work_unit_key: str
    ) -> int | None:
        items = await self._fetch_items(db_session, work_unit_key)
        unprocessed = [item for item in items if not item.processed]
        if not unprocessed:
            return None
        raw = (unprocessed[0].payload or {}).get("_retry_attempts")
        return None if raw is None else int(raw)

    async def test_retryable_error_leaves_items_unprocessed(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transient error stops the work unit after ONE batch fetch (no
        tight loop), leaves items unprocessed with no error, and releases
        the ActiveQueueSession row."""
        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload, n_messages=2
        )
        initial_semaphore_value = qm.semaphore._value

        batch_fetches = 0
        original_get_batch = qm.get_queue_item_batch

        async def counting_get_batch(*args: Any, **kwargs: Any) -> Any:
            nonlocal batch_fetches
            batch_fetches += 1
            return await original_get_batch(*args, **kwargs)

        with (
            patch.object(qm, "get_queue_item_batch", side_effect=counting_get_batch),
            patch(
                "src.deriver.queue_manager.process_representation_batch",
                side_effect=self._retryable_error(),
            ),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        assert batch_fetches == 1
        items = await self._fetch_items(db_session, work_unit_key)
        assert all(not item.processed for item in items)
        assert all(item.error is None for item in items)
        assert await self._aqs_rows(db_session, work_unit_key) == 0
        assert await self._retry_attempts_on_items(db_session, work_unit_key) == 1
        assert qm.semaphore._value == initial_semaphore_value

    async def test_retry_exhaustion_is_terminal(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At the attempt cap a transient error burns the first item exactly
        like today's terminal path and clears the counter."""
        from src.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS

        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload
        )
        await qm._set_work_unit_retry_attempts(  # pyright: ignore[reportPrivateUsage]
            work_unit_key, MAX_RETRYABLE_ATTEMPTS - 1
        )

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=self._retryable_error(),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "OperationalError" in items[0].error
        assert await self._retry_attempts_on_items(db_session, work_unit_key) is None

    async def test_non_retryable_error_burns_immediately(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """A non-retryable error keeps today's behavior verbatim: the first
        item is marked errored on the first attempt."""
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload
        )

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=ValueError("bad batch"),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "ValueError" in items[0].error
        assert await self._retry_attempts_on_items(db_session, work_unit_key) is None

    async def test_counter_cleared_after_success(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """A success wipes the accumulated attempt count for the work unit."""
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload
        )
        await qm._set_work_unit_retry_attempts(work_unit_key, 1)  # pyright: ignore[reportPrivateUsage]

        async def noop_batch(*_args: Any, **_kwargs: Any) -> None:
            return None

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=noop_batch,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert all(item.processed for item in items)
        assert all(item.error is None for item in items)
        # Counter lives on the oldest unprocessed item; once that item is
        # processed the budget is gone even if the payload key remains.
        assert await self._retry_attempts_on_items(db_session, work_unit_key) is None

    async def test_retry_budget_survives_reclaim_by_another_manager(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second QueueManager continues the durable attempt budget."""
        from src.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS

        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm1, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload
        )

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=self._retryable_error(),
        ):
            await qm1.process_work_unit(work_unit_key, worker_id)

        assert await self._retry_attempts_on_items(db_session, work_unit_key) == 1
        assert await self._aqs_rows(db_session, work_unit_key) == 0

        # Seed the remaining budget so the next reclaim is the terminal attempt.
        qm2 = QueueManager()
        await qm2._set_work_unit_retry_attempts(  # pyright: ignore[reportPrivateUsage]
            work_unit_key, MAX_RETRYABLE_ATTEMPTS - 1
        )
        claimed = await qm2.claim_work_units(db_session, [work_unit_key])
        worker_id_2 = "test_worker_2"
        qm2.worker_ownership[worker_id_2] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=claimed[work_unit_key]
        )
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=self._retryable_error(),
        ):
            await qm2.process_work_unit(work_unit_key, worker_id_2)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "OperationalError" in items[0].error

    async def test_process_item_strips_retry_counter_before_validation(self) -> None:
        """A reclaimed non-representation item must survive its own retry counter.

        The counter is written onto an *unprocessed* item so the budget outlives
        a work-unit reclaim -- which means the next claim re-reads it. Every
        payload model sets ``extra="forbid"``, so without the strip in
        ``process_item`` the reclaim raises extra_forbidden -> ValueError ->
        not retryable -> the item is burned terminally on the very attempt that
        was supposed to retry it. Representation tasks never hit this: their
        batch path reads the payload with ``.get()`` instead of validating,
        which is why the rest of this class cannot catch it.
        """
        raw: dict[str, Any] = {
            "task_type": "summary",
            "session_name": "s",
            "message_seq_in_session": 1,
            "message_public_id": "msg-public-id",
            "configuration": {
                "reasoning": {"enabled": True},
                "peer_card": {"use": True, "create": True},
                "summary": {
                    "enabled": True,
                    "messages_per_short_summary": 20,
                    "messages_per_long_summary": 60,
                },
                "dream": {"enabled": True},
            },
            RETRY_ATTEMPTS_PAYLOAD_KEY: 1,
        }

        # Pin the premise: the payload model must keep rejecting the key, so
        # this fails loudly if someone "fixes" the burn with extra="allow"
        # instead of stripping.
        with pytest.raises(ValidationError) as exc_info:
            SummaryPayload.model_validate(raw)
        assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())

        queue_item = models.QueueItem(
            task_type="summary",
            work_unit_key="summary:test-workspace:test-session",
            payload=raw,
            processed=False,
            workspace_name="test-workspace",
            message_id=1,
        )

        with patch(
            "src.deriver.consumer.summarizer.summarize_if_needed",
            new_callable=AsyncMock,
        ) as mock_summarize:
            await process_item(queue_item)

        mock_summarize.assert_awaited_once()
        # The strip must happen on a copy: the counter has to stay on the row so
        # the budget still advances if this attempt fails again.
        assert raw[RETRY_ATTEMPTS_PAYLOAD_KEY] == 1
