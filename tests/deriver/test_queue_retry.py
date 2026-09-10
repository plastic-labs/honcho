"""Queue-level retry behaviour: transient failures and empty structured responses.

Two related loss modes share one mechanism:

* a transient failure (DB deadlock, lost connection, provider transport) used to
  burn one queue item per cycle instead of letting the work unit be re-claimed;
* an empty structured response (`EmptyRepresentationError`) used to be consumed
  outright -- the items were marked processed with nothing derived from them.

Both are now retryable under a bounded per-work-unit budget that lives in
``queue.payload`` (no schema change), and both eventually fail *visibly*.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.deriver.consumer import process_item
from src.deriver.queue_manager import (
    MAX_RETRYABLE_ATTEMPTS,
    QueueManager,
    WorkerOwnership,
)
from src.exceptions import EmptyRepresentationError, EmptySummaryError
from src.utils.queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY, SummaryPayload
from src.utils.work_unit import construct_work_unit_key


class _QueueRetryTestBase:
    """Shared seeding/introspection helpers for the retry tests."""

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
        raw = (unprocessed[0].payload or {}).get(RETRY_ATTEMPTS_PAYLOAD_KEY)
        return None if raw is None else int(raw)


@pytest.mark.asyncio
class TestQueueRetry(_QueueRetryTestBase):
    """Bounded retry of transient errors in process_work_unit (DEV-1975).

    A transient failure (deadlock, lost connection, provider transport) must
    leave the batch's queue items unprocessed and release the work unit for
    re-claim, up to MAX_RETRYABLE_ATTEMPTS per work unit; terminal failures
    keep today's burn-one-item behavior.
    """

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


@pytest.mark.asyncio
class TestEmptyRepresentationRequeue(_QueueRetryTestBase):
    """An empty structured response must not consume its work unit.

    This is the queue-level half of the loss mode: the deriver refuses a
    degraded empty parse (`EmptyRepresentationError`), and the queue turns that
    into a bounded requeue rather than a silent burn.
    """

    async def test_degraded_empty_representation_requeues_batch(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First exhaustion: every item stays unprocessed, nothing errored."""
        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload, n_messages=2
        )

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=EmptyRepresentationError(
                "empty representation after 2 attempts"
            ),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 2
        assert all(not item.processed for item in items), (
            "a refused empty parse must not consume any item"
        )
        assert all(item.error is None for item in items)
        assert await self._retry_attempts_on_items(db_session, work_unit_key) == 1
        assert await self._aqs_rows(db_session, work_unit_key) == 0

    async def test_exhausted_empty_representation_fails_the_whole_batch(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At the cap the *whole* batch is marked errored, loudly.

        Not just the first item: a degraded response poisons every item it saw,
        and re-attempting the rest one at a time would just replay the same
        failure with a fresh budget each cycle.
        """
        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload, n_messages=3
        )
        await qm._set_work_unit_retry_attempts(  # pyright: ignore[reportPrivateUsage]
            work_unit_key, MAX_RETRYABLE_ATTEMPTS - 1
        )

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=EmptyRepresentationError(
                "empty representation after 2 attempts"
            ),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 3
        assert all(item.processed for item in items)
        assert all(item.error is not None for item in items)
        assert all(
            "EmptyRepresentationError" in item.error for item in items if item.error
        )
        assert await self._retry_attempts_on_items(db_session, work_unit_key) is None

    async def test_success_after_a_retry_consumes_the_batch_exactly_once(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The reclaim that succeeds marks the batch processed once, no error."""
        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload, n_messages=2
        )

        async def noop_batch(*_args: Any, **_kwargs: Any) -> None:
            return None

        # Attempt 1 (the work unit the seed already claimed): refused.
        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=EmptyRepresentationError(
                "empty representation after 2 attempts"
            ),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        assert await self._retry_attempts_on_items(db_session, work_unit_key) == 1

        # Attempt 2: the work unit is claimable again and this time succeeds.
        claimed = await qm.claim_work_units(db_session, [work_unit_key])
        assert claimed, "a refused batch must be claimable again"
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=claimed[work_unit_key]
        )
        await db_session.commit()

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=noop_batch,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 2
        assert all(item.processed for item in items)
        assert all(item.error is None for item in items)

    async def test_retry_counter_does_not_leak_into_the_next_batch(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
    ) -> None:
        """A later batch of the same work unit starts with a full budget.

        The counter is stored on the oldest unprocessed item, so once that item
        is processed the budget resets -- no success-path cleanup is needed and
        a stale key on a processed row can never shorten a later batch's budget.
        """
        qm, work_unit_key, worker_id, _ = await self._seed_work_unit(
            db_session, sample_session_with_peers, create_queue_payload
        )
        await qm._set_work_unit_retry_attempts(work_unit_key, 2)  # pyright: ignore[reportPrivateUsage]

        async def noop_batch(*_args: Any, **_kwargs: Any) -> None:
            return None

        with patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=noop_batch,
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        # A new item arrives for the same work unit (the next batch).
        session, peers = sample_session_with_peers
        message = models.Message(
            session_name=session.name,
            workspace_name=session.workspace_name,
            peer_name=peers[0].name,
            content="later message",
            token_count=10,
            seq_in_session=99,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)
        payload = create_queue_payload(
            message=message,
            task_type="representation",
            observed=peers[0].name,
            observer=peers[0].name,
        )
        db_session.add(
            models.QueueItem(
                session_id=session.id,
                task_type="representation",
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=session.workspace_name,
                message_id=message.id,
            )
        )
        await db_session.commit()

        assert await qm._get_work_unit_retry_attempts(work_unit_key) == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
class TestSummaryRetry(_QueueRetryTestBase):
    """The summarizer's twin of the empty-representation loss.

    `_create_summary` raises EmptySummaryError when a summary comes back empty
    in a way that looks degraded; nothing is persisted for that boundary when it
    falls back to a placeholder, so the item has to be re-claimed instead.
    """

    async def test_summary_item_is_requeued_when_the_summarizer_refuses(
        self,
        db_session: AsyncSession,
        sample_session_with_peers: tuple[models.Session, list[models.Peer]],
        create_queue_payload: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("src.deriver.queue_manager.RETRY_BACKOFF_SECONDS", 0.0)
        session, peers = sample_session_with_peers
        message = models.Message(
            session_name=session.name,
            workspace_name=session.workspace_name,
            peer_name=peers[0].name,
            content="Message for the summary",
            token_count=10,
            seq_in_session=1,
        )
        db_session.add(message)
        await db_session.commit()
        await db_session.refresh(message)

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
        await db_session.refresh(queue_item)

        qm = QueueManager()
        worker_id = "test_worker"
        claimed = await qm.claim_work_units(db_session, [work_unit_key])
        qm.worker_ownership[worker_id] = WorkerOwnership(
            work_unit_key=work_unit_key, aqs_id=claimed[work_unit_key]
        )
        await db_session.commit()

        with patch(
            "src.deriver.consumer.summarizer.summarize_if_needed",
            side_effect=EmptySummaryError("empty summary (finish_reasons=['length'])"),
        ):
            await qm.process_work_unit(work_unit_key, worker_id)

        items = await self._fetch_items(db_session, work_unit_key)
        assert len(items) == 1
        assert not items[0].processed, "a refused summary must not be consumed"
        assert items[0].error is None
        assert await self._retry_attempts_on_items(db_session, work_unit_key) == 1
        assert await self._aqs_rows(db_session, work_unit_key) == 0
