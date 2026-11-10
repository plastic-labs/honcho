"""A deep representation unit overflows into chunk units that drain in parallel."""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.deriver.enqueue import enqueue
from src.deriver.queue_manager import QueueManager, WorkerOwnership
from src.utils.work_unit import (
    base_work_unit_key,
    construct_work_unit_key,
    parse_work_unit_key,
)

SessionWithPeers = tuple[models.Session, list[models.Peer]]


# --- key format ---------------------------------------------------------------


def test_chunk_key_round_trips() -> None:
    payload = {"task_type": "representation", "session_name": "s", "observed": "bob"}
    base = construct_work_unit_key("ws", payload)
    chunk = construct_work_unit_key("ws", payload, chunk=12)

    assert base == "representation:ws:s:bob"
    assert chunk == "representation:ws:s:bob:chunk:12"

    parsed = parse_work_unit_key(chunk)
    assert (parsed.workspace_name, parsed.session_name, parsed.observed) == (
        "ws",
        "s",
        "bob",
    )
    assert parsed.chunk == 12
    assert parse_work_unit_key(base).chunk is None
    # A parsed key builds the same key again.
    assert construct_work_unit_key("ws", parsed) == chunk
    assert base_work_unit_key(chunk) == base
    assert base_work_unit_key(base) == base


def test_chunk_key_is_distinct_from_the_legacy_observer_format() -> None:
    """Five parts stay the legacy observer:observed key, even for a peer named chunk."""
    legacy = parse_work_unit_key("representation:ws:s:chunk:bob")
    assert (legacy.observer, legacy.observed, legacy.chunk) == ("chunk", "bob", None)

    observed_named_chunk = parse_work_unit_key("representation:ws:s:chunk:chunk:3")
    assert (observed_named_chunk.observed, observed_named_chunk.chunk) == ("chunk", 3)


def test_malformed_chunk_keys_are_rejected() -> None:
    for key in (
        "representation:ws:s:bob:chunk:x",
        "representation:ws:s:bob:part:3",
        "representation:ws:s:bob:chunk:3:4",
    ):
        with pytest.raises(ValueError):
            parse_work_unit_key(key)


def test_only_representation_units_can_be_chunked() -> None:
    summary = {
        "task_type": "summary",
        "session_name": "s",
        "observer": "a",
        "observed": "b",
    }
    with pytest.raises(ValueError, match="only representation"):
        construct_work_unit_key("ws", summary, chunk=1)


def test_tenant_prefixed_chunk_key_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    payload = {"task_type": "representation", "session_name": "s", "observed": "bob"}
    key = construct_work_unit_key("ws", payload, tenant_id="acme", chunk=2)

    assert key == "acme:representation:ws:s:bob:chunk:2"
    parsed = parse_work_unit_key(key)
    assert (parsed.tenant_id, parsed.observed, parsed.chunk) == ("acme", "bob", 2)
    assert base_work_unit_key(key) == "acme:representation:ws:s:bob"


# --- enqueue routing ----------------------------------------------------------


async def _add_messages(
    db_session: AsyncSession, session: models.Session, peer_names: list[str]
) -> list[models.Message]:
    """Append one message per name to the session, continuing its sequence."""
    last_seq = (
        await db_session.scalar(
            select(models.Message.seq_in_session)
            .where(
                models.Message.workspace_name == session.workspace_name,
                models.Message.session_name == session.name,
            )
            .order_by(models.Message.seq_in_session.desc())
            .limit(1)
        )
        or 0
    )
    messages = [
        models.Message(
            session_name=session.name,
            workspace_name=session.workspace_name,
            peer_name=peer_name,
            content=f"message {last_seq + offset}",
            token_count=10,
            seq_in_session=last_seq + offset,
        )
        for offset, peer_name in enumerate(peer_names, start=1)
    ]
    db_session.add_all(messages)
    await db_session.commit()
    for message in messages:
        await db_session.refresh(message)
    return messages


def _enqueue_payload(messages: list[models.Message]) -> list[dict[str, Any]]:
    return [
        {
            "workspace_name": message.workspace_name,
            "session_name": message.session_name,
            "message_id": message.id,
            "content": message.content,
            "peer_name": message.peer_name,
            "created_at": message.created_at,
            "message_public_id": message.public_id,
            "seq_in_session": message.seq_in_session,
            "configuration": None,
        }
        for message in messages
    ]


async def _chunk_by_seq(
    db_session: AsyncSession, session: models.Session
) -> dict[int, int | None]:
    """Each queued message's sequence number mapped to the chunk it was routed to."""
    rows = await db_session.execute(
        select(models.Message.seq_in_session, models.QueueItem.work_unit_key)
        .join(models.QueueItem, models.QueueItem.message_id == models.Message.id)
        .where(
            models.QueueItem.session_id == session.id,
            models.QueueItem.task_type == "representation",
        )
    )
    return {seq: parse_work_unit_key(key).chunk for seq, key in rows.all()}


def _chunking(monkeypatch: pytest.MonkeyPatch, *, threshold: int, size: int) -> None:
    monkeypatch.setattr(
        settings.DERIVER, "REPRESENTATION_CHUNK_THRESHOLD_MESSAGES", threshold
    )
    monkeypatch.setattr(settings.DERIVER, "REPRESENTATION_CHUNK_MESSAGES", size)


async def test_chunking_is_off_by_default(
    db_session: AsyncSession, sample_session_with_peers: SessionWithPeers
) -> None:
    session, peers = sample_session_with_peers
    messages = await _add_messages(db_session, session, [peers[0].name] * 6)

    await enqueue(_enqueue_payload(messages), tenant_id=None)

    assert set((await _chunk_by_seq(db_session, session)).values()) == {None}


async def test_messages_past_the_threshold_go_to_chunks(
    db_session: AsyncSession,
    sample_session_with_peers: SessionWithPeers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base unit fills to the threshold; later messages are keyed by position."""
    _chunking(monkeypatch, threshold=3, size=2)
    session, peers = sample_session_with_peers
    messages = await _add_messages(db_session, session, [peers[0].name] * 7)

    await enqueue(_enqueue_payload(messages), tenant_id=None)

    assert await _chunk_by_seq(db_session, session) == {
        1: None,
        2: None,
        3: None,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
    }


async def test_depth_carries_across_calls_and_recovers_after_a_drain(
    db_session: AsyncSession,
    sample_session_with_peers: SessionWithPeers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later call reads the pending depth; a drained base unit takes messages again."""
    _chunking(monkeypatch, threshold=3, size=100)
    session, peers = sample_session_with_peers
    name = peers[0].name

    await enqueue(
        _enqueue_payload(await _add_messages(db_session, session, [name] * 3)),
        tenant_id=None,
    )
    await enqueue(
        _enqueue_payload(await _add_messages(db_session, session, [name])),
        tenant_id=None,
    )
    routed = await _chunk_by_seq(db_session, session)
    assert routed == {1: None, 2: None, 3: None, 4: 0}

    # Drain the base unit: its batch row goes away with its last pending item.
    await db_session.execute(
        update(models.QueueItem)
        .where(
            models.QueueItem.session_id == session.id,
            models.QueueItem.work_unit_key.not_like("%:chunk:%"),
        )
        .values(processed=True)
    )
    await db_session.commit()

    await enqueue(
        _enqueue_payload(await _add_messages(db_session, session, [name])),
        tenant_id=None,
    )
    assert (await _chunk_by_seq(db_session, session))[5] is None


# --- draining -----------------------------------------------------------------


async def _queue_items(
    db_session: AsyncSession,
    create_queue_payload: Callable[..., Any],
    session: models.Session,
    messages: list[models.Message],
    *,
    chunk: int | None,
) -> str:
    """Queue one representation item per message on the base unit or a chunk."""
    work_unit_key = ""
    for message in messages:
        payload = create_queue_payload(
            message=message,
            task_type="representation",
            observed=message.peer_name,
            observer=message.peer_name,
        )
        work_unit_key = construct_work_unit_key(
            session.workspace_name, payload, chunk=chunk
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
    return work_unit_key


async def _claim(
    qm: QueueManager, db_session: AsyncSession, work_unit_key: str, worker_id: str
) -> str:
    claimed = await qm.claim_work_units(db_session, [work_unit_key])
    await db_session.commit()
    qm.worker_ownership[worker_id] = WorkerOwnership(
        work_unit_key=work_unit_key, aqs_id=claimed[work_unit_key]
    )
    return claimed[work_unit_key]


async def test_a_chunk_drains_its_own_items_with_the_session_as_context(
    db_session: AsyncSession,
    sample_session_with_peers: SessionWithPeers,
    create_queue_payload: Callable[..., Any],
) -> None:
    """The batch fetch is scoped to the chunk's items but still reads the message before it."""
    session, peers = sample_session_with_peers
    observed, other = peers[0].name, peers[1].name
    first, reply, third, fourth = await _add_messages(
        db_session, session, [observed, other, observed, observed]
    )
    await _queue_items(db_session, create_queue_payload, session, [first], chunk=None)
    chunk_key = await _queue_items(
        db_session, create_queue_payload, session, [third, fourth], chunk=1
    )

    qm = QueueManager()
    aqs_id = await _claim(qm, db_session, chunk_key, "chunk_worker")
    batch = await qm.get_queue_item_batch("representation", chunk_key, aqs_id)

    assert [item.message_id for item in batch.items_to_process] == [
        third.id,
        fourth.id,
    ]
    # The other peer's reply is not this chunk's item; it is there as context.
    assert [message.id for message in batch.messages_context] == [
        reply.id,
        third.id,
        fourth.id,
    ]


async def test_one_poll_claims_the_base_unit_and_its_chunks(
    db_session: AsyncSession,
    sample_session_with_peers: SessionWithPeers,
    create_queue_payload: Callable[..., Any],
    batch_gate_settings: Callable[..., None],
) -> None:
    """Chunks are separate work units, so free workers take them at the same time."""
    batch_gate_settings(flush_enabled=True, workers=4)
    session, peers = sample_session_with_peers
    messages = await _add_messages(db_session, session, [peers[0].name] * 3)
    keys = {
        await _queue_items(
            db_session, create_queue_payload, session, [message], chunk=chunk
        )
        for message, chunk in zip(messages, (None, 1, 2), strict=True)
    }

    claimed = await QueueManager().get_and_claim_work_units()

    assert set(claimed) == keys
    assert len(keys) == 3


async def test_queue_empty_waits_for_every_chunk(
    db_session: AsyncSession,
    sample_session_with_peers: SessionWithPeers,
    create_queue_payload: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """queue.empty speaks for the session and peer: it fires when the last sibling drains."""
    _chunking(monkeypatch, threshold=1, size=100)
    session, peers = sample_session_with_peers
    first, second, third = await _add_messages(db_session, session, [peers[0].name] * 3)
    base_key = await _queue_items(
        db_session, create_queue_payload, session, [first], chunk=None
    )
    chunk_keys = [
        await _queue_items(
            db_session, create_queue_payload, session, [message], chunk=chunk
        )
        for message, chunk in ((second, 1), (third, 2))
    ]

    async def noop_batch(*_args: Any, **_kwargs: Any) -> None:
        return None

    qm = QueueManager()
    publish = AsyncMock()
    with (
        patch(
            "src.deriver.queue_manager.process_representation_batch",
            side_effect=noop_batch,
        ),
        patch("src.deriver.queue_manager.publish_webhook_event", publish),
    ):
        # A chunk drains while the base unit and the other chunk are pending.
        await _claim(qm, db_session, chunk_keys[0], "worker_a")
        await qm.process_work_unit(chunk_keys[0], "worker_a")
        publish.assert_not_awaited()

        # The base unit drains while a chunk is still pending.
        await _claim(qm, db_session, base_key, "worker_b")
        await qm.process_work_unit(base_key, "worker_b")
        publish.assert_not_awaited()

        # The last sibling drains.
        await _claim(qm, db_session, chunk_keys[1], "worker_c")
        await qm.process_work_unit(chunk_keys[1], "worker_c")
        publish.assert_awaited_once()

    assert publish.await_args is not None
    event = publish.await_args.args[0]
    assert event.session_id == session.name
    assert event.observed == peers[0].name
