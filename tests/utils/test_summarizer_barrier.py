"""Both summary tiers save behind the lifecycle barrier.

A summary is stored on its session row and addressed by session name. The
summarizer reads the session and its messages, calls the model, and saves in a
new transaction. Each test lets a deletion land during the model call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.crud.lifecycle_barrier import ClaimStamp, claim_context
from src.exceptions import (
    WriteClaimLostError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)
from src.utils import summarizer
from src.utils.summarizer import SUMMARIES_KEY, Summary, SummaryType
from tests.deriver.conftest import independent_sessions

SUMMARY_TEXT = "alice and bob talked about tea"
SHORT_EVERY = 2
LONG_EVERY = 4

DuringModelCall = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Seeded:
    workspace: str
    session: str
    session_id: str
    message_ids: list[int]
    message_public_ids: list[str]


async def _seed(db: AsyncSession) -> Seeded:
    """A session with four messages: a short boundary at 2, both tiers at 4."""
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.flush()
    peer = models.Peer(name="alice", workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add_all([peer, session])
    await db.flush()
    messages = [
        models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            content=f"message {seq} about tea",
            token_count=5,
            seq_in_session=seq,
        )
        for seq in range(1, LONG_EVERY + 1)
    ]
    for message in messages:
        db.add(message)
        await db.flush()
    seeded = Seeded(
        workspace=workspace.name,
        session=session.name,
        session_id=session.id,
        message_ids=[m.id for m in messages],
        message_public_ids=[m.public_id for m in messages],
    )
    await db.commit()
    return seeded


def _configuration() -> Any:
    configuration = Mock()
    configuration.summary.enabled = True
    configuration.summary.messages_per_short_summary = SHORT_EVERY
    configuration.summary.messages_per_long_summary = LONG_EVERY
    configuration.summary.custom_instructions = None
    configuration.summary.custom_instructions_source = None
    return configuration


async def _summarize(
    seeded: Seeded, seq: int, during_model_call: DuringModelCall | None = None
) -> None:
    """Summarize at message ``seq``; the first model call lets a deletion land."""
    pending = [during_model_call] if during_model_call is not None else []

    async def create_summary(**kwargs: Any) -> tuple[Summary, bool, int, int]:
        if pending:
            await pending.pop()()
        summary_type: SummaryType = kwargs["summary_type"]
        return (
            Summary(
                content=SUMMARY_TEXT,
                message_id=kwargs["last_message_id"],
                summary_type=summary_type.value,
                created_at=datetime.now(UTC).isoformat(),
                token_count=7,
                message_public_id=kwargs["message_public_id"],
            ),
            False,
            10,
            7,
        )

    with patch.object(summarizer, "_create_summary", create_summary):
        await summarizer.summarize_if_needed(
            seeded.workspace,
            seeded.session,
            seeded.message_ids[seq - 1],
            seq,
            seeded.message_public_ids[seq - 1],
            _configuration(),
        )


async def _stored_summaries(db: AsyncSession, seeded: Seeded) -> dict[str, Any] | None:
    """The summaries on the session of this name, or None when it has no row."""
    row = (
        await db.execute(
            select(models.Session.internal_metadata).where(
                models.Session.workspace_name == seeded.workspace,
                models.Session.name == seeded.session,
            )
        )
    ).one_or_none()
    return None if row is None else row.internal_metadata.get(SUMMARIES_KEY, {})


# The message position that triggers each case: the short tier alone, or both.
TIERS = pytest.mark.parametrize(
    "seq", [SHORT_EVERY, LONG_EVERY], ids=["short-tier", "both-tiers"]
)


@TIERS
async def test_an_undisturbed_summary_is_saved(
    db_session: AsyncSession, seq: int
) -> None:
    seeded = await _seed(db_session)
    await _summarize(seeded, seq)

    stored = await _stored_summaries(db_session, seeded)
    assert stored is not None
    expected = {SummaryType.SHORT.value}
    if seq == LONG_EVERY:
        expected.add(SummaryType.LONG.value)
    assert set(stored) == expected
    assert {entry["content"] for entry in stored.values()} == {SUMMARY_TEXT}


@TIERS
async def test_a_session_deleted_during_the_model_call_gets_no_summary(
    db_session: AsyncSession, db_engine: AsyncEngine, seq: int
) -> None:
    seeded = await _seed(db_session)

    async def delete_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, seeded.workspace, seeded.session)

    with pytest.raises(WriteParentGoneError) as raised:
        await _summarize(seeded, seq, delete_the_session)

    assert raised.value.scope == "session"
    assert await _stored_summaries(db_session, seeded) is None


@TIERS
async def test_a_session_recreated_under_the_same_name_gets_no_summary(
    db_session: AsyncSession, db_engine: AsyncEngine, seq: int
) -> None:
    """The save addresses the session by name. The new session must stay empty."""
    seeded = await _seed(db_session)

    async def delete_and_recreate_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, seeded.workspace, seeded.session)
            db.add(models.Session(name=seeded.session, workspace_name=seeded.workspace))
            await db.commit()

    with pytest.raises(WriteParentGoneError):
        await _summarize(seeded, seq, delete_and_recreate_the_session)

    assert await _stored_summaries(db_session, seeded) == {}


@TIERS
async def test_a_summarized_message_deleted_during_the_model_call_is_not_summarized(
    db_session: AsyncSession, db_engine: AsyncEngine, seq: int
) -> None:
    seeded = await _seed(db_session)

    async def delete_a_message() -> None:
        # The boundary message is in the window of every tier that runs.
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                update(models.Message)
                .where(
                    models.Message.tenant_id == models.DEFAULT_TENANT_ID,
                    models.Message.id == seeded.message_ids[seq - 1],
                )
                .values(deleted_at=datetime.now(UTC))
            )
            await db.commit()

    # Stale, not gone: the queue retries it, and the retry reads live messages.
    with pytest.raises(WriteGenerationStaleError) as raised:
        await _summarize(seeded, seq, delete_a_message)

    assert raised.value.scope == "message"
    assert await _stored_summaries(db_session, seeded) == {}


async def test_a_deleted_session_is_skipped_before_the_model_call(
    db_session: AsyncSession,
) -> None:
    seeded = await _seed(db_session)
    await crud.delete_session(db_session, seeded.workspace, seeded.session)

    async def must_not_run() -> None:
        raise AssertionError("the model was called for a deleted session")

    await _summarize(seeded, SHORT_EVERY, must_not_run)
    assert await _stored_summaries(db_session, seeded) is None


async def test_a_worker_that_lost_its_claim_saves_no_summary(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The summarizer reads the running work unit's claim from the context."""
    seeded = await _seed(db_session)
    work_unit_key = f"summary:{seeded.workspace}:{seeded.session}:None:None"
    item = models.QueueItem(
        session_id=seeded.session_id,
        work_unit_key=work_unit_key,
        task_type="summary",
        payload={},
        workspace_name=seeded.workspace,
        message_id=seeded.message_ids[SHORT_EVERY - 1],
    )
    claim = models.ActiveQueueSession(work_unit_key=work_unit_key)
    db_session.add_all([item, claim])
    await db_session.commit()
    stamp = ClaimStamp(work_unit_key, claim.id, None, (item.id,))

    async def revoke_the_claim() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.id == stamp.claim_id
                )
            )
            await db.commit()

    pending = [revoke_the_claim]

    async def create_summary(**kwargs: Any) -> tuple[Summary, bool, int, int]:
        await pending.pop()()
        return (
            Summary(
                content=SUMMARY_TEXT,
                message_id=kwargs["last_message_id"],
                summary_type=kwargs["summary_type"].value,
                created_at=datetime.now(UTC).isoformat(),
                token_count=7,
                message_public_id=kwargs["message_public_id"],
            ),
            False,
            10,
            7,
        )

    with (
        patch.object(summarizer, "_create_summary", create_summary),
        pytest.raises(WriteClaimLostError),
    ):
        async with claim_context(stamp):
            await summarizer.summarize_if_needed(
                seeded.workspace,
                seeded.session,
                seeded.message_ids[SHORT_EVERY - 1],
                SHORT_EVERY,
                seeded.message_public_ids[SHORT_EVERY - 1],
                _configuration(),
                queue_item_id=item.id,
            )

    assert await _stored_summaries(db_session, seeded) == {}
