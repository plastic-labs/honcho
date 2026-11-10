"""The deriver's save runs behind the lifecycle barrier.

Each test runs a real work unit through ``QueueManager.process_work_unit``. The
model call is replaced by a function that first lets a deletion land, as it
would while the worker waits on the provider, and then returns conclusions.
The worker holds its input in memory by then, so only the final write can stop
it from landing under the deleted parent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import crud, models
from src.deriver import queue_manager
from src.deriver.queue_manager import QueueManager
from src.llm import HonchoLLMCallResponse
from src.utils.representation import ExplicitObservationBase, PromptRepresentation
from tests.deriver.conftest import independent_sessions

CONCLUSION = "alice drinks green tea"
CONTEXT_TEXT = "bob asks what alice drinks"

DuringModelCall = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class Unit:
    workspace: str
    session: str
    session_id: str
    observed: str
    observer: str
    work_unit_key: str
    context_message_id: int
    queued_message_id: int
    queue_item_id: int


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue_manager, "RETRY_BACKOFF_SECONDS", 0.0)


async def _seed_unit(db: AsyncSession) -> Unit:
    """One queued message from ``alice`` after a context-only message from ``bob``.

    ``alice`` observes herself and ``carol`` observes her, so the batch saves
    to two collections. ``bob`` only wrote the context message, which has no
    queue row of its own.
    """
    workspace = models.Workspace(name=str(generate_nanoid()))
    db.add(workspace)
    await db.flush()
    alice = models.Peer(name="alice", workspace_name=workspace.name)
    bob = models.Peer(name="bob", workspace_name=workspace.name)
    carol = models.Peer(name="carol", workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db.add_all([alice, bob, carol, session])
    await db.flush()
    context = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=bob.name,
        content=CONTEXT_TEXT,
        token_count=5,
        seq_in_session=1,
    )
    db.add(context)
    await db.flush()
    queued = models.Message(
        workspace_name=workspace.name,
        session_name=session.name,
        peer_name=alice.name,
        content="green tea, every morning",
        token_count=5,
        seq_in_session=2,
    )
    db.add(queued)
    await db.flush()
    work_unit_key = f"representation:{workspace.name}:{session.name}:{alice.name}"
    item = models.QueueItem(
        session_id=session.id,
        work_unit_key=work_unit_key,
        task_type="representation",
        payload={"observers": [alice.name, carol.name], "observed": alice.name},
        workspace_name=workspace.name,
        message_id=queued.id,
    )
    db.add(item)
    await db.flush()
    unit = Unit(
        workspace=workspace.name,
        session=session.name,
        session_id=session.id,
        observed=alice.name,
        observer=carol.name,
        work_unit_key=work_unit_key,
        context_message_id=context.id,
        queued_message_id=queued.id,
        queue_item_id=item.id,
    )
    await db.commit()
    return unit


async def _run_unit(
    db: AsyncSession, unit: Unit, during_model_call: DuringModelCall | None = None
) -> list[str]:
    """Claim the unit and process it once. Returns the prompts the model saw."""
    prompts: list[str] = []
    pending = [during_model_call] if during_model_call is not None else []

    async def model_call(
        *_: Any, prompt: str, **_kwargs: Any
    ) -> HonchoLLMCallResponse[Any]:
        prompts.append(prompt)
        if pending:
            await pending.pop()()
        return HonchoLLMCallResponse(
            content=PromptRepresentation(
                explicit=[ExplicitObservationBase(content=CONCLUSION)]
            ),
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

    manager = QueueManager()
    claimed = await manager.claim_work_units(db, [unit.work_unit_key])
    await db.commit()
    manager.track_worker_work_unit(
        "worker", unit.work_unit_key, claimed[unit.work_unit_key]
    )
    with patch("src.deriver.deriver.honcho_llm_call", model_call):
        await manager.process_work_unit(unit.work_unit_key, "worker")
    return prompts


async def _conclusions(db: AsyncSession, unit: Unit) -> list[tuple[str, str]]:
    """``(observer, content)`` of every live conclusion in the workspace."""
    rows = await db.execute(
        select(models.Document.observer, models.Document.content)
        .where(
            models.Document.workspace_name == unit.workspace,
            models.Document.deleted_at.is_(None),
        )
        .order_by(models.Document.observer)
    )
    return [(row.observer, row.content) for row in rows]


async def _count(db: AsyncSession, model: Any, *where: Any) -> int:
    return int(
        await db.scalar(select(func.count()).select_from(model).where(*where)) or 0
    )


async def _queue_item(db: AsyncSession, unit: Unit) -> tuple[bool, str | None] | None:
    row = (
        await db.execute(
            select(models.QueueItem.processed, models.QueueItem.error).where(
                models.QueueItem.id == unit.queue_item_id
            )
        )
    ).one_or_none()
    return None if row is None else (row.processed, row.error)


async def test_an_undisturbed_batch_saves_for_every_observer(
    db_session: AsyncSession,
) -> None:
    unit = await _seed_unit(db_session)
    prompts = await _run_unit(db_session, unit)

    assert len(prompts) == 1
    assert CONTEXT_TEXT in prompts[0]
    assert await _conclusions(db_session, unit) == [
        (unit.observed, CONCLUSION),
        (unit.observer, CONCLUSION),
    ]
    assert await _queue_item(db_session, unit) == (True, None)


async def test_a_session_deleted_during_the_model_call_gets_no_conclusions(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The live case: deletion removes the worker's claim, and the worker writes anyway."""
    unit = await _seed_unit(db_session)

    async def delete_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, unit.workspace, unit.session)

    prompts = await _run_unit(db_session, unit, delete_the_session)

    assert len(prompts) == 1
    assert await _conclusions(db_session, unit) == []
    # The purged name was not recreated, and no collection was left behind.
    assert (
        await _count(db_session, models.Session, models.Session.name == unit.session)
        == 0
    )
    assert (
        await _count(
            db_session,
            models.Collection,
            models.Collection.workspace_name == unit.workspace,
        )
        == 0
    )
    assert await _queue_item(db_session, unit) is None
    assert await _count(db_session, models.ActiveQueueSession) == 0


async def test_a_workspace_deleted_during_the_model_call_gets_no_conclusions(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    unit = await _seed_unit(db_session)

    async def delete_the_workspace() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_workspace(db, unit.workspace)

    await _run_unit(db_session, unit, delete_the_workspace)

    assert await _conclusions(db_session, unit) == []
    assert (
        await _count(
            db_session, models.Workspace, models.Workspace.name == unit.workspace
        )
        == 0
    )
    assert (
        await _count(
            db_session, models.Peer, models.Peer.workspace_name == unit.workspace
        )
        == 0
    )


async def test_a_session_recreated_under_the_same_name_gets_no_conclusions(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The batch was read from the old session. The new one must stay empty."""
    unit = await _seed_unit(db_session)

    async def delete_and_recreate_the_session() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_session(db, unit.workspace, unit.session)
            db.add(models.Session(name=unit.session, workspace_name=unit.workspace))
            await db.commit()

    await _run_unit(db_session, unit, delete_and_recreate_the_session)

    assert await _conclusions(db_session, unit) == []
    assert (
        await _count(db_session, models.Session, models.Session.name == unit.session)
        == 1
    )


async def test_a_context_message_deleted_during_the_model_call_is_recomputed_without_it(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The context message has no queue row, so removing queued work cannot reach it."""
    unit = await _seed_unit(db_session)

    async def delete_the_context_message() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                update(models.Message)
                .where(
                    models.Message.tenant_id == models.DEFAULT_TENANT_ID,
                    models.Message.id == unit.context_message_id,
                )
                .values(deleted_at=datetime.now(UTC))
            )
            await db.commit()

    first = await _run_unit(db_session, unit, delete_the_context_message)

    # Refused: the conclusions were derived from a prompt holding deleted text.
    assert CONTEXT_TEXT in first[0]
    assert await _conclusions(db_session, unit) == []
    assert await _queue_item(db_session, unit) == (False, None)

    # Recomputed from live input only.
    second = await _run_unit(db_session, unit)
    assert len(second) == 1
    assert CONTEXT_TEXT not in second[0]
    assert len(await _conclusions(db_session, unit)) == 2
    assert await _queue_item(db_session, unit) == (True, None)


async def test_a_deleted_queued_message_is_dropped_without_a_model_call(
    db_session: AsyncSession,
) -> None:
    unit = await _seed_unit(db_session)
    await db_session.execute(
        update(models.Message)
        .where(
            models.Message.tenant_id == models.DEFAULT_TENANT_ID,
            models.Message.id == unit.queued_message_id,
        )
        .values(deleted_at=datetime.now(UTC))
    )
    await db_session.commit()

    prompts = await _run_unit(db_session, unit)

    assert prompts == []
    assert await _conclusions(db_session, unit) == []
    assert await _queue_item(db_session, unit) == (True, None)


async def test_an_observer_deleted_during_the_model_call_is_left_out(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """Peers have no generation. Liveness is checked for observer and observed."""
    unit = await _seed_unit(db_session)

    async def delete_the_observer() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                update(models.Peer)
                .where(
                    models.Peer.workspace_name == unit.workspace,
                    models.Peer.name == unit.observer,
                )
                .values(deleted_at=datetime.now(UTC))
            )
            await db.commit()

    await _run_unit(db_session, unit, delete_the_observer)

    # The observed peer's own representation is unaffected.
    assert await _conclusions(db_session, unit) == [(unit.observed, CONCLUSION)]
    assert (
        await _count(
            db_session,
            models.Collection,
            models.Collection.workspace_name == unit.workspace,
            models.Collection.observer == unit.observer,
        )
        == 0
    )
    assert await _queue_item(db_session, unit) == (True, None)


async def test_the_observed_peer_deleted_during_the_model_call_gets_no_conclusions(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    unit = await _seed_unit(db_session)

    async def delete_the_observed_peer() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                update(models.Peer)
                .where(
                    models.Peer.workspace_name == unit.workspace,
                    models.Peer.name == unit.observed,
                )
                .values(deleted_at=datetime.now(UTC))
            )
            await db.commit()

    await _run_unit(db_session, unit, delete_the_observed_peer)

    assert await _conclusions(db_session, unit) == []
    # Dropped, not failed: a deleted parent is not an error of the work.
    assert await _queue_item(db_session, unit) == (True, None)


async def test_a_worker_that_lost_its_claim_writes_nothing_and_leaves_the_work(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """A reaped claim may already belong to another worker, which will write."""
    unit = await _seed_unit(db_session)

    async def revoke_the_claim() -> None:
        async with independent_sessions(db_engine)() as db:
            await db.execute(
                delete(models.ActiveQueueSession).where(
                    models.ActiveQueueSession.work_unit_key == unit.work_unit_key
                )
            )
            await db.commit()

    await _run_unit(db_session, unit, revoke_the_claim)

    assert await _conclusions(db_session, unit) == []
    assert await _queue_item(db_session, unit) == (False, None)


async def test_a_conclusion_erased_during_the_model_call_is_not_written_back(
    db_session: AsyncSession, db_engine: AsyncEngine
) -> None:
    """The batch re-derives a conclusion that is erased while the model runs.

    The first write is refused. The retry recomputes from the live messages.
    """
    unit = await _seed_unit(db_session)
    await _run_unit(db_session, unit)
    erased_id = await db_session.scalar(
        select(models.Document.id).where(
            models.Document.workspace_name == unit.workspace,
            models.Document.observer == unit.observed,
        )
    )
    assert isinstance(erased_id, str)
    # Queue the same message again: its batch re-derives the same conclusion.
    again = models.QueueItem(
        session_id=unit.session_id,
        work_unit_key=unit.work_unit_key,
        task_type="representation",
        payload={"observers": [unit.observed], "observed": unit.observed},
        workspace_name=unit.workspace,
        message_id=unit.queued_message_id,
    )
    db_session.add(again)
    await db_session.commit()

    async def erase_the_conclusion() -> None:
        async with independent_sessions(db_engine)() as db:
            await crud.delete_document_by_id(db, unit.workspace, erased_id)

    await _run_unit(db_session, unit, erase_the_conclusion)

    assert await _conclusions(db_session, unit) == [(unit.observer, CONCLUSION)]
    await db_session.refresh(again)
    assert not again.processed
