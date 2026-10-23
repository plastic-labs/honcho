from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import Select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.pending_work import (
    pending_work_for_message,
    pending_work_for_peer,
    pending_work_for_session,
)

pytestmark = pytest.mark.asyncio


class _Queue:
    """Pending and processed queue rows across two workspaces, keyed by label."""

    def __init__(self, workspace: str, other_workspace: str, message_id: int):
        self.workspace: str = workspace
        self.other_workspace: str = other_workspace
        self.message_id: int = message_id
        self.ids: dict[str, int] = {}


def _item(
    workspace_name: str | None,
    task_type: str,
    payload: dict[str, Any],
    *,
    processed: bool = False,
    message_id: int | None = None,
) -> models.QueueItem:
    return models.QueueItem(
        work_unit_key=str(generate_nanoid()),
        task_type=task_type,
        payload=payload,
        processed=processed,
        workspace_name=workspace_name,
        message_id=message_id,
    )


@pytest.fixture
async def queue(db_session: AsyncSession) -> _Queue:
    workspace = models.Workspace(name=str(generate_nanoid()))
    other_workspace = models.Workspace(name=str(generate_nanoid()))
    db_session.add_all([workspace, other_workspace])
    await db_session.flush()
    db_session.add_all(
        [
            models.Peer(name=name, workspace_name=workspace.name)
            for name in ("alice", "bob", "carol", "dave")
        ]
    )
    session = models.Session(name="s1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()
    message = models.Message(
        session_name="s1",
        content="hi",
        token_count=1,
        seq_in_session=1,
        peer_name="carol",
        workspace_name=workspace.name,
    )
    db_session.add(message)
    await db_session.flush()

    representation = {
        "task_type": "representation",
        "session_name": "s1",
        "observers": ["alice", "bob"],
        "observed": "carol",
    }
    items = {
        "representation": _item(
            workspace.name, "representation", representation, message_id=message.id
        ),
        "processed": _item(
            workspace.name,
            "representation",
            representation,
            processed=True,
            message_id=message.id,
        ),
        "summary": _item(
            workspace.name,
            "summary",
            {"task_type": "summary", "session_name": "s1"},
            message_id=message.id,
        ),
        "dream": _item(
            workspace.name,
            "dream",
            {
                "task_type": "dream",
                "observer": "alice",
                "observed": "dave",
                "session_name": "s2",
            },
        ),
        "scope": _item(
            workspace.name,
            "scope_backfill",
            {
                "task_type": "scope_backfill",
                "scope_peer": "alice",
                "session_name": "s1",
            },
        ),
        "webhook": _item(
            workspace.name, "webhook", {"task_type": "webhook", "data": {}}
        ),
        "other_workspace": _item(
            other_workspace.name,
            "representation",
            {**representation, "observers": ["alice"]},
        ),
        # Tenant-less reconciler lane; must never be attributed to a workspace
        "global": _item(
            None,
            "reconciler",
            {"observed": "alice", "observers": ["alice"], "session_name": "s1"},
        ),
    }
    db_session.add_all(items.values())
    await db_session.flush()

    result = _Queue(workspace.name, other_workspace.name, message.id)
    result.ids = {label: item.id for label, item in items.items()}
    return result


async def _labels(
    db: AsyncSession, queue: _Queue, stmt: Select[tuple[models.QueueItem]]
) -> list[str]:
    by_id = {item_id: label for label, item_id in queue.ids.items()}
    return [by_id[item.id] for item in (await db.scalars(stmt)).all()]


@pytest.mark.parametrize(
    ("peer", "expected"),
    [
        ("alice", ["representation", "dream", "scope"]),
        ("bob", ["representation"]),
        ("carol", ["representation"]),
        ("dave", ["dream"]),
        ("erin", []),
    ],
)
async def test_pending_work_for_peer(
    db_session: AsyncSession, queue: _Queue, peer: str, expected: list[str]
) -> None:
    stmt = pending_work_for_peer(queue.workspace, peer)

    assert await _labels(db_session, queue, stmt) == expected


@pytest.mark.parametrize(
    ("session_name", "expected"),
    [
        ("s1", ["representation", "summary", "scope"]),
        ("s2", ["dream"]),
        ("s3", []),
    ],
)
async def test_pending_work_for_session(
    db_session: AsyncSession, queue: _Queue, session_name: str, expected: list[str]
) -> None:
    stmt = pending_work_for_session(queue.workspace, session_name)

    assert await _labels(db_session, queue, stmt) == expected


async def test_pending_work_for_message(
    db_session: AsyncSession, queue: _Queue
) -> None:
    stmt = pending_work_for_message(queue.workspace, queue.message_id)

    assert await _labels(db_session, queue, stmt) == ["representation", "summary"]


async def test_lookups_are_scoped_to_the_workspace(
    db_session: AsyncSession, queue: _Queue
) -> None:
    assert await _labels(
        db_session, queue, pending_work_for_peer(queue.other_workspace, "alice")
    ) == ["other_workspace"]
    assert await _labels(
        db_session, queue, pending_work_for_session(queue.other_workspace, "s1")
    ) == ["other_workspace"]
    assert (
        await _labels(
            db_session,
            queue,
            pending_work_for_message(queue.other_workspace, queue.message_id),
        )
        == []
    )


@pytest.mark.parametrize(
    ("stmt", "indexes"),
    [
        (
            pending_work_for_peer("w", "alice"),
            {
                "ix_queue_pending_observers",
                "ix_queue_pending_observed",
                "ix_queue_pending_observer",
                "ix_queue_pending_scope_peer",
            },
        ),
        (pending_work_for_session("w", "s1"), {"ix_queue_pending_session_name"}),
    ],
)
async def test_lookups_can_use_the_pending_indexes(
    db_session: AsyncSession,
    stmt: Select[tuple[models.QueueItem]],
    indexes: set[str],
) -> None:
    # With every other queue index gone and seq scans disabled, a lookup whose
    # expressions don't match the partial indexes plans a (disabled) seq scan.
    # The generic plan keeps parameters unbound, as a reused prepared statement does.
    schema = settings.DB.SCHEMA
    other_indexes = (
        await db_session.scalars(
            text(
                "SELECT indexname FROM pg_indexes WHERE schemaname = :schema"
                + " AND tablename = 'queue' AND indexname NOT LIKE 'ix_queue_pending_%'"
            ),
            {"schema": schema},
        )
    ).all()
    for name in other_indexes:
        if name == "pk_queue":
            await db_session.execute(
                text(f'ALTER TABLE "{schema}".queue DROP CONSTRAINT pk_queue')
            )
        else:
            await db_session.execute(text(f'DROP INDEX "{schema}"."{name}"'))
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))
    await db_session.execute(text("SET LOCAL plan_cache_mode = force_generic_plan"))

    compiled = stmt.compile(dialect=postgresql.asyncpg.dialect())
    args = ", ".join(
        f"'{compiled.params[name]}'" for name in compiled.positiontup or []
    )
    await db_session.execute(text(f"PREPARE pending_lookup AS {compiled}"))
    try:
        plan = "\n".join(
            (
                await db_session.scalars(
                    text(f"EXPLAIN EXECUTE pending_lookup({args})")
                )
            ).all()
        )
    finally:
        await db_session.execute(text("DEALLOCATE pending_lookup"))

    assert "Seq Scan" not in plan, plan
    for index in indexes:
        assert index in plan, plan
