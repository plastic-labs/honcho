"""Every multi-row write to ``queue`` locks in id order or skips locked rows.

Session and workspace deletion, and a worker's batch completion, lock queue rows
in id order (``crud.deriver.queue_ids_locked_in_order``). A predicate UPDATE or
DELETE locks rows in whatever order its plan scans them, so against an
id-ordered writer over overlapping rows it can deadlock. Each multi-row writer
here therefore locks in id order, or, for the reconciler's cleanup, skips rows
another transaction holds.

The allowlist at the bottom fails when a write to ``queue`` is added or removed
without deciding how it locks.
"""

from __future__ import annotations

import ast
import asyncio
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import models
from src.deriver.queue_manager import QueueManager
from src.reconciler.queue_cleanup import cleanup_queue_items
from src.utils.queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY
from tests.deriver.conftest import SeedWorkUnit, independent_sessions

SRC = Path(__file__).resolve().parents[2] / "src"
# The table may be schema-qualified (public.queue) depending on DB_SCHEMA.
QUEUE_WRITE = re.compile(r"^(UPDATE|DELETE FROM) (\w+\.)?queue ")
ID_ORDER_LOCK = re.compile(r"ORDER BY (\w+\.)?queue\.id FOR UPDATE\)")
SKIP_LOCKED = re.compile(r"FOR UPDATE SKIP LOCKED\)")


@contextmanager
def captured_queue_writes(engine: AsyncEngine) -> Generator[list[str]]:
    """The UPDATE and DELETE statements on ``queue`` sent while the block runs."""
    statements: list[str] = []

    def record(
        _conn: Any, _cursor: Any, statement: str, *_args: Any, **_kwargs: Any
    ) -> None:
        head = " ".join(statement.split())
        if QUEUE_WRITE.match(head):
            statements.append(head)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)


async def _rows(
    db: AsyncSession, work_unit_key: str
) -> list[tuple[int, bool, str | None, dict[str, Any]]]:
    result = await db.execute(
        select(
            models.QueueItem.id,
            models.QueueItem.processed,
            models.QueueItem.error,
            models.QueueItem.payload,
        )
        .where(models.QueueItem.work_unit_key == work_unit_key)
        .order_by(models.QueueItem.id)
        .execution_options(populate_existing=True)
    )
    return [tuple(row) for row in result.all()]


@pytest.mark.asyncio
async def test_clearing_retry_attempts_locks_in_id_order(
    db_engine: AsyncEngine, db_session: AsyncSession, seed_work_unit: SeedWorkUnit
) -> None:
    key = "representation:retry-clear:unit"
    other = "representation:retry-clear:other"
    items = await seed_work_unit(key, token_counts=(1, 1, 1))
    other_items = await seed_work_unit(other, token_counts=(1,))
    await db_session.execute(
        update(models.QueueItem)
        .where(models.QueueItem.id.in_([i.id for i in items + other_items]))
        .values(payload={RETRY_ATTEMPTS_PAYLOAD_KEY: 2, "keep": True})
    )
    await db_session.commit()

    with captured_queue_writes(db_engine) as statements:
        await QueueManager()._clear_work_unit_retry_attempts(key)  # pyright: ignore[reportPrivateUsage]

    assert len(statements) == 1 and ID_ORDER_LOCK.search(statements[0])
    assert [row[3] for row in await _rows(db_session, key)] == [{"keep": True}] * 3
    # Another unit's counter is untouched.
    assert (await _rows(db_session, other))[0][3] == {
        RETRY_ATTEMPTS_PAYLOAD_KEY: 2,
        "keep": True,
    }


@pytest.mark.asyncio
async def test_quarantining_an_invalid_key_locks_in_id_order(
    db_engine: AsyncEngine, db_session: AsyncSession, seed_work_unit: SeedWorkUnit
) -> None:
    key = "not-a-valid-key"
    await seed_work_unit(key, token_counts=(1, 1))
    await seed_work_unit("representation:invalid:other", token_counts=(1,))

    with captured_queue_writes(db_engine) as statements:
        await QueueManager()._mark_invalid_work_unit_as_errored(  # pyright: ignore[reportPrivateUsage]
            key, ValueError("unparsable")
        )

    assert len(statements) == 1 and ID_ORDER_LOCK.search(statements[0])
    rows = await _rows(db_session, key)
    assert [(processed, error) for _, processed, error, _ in rows] == [
        (True, "ValueError: unparsable")
    ] * 2
    other = await _rows(db_session, "representation:invalid:other")
    assert [(processed, error) for _, processed, error, _ in other] == [(False, None)]


@pytest.mark.asyncio
async def test_failing_an_unscoped_key_locks_in_id_order(
    db_engine: AsyncEngine, db_session: AsyncSession, seed_work_unit: SeedWorkUnit
) -> None:
    key = "representation:unscoped:unit"
    await seed_work_unit(key, token_counts=(1, 1))
    already_done = await seed_work_unit(key, token_counts=(1,), processed=True)

    with captured_queue_writes(db_engine) as statements:
        await QueueManager()._fail_unscoped_work_unit(key)  # pyright: ignore[reportPrivateUsage]

    assert len(statements) == 1 and ID_ORDER_LOCK.search(statements[0])
    rows = {row[0]: row for row in await _rows(db_session, key)}
    done = rows.pop(already_done[0].id)
    assert done[1] is True and done[2] is None
    assert all(processed and error for _, processed, error, _ in rows.values())
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_cleanup_skips_a_row_another_transaction_holds(
    db_engine: AsyncEngine, db_session: AsyncSession, seed_work_unit: SeedWorkUnit
) -> None:
    key = "representation:cleanup:unit"
    held, *free = await seed_work_unit(key, token_counts=(1, 1, 1), processed=True)
    sessions = independent_sessions(db_engine)

    async with sessions() as holder:
        # Stands in for a session deletion that has locked this row.
        await holder.execute(
            text("SELECT id FROM queue WHERE id = :id FOR UPDATE"), {"id": held.id}
        )
        with captured_queue_writes(db_engine) as statements:
            # Without SKIP LOCKED this waits for the holder and times out.
            deleted = await asyncio.wait_for(cleanup_queue_items(), timeout=10)
        assert len(statements) == 1 and SKIP_LOCKED.search(statements[0])
        assert deleted == len(free)
        assert [row[0] for row in await _rows(db_session, key)] == [held.id]
        await holder.rollback()

    # The skipped row goes on the next pass.
    assert await cleanup_queue_items() == 1
    assert await _rows(db_session, key) == []


# How each write to ``queue`` in src/ locks, keyed by "module:function".
#   single_row  - matches one row by id
#   id_order    - locks its rows through queue_ids_locked_in_order
#   skip_locked - FOR UPDATE SKIP LOCKED, so it never waits
EXPECTED_QUEUE_WRITES = {
    "src/crud/deriver.py:delete_queue_items_and_claims": "id_order",
    "src/deriver/queue_manager.py:_set_work_unit_retry_attempts": "single_row",
    "src/deriver/queue_manager.py:_clear_work_unit_retry_attempts": "id_order",
    "src/deriver/queue_manager.py:_mark_invalid_work_unit_as_errored": "id_order",
    "src/deriver/queue_manager.py:mark_queue_items_as_processed": "id_order",
    "src/deriver/queue_manager.py:mark_queue_item_as_errored": "single_row",
    "src/deriver/queue_manager.py:_fail_unscoped_work_unit": "id_order",
    "src/reconciler/queue_cleanup.py:cleanup_queue_items": "skip_locked",
}

POLICY_MARKERS = {
    "single_row": "models.QueueItem.id == ",
    "id_order": "queue_ids_locked_in_order(",
    "skip_locked": "skip_locked=True",
}


def _is_queue_write(node: ast.AST) -> bool:
    """``update(models.QueueItem)`` / ``delete(models.QueueItem)``, or the bare name."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"update", "delete"}
        and node.args
    ):
        return False
    target = node.args[0]
    return (isinstance(target, ast.Attribute) and target.attr == "QueueItem") or (
        isinstance(target, ast.Name) and target.id == "QueueItem"
    )


def _queue_writes() -> dict[str, str]:
    """Each function in src/ that writes ``queue``, with its source."""
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            direct = [
                node
                for node in ast.walk(function)
                if _is_queue_write(node)
                # Attribute a write to the innermost function only.
                and not any(
                    isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef)
                    and inner is not function
                    and node in ast.walk(inner)
                    for inner in ast.walk(function)
                )
            ]
            if direct:
                site = f"{path.relative_to(SRC.parent)}:{function.name}"
                found[site] = ast.get_source_segment(source, function) or ""
    return found


def test_every_queue_write_has_a_lock_policy() -> None:
    writes = _queue_writes()
    assert set(writes) == set(EXPECTED_QUEUE_WRITES), (
        "A write to queue was added or removed. Decide how it locks (one row by "
        "id, queue_ids_locked_in_order, or SKIP LOCKED) and update "
        "EXPECTED_QUEUE_WRITES."
    )
    for site, policy in EXPECTED_QUEUE_WRITES.items():
        assert POLICY_MARKERS[policy] in writes[site], (
            f"{site} is listed as {policy} but does not lock that way"
        )
