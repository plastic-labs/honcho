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
from sqlalchemy import event, func, select, text, update
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


# How each write to ``queue`` in src/ locks, keyed by "module:function". Every
# write statement in a listed function must lock that way.
#   single_row  - matches one row by id
#   id_order    - locks its rows in id order (queue_ids_locked_in_order)
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

# Text that shows a write statement locks the listed way, for ORM and raw SQL.
POLICY_MARKERS = {
    "single_row": ("QueueItem.id == ",),
    "id_order": ("queue_ids_locked_in_order(", "ORDER BY id FOR UPDATE"),
    "skip_locked": ("skip_locked=True", "SKIP LOCKED"),
}

# A raw-SQL write to the queue table, schema-qualified or quoted or not.
RAW_QUEUE_WRITE = re.compile(
    r"\b(UPDATE|DELETE\s+FROM)\s+(\"?\w+\"?\.)?\"?queue\"?(\s|$)", re.IGNORECASE
)


def _is_queue_target(node: ast.AST) -> bool:
    """``QueueItem``, ``models.QueueItem``, or either one's ``__table__``."""
    if isinstance(node, ast.Attribute) and node.attr == "__table__":
        node = node.value
    return (isinstance(node, ast.Attribute) and node.attr == "QueueItem") or (
        isinstance(node, ast.Name) and node.id == "QueueItem"
    )


def _is_queue_write(node: ast.AST) -> bool:
    """A statement-building call that writes ``queue``.

    Covers ``update``/``delete`` of the model, bare or through a module
    (``sa.update``), the table's own ``.update()``/``.delete()``, and an
    ``UPDATE queue``/``DELETE FROM queue`` string passed to a call such as
    ``text()`` or ``exec_driver_sql()``.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = (
        func.id
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else None
    )
    if name in {"update", "delete"}:
        # update(QueueItem), sa.update(QueueItem), sqlalchemy.delete(QueueItem)
        if node.args and _is_queue_target(node.args[0]):
            return True
        # QueueItem.__table__.update()
        if isinstance(func, ast.Attribute) and _is_queue_target(func.value):
            return isinstance(func.value, ast.Attribute) and (
                func.value.attr == "__table__"
            )
    return any(
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and RAW_QUEUE_WRITE.search(arg.value) is not None
        for arg in node.args
    )


def _queue_writes_in(label: str, source: str) -> dict[str, list[str]]:
    """Each function that writes ``queue``, with the source of each write statement."""
    tree = ast.parse(source)
    found: dict[str, list[str]] = {}

    def visit(node: ast.AST, function: str | None, statement: ast.stmt | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        if isinstance(node, ast.stmt):
            statement = node
        if function is not None and statement is not None and _is_queue_write(node):
            found.setdefault(f"{label}:{function}", []).append(
                ast.get_source_segment(source, statement) or ""
            )
        for child in ast.iter_child_nodes(node):
            visit(child, function, statement)

    visit(tree, None, None)
    return found


def _queue_writes() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        found |= _queue_writes_in(str(path.relative_to(SRC.parent)), path.read_text())
    return found


def _unlisted_or_unlocked(
    writes: dict[str, list[str]], expected: dict[str, str]
) -> list[str]:
    """Writes that aren't listed, or that don't lock the way their listing says."""
    problems = [f"{site}: not listed" for site in writes if site not in expected]
    problems += [
        f"{site}: listed but has no queue write"
        for site in expected
        if site not in writes
    ]
    for site, statements in writes.items():
        policy = expected.get(site)
        if policy is None:
            continue
        for statement in statements:
            if not any(marker in statement for marker in POLICY_MARKERS[policy]):
                problems.append(f"{site}: a write does not lock as {policy}")
    return problems


def test_every_queue_write_has_a_lock_policy() -> None:
    problems = _unlisted_or_unlocked(_queue_writes(), EXPECTED_QUEUE_WRITES)
    assert problems == [], (
        "Every write to queue must be listed in EXPECTED_QUEUE_WRITES and lock "
        "the listed way (one row by id, queue_ids_locked_in_order, or SKIP "
        f"LOCKED): {problems}"
    )


@pytest.mark.parametrize(
    "write",
    [
        "await db.execute(sa.update(models.QueueItem).values(processed=True))",
        "await db.execute(models.QueueItem.__table__.update().values(processed=True))",
        'await db.execute(text("DELETE FROM queue WHERE processed"))',
        'await db.execute(text("UPDATE public.queue SET processed = true"))',
    ],
    ids=["module-update", "table-update", "raw-delete", "raw-schema-update"],
)
def test_the_guard_finds_each_form_of_queue_write(write: str) -> None:
    source = f"async def writer(db):\n    {write}\n"
    assert _unlisted_or_unlocked(_queue_writes_in("m.py", source), {}) == [
        "m.py:writer: not listed"
    ]


def test_the_guard_checks_every_write_in_a_listed_function() -> None:
    source = (
        "async def writer(db):\n"
        "    await db.execute(update(models.QueueItem).where("
        "models.QueueItem.id.in_(queue_ids_locked_in_order(x))))\n"
        "    await db.execute(update(models.QueueItem).where(y))\n"
    )
    writes = _queue_writes_in("m.py", source)
    assert len(writes["m.py:writer"]) == 2
    assert _unlisted_or_unlocked(writes, {"m.py:writer": "id_order"}) == [
        "m.py:writer: a write does not lock as id_order"
    ]


# The race the locking prevents, run for real. Each writer races a session
# deletion over the same rows. The rows are inserted in descending id order, so
# a write that locks in scan order takes them in the opposite order from the
# deletion's id-ordered lock; on main that deadlocks every round.
RACE_ROWS = 3000
RACE_ROUNDS = 3
RACE_TIMEOUT_SECONDS = 30.0


async def _seed_descending_rows(
    db: AsyncSession,
    *,
    session_id: str,
    workspace_name: str,
    work_unit_key: str,
    processed: bool,
    first_id: int,
) -> None:
    payload = f'{{"{RETRY_ATTEMPTS_PAYLOAD_KEY}": 1}}'
    await db.execute(
        text(
            "INSERT INTO queue (id, session_id, work_unit_key, task_type, payload,"
            + " processed, workspace_name, tenant_id)"
            + " OVERRIDING SYSTEM VALUE"
            + " SELECT g, :session_id, :key, 'representation', CAST(:payload AS jsonb),"
            + " :processed, :ws, :tenant"
            + " FROM generate_series(:hi, :lo, -1) AS g"
        ),
        {
            "session_id": session_id,
            "key": work_unit_key,
            "payload": payload,
            "processed": processed,
            "ws": workspace_name,
            "tenant": models.DEFAULT_TENANT_ID,
            "hi": first_id + RACE_ROWS - 1,
            "lo": first_id,
        },
    )
    await db.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "writer",
    [
        "_clear_work_unit_retry_attempts",
        "_mark_invalid_work_unit_as_errored",
        "_fail_unscoped_work_unit",
        "cleanup_queue_items",
    ],
)
async def test_a_writer_racing_a_session_deletion_does_not_deadlock(
    writer: str,
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    from src.crud.deriver import delete_queue_items_and_claims

    workspace, _ = sample_data
    sessions = independent_sessions(db_engine)
    manager = QueueManager()

    for round_ in range(RACE_ROUNDS):
        session = models.Session(
            name=f"race-{writer}-{round_}", workspace_name=workspace.name
        )
        db_session.add(session)
        await db_session.commit()
        key = f"representation:{workspace.name}:{session.name}:race"
        await _seed_descending_rows(
            db_session,
            session_id=session.id,
            workspace_name=workspace.name,
            work_unit_key=key,
            processed=writer == "cleanup_queue_items",
            first_id=50_000_000 + round_ * RACE_ROWS,
        )

        async def write(key: str = key) -> None:
            if writer == "cleanup_queue_items":
                await cleanup_queue_items()
            elif writer == "_mark_invalid_work_unit_as_errored":
                await manager._mark_invalid_work_unit_as_errored(  # pyright: ignore[reportPrivateUsage]
                    key, ValueError("unparsable")
                )
            else:
                await getattr(manager, writer)(key)

        async def delete(key: str = key, session_id: str = session.id) -> None:
            async with sessions() as db:
                await delete_queue_items_and_claims(
                    db,
                    [[models.QueueItem.session_id == session_id]],
                    models.ActiveQueueSession.work_unit_key == key,
                )
                await db.commit()

        results = await asyncio.wait_for(
            asyncio.gather(write(), delete(), return_exceptions=True),
            RACE_TIMEOUT_SECONDS,
        )
        assert [r for r in results if isinstance(r, BaseException)] == [], (
            f"round {round_}: {results}"
        )
        remaining = await db_session.scalar(
            select(func.count())
            .select_from(models.QueueItem)
            .where(models.QueueItem.work_unit_key == key)
        )
        assert remaining == 0
