"""get_messages_id_range with a token_limit: the backward walk."""

import random
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import event, func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud import message as message_crud

# Token counts per session, oldest first. A 0 is an empty message.
_rng = random.Random(7)
SESSIONS: dict[str, list[int]] = {
    "edge": [0, 0, 4, 0, 9, 0, 0, 1, 30, 2, 0, 3, 0, 0],
    "zeros": [0] * 8,
    "one_large": [100],
    "random": [0 if _rng.random() < 0.25 else _rng.randint(1, 20) for _ in range(60)]
    + [500]
    + [_rng.choice([0, 1, 2]) for _ in range(12)],
    "empty": [],
}
LIMITS = [*range(-1, 45), 99, 100, 101, 499, 500, 501, 600, 10_000]


async def _old_get_messages_id_range(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    start_id: int,
    end_id: int | None,
    token_limit: int,
) -> list[int]:
    """get_messages_id_range before the walk, with its window query. Returns ids."""
    if start_id < 0 or (end_id is not None and (start_id >= end_id or end_id <= 0)):
        return []
    base_conditions = [
        models.Message.workspace_name == workspace_name,
        models.Message.session_name == session_name,
        models.Message.id >= start_id,
    ]
    if end_id:
        base_conditions.append(models.Message.id < end_id)
    if not token_limit:
        # No window. This path has no ORDER BY, before and after.
        stmt = select(models.Message.id).where(*base_conditions)
        return sorted((await db.execute(stmt)).scalars())
    tenant_id = models.scoped_tenant_id()
    token_subquery = (
        select(
            models.Message.id,
            func.sum(models.Message.token_count)
            .over(order_by=models.Message.id.desc())
            .label("running_token_sum"),
        )
        .where(models.Message.tenant_id == tenant_id, *base_conditions)
        .subquery()
    )
    stmt = (
        select(models.Message.id)
        .join(
            token_subquery,
            (models.Message.tenant_id == tenant_id)
            & (models.Message.id == token_subquery.c.id),
        )
        .where(token_subquery.c.running_token_sum <= token_limit)
        .order_by(models.Message.id)
    )
    return list((await db.execute(stmt)).scalars())


async def _add_messages(
    db: AsyncSession,
    workspace_name: str,
    peer_name: str,
    token_counts: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Insert the sessions with their messages interleaved; return ids per session."""
    db.add_all(
        models.Session(name=name, workspace_name=workspace_name)
        for name in token_counts
    )
    await db.flush()
    rows: list[dict[str, Any]] = []
    for seq in range(max(map(len, token_counts.values())) + 1):
        for name, counts in token_counts.items():
            if seq < len(counts):
                rows.append(
                    {
                        "public_id": generate_nanoid(),
                        "session_name": name,
                        "workspace_name": workspace_name,
                        "peer_name": peer_name,
                        "content": "x" * counts[seq],
                        "token_count": counts[seq],
                        "seq_in_session": seq + 1,
                    }
                )
    if rows:
        await db.execute(insert(models.Message), rows)
    await db.commit()
    ids: dict[str, list[int]] = {name: [] for name in token_counts}
    for message_id, name in await db.execute(
        select(models.Message.id, models.Message.session_name)
        .where(models.Message.workspace_name == workspace_name)
        .order_by(models.Message.id)
    ):
        ids[name].append(message_id)
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [1, 3, 1000])
async def test_walk_matches_the_old_window(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
    page_size: int,
):
    workspace, peer = sample_data
    ids = await _add_messages(db_session, workspace.name, peer.name, SESSIONS)
    monkeypatch.setattr(message_crud, "TOKEN_WALK_PAGE_SIZE", page_size)

    for name, session_ids in ids.items():
        mid = session_ids[len(session_ids) // 2] if session_ids else 1
        bounds = [(0, None), (mid, None), (0, mid + 3), (mid, mid + 3)]
        if session_ids:
            bounds += [(session_ids[1 % len(session_ids)], session_ids[-1])]
        for start_id, end_id in bounds:
            for token_limit in LIMITS:
                expected = await _old_get_messages_id_range(
                    db_session, workspace.name, name, start_id, end_id, token_limit
                )
                messages = await crud.get_messages_id_range(
                    db_session,
                    workspace.name,
                    name,
                    start_id=start_id,
                    end_id=end_id,
                    token_limit=token_limit,
                )
                got = [m.id for m in messages]
                if not token_limit:
                    got.sort()
                assert got == expected, (
                    name,
                    start_id,
                    end_id,
                    token_limit,
                )


@pytest.mark.asyncio
async def test_walk_reads_pages_until_the_budget_is_met(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    session_length = 3000
    ids = await _add_messages(
        db_session, workspace.name, peer.name, {"long": [10] * session_length}
    )
    page_size = 4
    monkeypatch.setattr(message_crud, "TOKEN_WALK_PAGE_SIZE", page_size)
    await db_session.execute(text("ANALYZE messages"))

    async def tuples_read() -> int:
        """Rows that scans of messages and its indexes read in this transaction."""
        return (
            await db_session.execute(
                text("""
                    SELECT pg_stat_get_xact_tuples_returned('messages'::regclass)
                        + sum(pg_stat_get_xact_tuples_returned(indexrelid))
                    FROM pg_index WHERE indrelid = 'messages'::regclass
                """)
            )
        ).scalar_one()

    statements: list[tuple[str, Any]] = []

    def capture(*args: Any) -> None:
        statements.append((args[2], args[3]))

    engine = db_session.get_bind()
    before = await tuples_read()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        messages = await crud.get_messages_id_range(
            db_session, workspace.name, "long", token_limit=105
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    walk_and_fetch = await tuples_read() - before

    # 10 messages fit. The 11th is in the third page of 4, and the walk stops there.
    assert [m.id for m in messages] == ids["long"][-10:]
    assert len(statements) == 4
    for sql, params in statements:
        assert "messages.tenant_id = " in sql
        assert models.DEFAULT_TENANT_ID in params.values()
    for sql, params in statements[:3]:
        assert "LIMIT" in sql
        assert page_size in params.values()
    assert "BETWEEN" in statements[3][0]
    # The walk reads 3 pages, the fetch reads the 10 messages.
    assert walk_and_fetch <= 3 * page_size + 10

    before = await tuples_read()
    await _old_get_messages_id_range(db_session, workspace.name, "long", 0, None, 105)
    assert await tuples_read() - before >= session_length
