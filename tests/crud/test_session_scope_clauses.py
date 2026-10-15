"""Observer session scope is enforced in SQL, not as a fetched name list.

A peer's session-membership count is unbounded. Materializing it turns every
session name into its own bind parameter, and the PostgreSQL wire protocol caps
parameters at 65535 per statement, so a peer in enough sessions yields a
statement the driver refuses to serialize. `get_observation_context` applies the
scope twice in one statement, which halves that ceiling to ~32.7k sessions. The
message tools (`grep_messages`, `get_messages_by_date_range`, pgvector semantic
search) apply it once each.

These tests assert on compiled SQL and never execute a statement.
"""

import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import Select, select
from sqlalchemy.dialects import postgresql

from src import models
from src.config import settings
from src.crud import message as message_crud
from src.crud.message import resolve_session_scope_clauses
from src.utils.agent_tools import get_observation_context
from src.utils.scopes import scope_peer_name


def _compile(stmt: Select[Any]) -> tuple[str, dict[str, Any]]:
    compiled = stmt.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    return str(compiled), dict(compiled.params)


class _FakeResult:
    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[Any]:
        return []


class _CapturingDB:
    """Captures statements instead of executing them."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, stmt: Any) -> _FakeResult:
        self.statements.append(stmt)
        return _FakeResult()


@pytest.mark.parametrize(
    ("session_allowlist", "observer", "expect_exists", "expected_params"),
    [
        # Observer only: membership becomes a correlated EXISTS, so only the
        # workspace and peer are bound — never the session names, which is what
        # keeps the parameter count from growing with membership.
        (None, "observer-peer", True, ["observer-peer", "workspace"]),
        # Allowlist only: caller-supplied and therefore bounded, so IN is fine.
        (["s1", "s2"], None, False, ["s1", "s2"]),
        # Both: the EXISTS is intersected with the bounded IN.
        (["s1"], "observer-peer", True, ["observer-peer", "s1", "workspace"]),
        # Neither: unrestricted, nothing filtered and nothing bound.
        (None, None, False, []),
    ],
    ids=["observer-only", "allowlist-only", "observer-and-allowlist", "unrestricted"],
)
def test_scope_clause_shape(
    session_allowlist: list[str] | None,
    observer: str | None,
    expect_exists: bool,
    expected_params: list[str],
) -> None:
    clauses, deny = resolve_session_scope_clauses(
        "workspace",
        None,
        session_allowlist,
        observer,
        models.Message.session_name,
    )

    assert not deny

    sql, params = _compile(select(models.Message.public_id).where(*clauses))

    assert ("EXISTS" in sql.upper()) is expect_exists
    assert ("session_peers" in sql) is expect_exists
    assert sorted(params.values()) == expected_params


@pytest.mark.parametrize(
    ("session_name", "session_allowlist", "observer"),
    [
        # An empty allowlist fails closed rather than matching everything.
        (None, [], "observer-peer"),
        (None, [], None),
        # A pinned session the allowlist forbids fails closed.
        ("s9", ["s1", "s2"], "observer-peer"),
    ],
    ids=[
        "empty-allowlist-with-observer",
        "empty-allowlist-without-observer",
        "pinned-session-not-in-allowlist",
    ],
)
def test_scope_fails_closed(
    session_name: str | None,
    session_allowlist: list[str] | None,
    observer: str | None,
) -> None:
    clauses, deny = resolve_session_scope_clauses(
        "workspace",
        session_name,
        session_allowlist,
        observer,
        models.Message.session_name,
    )

    assert deny
    assert clauses == []


@pytest.mark.asyncio
async def test_get_observation_context_scope_costs_no_per_session_parameters() -> None:
    """The statement's parameter count depends on message_ids, not membership."""
    db = _CapturingDB()
    message_ids = [f"msg-{i}" for i in range(5)]

    await get_observation_context(
        db,  # pyright: ignore[reportArgumentType]
        "workspace",
        None,
        message_ids,
        observer="observer-peer",
    )

    assert len(db.statements) == 1
    sql, params = _compile(db.statements[0])

    # The scope must be applied to *both* the CTE and the outer select — a
    # materialized list would have cost two parameters per session there.
    # Count the subquery's FROM rather than EXISTS: the adjacency check is also
    # an EXISTS, so counting those would pass with the CTE's scope missing.
    assert sql.count("FROM public.session_peers") == 2
    # Both must correlate to the enclosing `messages` row. An uncorrelated
    # subquery compiles just as happily and would silently drop scoping
    # instead of enforcing it.
    assert sql.count("session_peers.session_name = public.messages.session_name") == 2

    # Everything bound is either a message id, the workspace, the peer, or the
    # ±1 adjacency window — nothing that scales with the peer's session count.
    expected = set(message_ids) | {"workspace", "observer-peer", -1, 1}
    assert set(params.values()) <= expected


@pytest.mark.asyncio
async def test_get_observation_context_denies_without_querying() -> None:
    """Fail-closed scopes must not reach the database at all."""
    db = _CapturingDB()

    result = await get_observation_context(
        db,  # pyright: ignore[reportArgumentType]
        "workspace",
        None,
        ["msg-1"],
        observer="observer-peer",
        session_allowlist=[],
    )

    assert result == []
    assert db.statements == []


@pytest.fixture
def capturing_tracked_db(monkeypatch: pytest.MonkeyPatch) -> _CapturingDB:
    """Routes the CRUD module's own `tracked_db` sessions to a capturing fake."""
    db = _CapturingDB()

    @asynccontextmanager
    async def _tracked_db(*_args: object, **_kwargs: object):
        yield db

    monkeypatch.setattr(message_crud, "tracked_db", _tracked_db)
    return db


def _assert_scoped_in_sql(stmt: Any, table: str) -> None:
    """The one statement run is the search itself, scoped by a correlated EXISTS.

    A name-list implementation runs a membership lookup first, so asserting on
    the search statement (rather than any statement) is what catches it.
    """
    sql, params = _compile(stmt)
    assert f"session_peers.session_name = public.{table}.session_name" in sql
    # The subquery must read session_peers alone: had it pulled the outer table
    # into its own FROM, it would be an uncorrelated cross-join that matches
    # whenever the observer belongs to any session at all -- failing open.
    assert re.search(r"EXISTS \(SELECT 1\s+FROM public\.session_peers\s+WHERE", sql)
    assert "observer-peer" in params.values()


@pytest.mark.asyncio
async def test_grep_messages_scopes_in_sql(capturing_tracked_db: _CapturingDB) -> None:
    await message_crud.grep_messages(
        "workspace", None, "needle", observer="observer-peer"
    )

    assert len(capturing_tracked_db.statements) == 1
    _assert_scoped_in_sql(capturing_tracked_db.statements[0], "messages")


@pytest.mark.asyncio
async def test_get_messages_by_date_range_scopes_in_sql() -> None:
    db = _CapturingDB()

    await message_crud.get_messages_by_date_range(
        db,  # pyright: ignore[reportArgumentType]
        "workspace",
        None,
        observer="observer-peer",
    )

    assert len(db.statements) == 1
    _assert_scoped_in_sql(db.statements[0], "messages")


@pytest.mark.asyncio
async def test_pgvector_message_search_scopes_in_sql(
    capturing_tracked_db: _CapturingDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "pgvector")

    await message_crud._semantic_search_messages(  # pyright: ignore[reportPrivateUsage]
        "workspace",
        None,
        query_embedding=[0.1, 0.2],
        operation_name="test",
        observer="observer-peer",
    )

    assert len(capturing_tracked_db.statements) == 1
    _assert_scoped_in_sql(capturing_tracked_db.statements[0], "message_embeddings")


async def _grep_with_empty_allowlist(_db: _CapturingDB) -> list[Any]:
    return await message_crud.grep_messages(
        "workspace", None, "x", observer="observer-peer", session_allowlist=[]
    )


async def _date_range_with_empty_allowlist(db: _CapturingDB) -> list[Any]:
    return await message_crud.get_messages_by_date_range(
        db,  # pyright: ignore[reportArgumentType]
        "workspace",
        None,
        observer="observer-peer",
        session_allowlist=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [_grep_with_empty_allowlist, _date_range_with_empty_allowlist],
    ids=["grep_messages", "get_messages_by_date_range"],
)
async def test_message_tools_deny_without_querying(
    capturing_tracked_db: _CapturingDB,
    call: Callable[[_CapturingDB], Awaitable[list[Any]]],
) -> None:
    assert await call(capturing_tracked_db) == []
    assert capturing_tracked_db.statements == []


@pytest.mark.asyncio
async def test_pinned_session_with_scope_peer_requires_active_membership() -> None:
    """A pinned session normally bypasses observer scoping, but a named-scope
    peer must still be an active member of it."""
    db = _CapturingDB()
    observer = scope_peer_name("team")

    await message_crud.get_messages_by_date_range(
        db,  # pyright: ignore[reportArgumentType]
        "workspace",
        "pinned-session",
        observer=observer,
    )

    sql, params = _compile(db.statements[0])
    assert "public.messages.session_name = " in sql
    assert "session_peers.left_at IS NULL" in sql
    assert {"pinned-session", observer} <= set(params.values())
