"""Observer session scope is enforced in SQL, not as a fetched name list.

A peer's session-membership count is unbounded. Materializing it turns every
session name into its own bind parameter, and the PostgreSQL wire protocol caps
parameters at 65535 per statement, so a peer in enough sessions yields a
statement the driver refuses to serialize. `get_observation_context` applies the
scope twice in one statement, which halves that ceiling to ~32.7k sessions.

These tests assert on compiled SQL and never execute a statement.
"""

from typing import Any

import pytest
from sqlalchemy import Select, select
from sqlalchemy.dialects import postgresql

from src import models
from src.crud.message import resolve_session_scope_clauses
from src.utils.agent_tools import get_observation_context


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
