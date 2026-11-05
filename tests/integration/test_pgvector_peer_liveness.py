"""Pair-pinned vector guards on real tenant-colliding lifecycle graphs."""

import re
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import (
    Connection,
    DateTime,
    Engine,
    event,
    literal,
    select,
    text,
    update,
)
from sqlalchemy import cast as sql_cast
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import aliased

from src import crud, models, schemas
from src.config import settings
from src.crud.lifecycle import live_document_row
from src.db import CROSS_TENANT, tenant_context
from src.utils.scopes import scope_peer_name
from tests.integration.lifecycle_fixtures import migrated_lifecycle_engine

QUERY = [1.0] + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 1)
FAR = [0.0, 1.0] + [0.0] * (settings.EMBEDDING.VECTOR_DIMENSIONS - 2)
PAIR_LABELS = {"session", "manual", "other-session", "far"}
SCOPE_PEER = scope_peer_name("team")


@pytest.fixture
def pair_engine(worker_id: str) -> Iterator[Engine]:
    """Give each correctness graph a fresh migrated database and HNSW index."""
    with migrated_lifecycle_engine(worker_id) as engine:
        yield engine


@pytest.fixture
async def pair_db(pair_engine: Engine) -> AsyncIterator[AsyncSession]:
    """Use an owner transaction on this test's UUID-scoped throwaway database."""
    engine = create_async_engine(
        pair_engine.url, connect_args={"prepare_threshold": None}
    )
    try:
        async with AsyncSession(engine, info={CROSS_TENANT: True}) as db:
            yield db
            await db.rollback()
    finally:
        await engine.dispose()


@pytest.fixture
def pair_app_role(pair_engine: Engine) -> Iterator[str]:
    """Grant an owned non-superuser/non-bypass role the fixture's read surface."""
    role = "pair_app_" + uuid4().hex
    with pair_engine.begin() as conn:
        conn.execute(text(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))
    try:
        yield role
    finally:
        with pair_engine.begin() as conn:
            conn.execute(text(f"DROP OWNED BY {role}"))
            conn.execute(text(f"DROP ROLE {role}"))


async def _seed(
    db: AsyncSession, tenant_a: str = "pair-a", tenant_b: str = "pair-b"
) -> None:
    """Create distinct peers, reversed/scoped pairs, colliding names and manual rows."""
    for tenant in (tenant_a, tenant_b):
        db.add(models.Tenant(tenant_id=tenant))
    await db.flush()
    for tenant in (tenant_a, tenant_b):
        for workspace in ("same-ws", "other-ws"):
            db.add(models.Workspace(tenant_id=tenant, name=workspace))
    await db.flush()
    for tenant in (tenant_a, tenant_b):
        for workspace in ("same-ws", "other-ws"):
            db.add_all(
                models.Peer(tenant_id=tenant, workspace_name=workspace, name=name)
                for name in ("p3", "p4", SCOPE_PEER)
            )
            db.add_all(
                models.Session(
                    tenant_id=tenant,
                    workspace_name=workspace,
                    name=name,
                    is_active=False,
                )
                for name in ("s1", "s2")
            )
    await db.flush()
    ids = {
        (workspace, label): generate_nanoid()
        for workspace in ("same-ws", "other-ws")
        for label in (*sorted(PAIR_LABELS), "reverse", "scope-session", "scope-manual")
    }
    for tenant in (tenant_a, tenant_b):
        for workspace in ("same-ws", "other-ws"):
            db.add_all(
                models.Collection(
                    tenant_id=tenant,
                    workspace_name=workspace,
                    observer=observer,
                    observed=observed,
                )
                for observer, observed in (
                    ("p3", "p4"),
                    ("p4", "p3"),
                    (SCOPE_PEER, "p4"),
                )
            )
            db.add(
                models.SessionPeer(
                    tenant_id=tenant,
                    workspace_name=workspace,
                    session_name="s1",
                    peer_name="p3",
                    left_at=datetime.now(UTC),
                )
            )
    await db.flush()
    for tenant in (tenant_a, tenant_b):
        for workspace in ("same-ws", "other-ws"):
            for label, observer, observed, session, level, embedding in (
                ("session", "p3", "p4", "s1", "explicit", QUERY),
                ("manual", "p3", "p4", None, "deductive", QUERY),
                ("other-session", "p3", "p4", "s2", "explicit", QUERY),
                ("far", "p3", "p4", "s2", "inductive", FAR),
                ("reverse", "p4", "p3", "s1", "explicit", QUERY),
                ("scope-session", SCOPE_PEER, "p4", "s1", "explicit", QUERY),
                ("scope-manual", SCOPE_PEER, "p4", None, "deductive", QUERY),
            ):
                db.add(
                    models.Document(
                        tenant_id=tenant,
                        id=ids[workspace, label],
                        workspace_name=workspace,
                        observer=observer,
                        observed=observed,
                        content=label,
                        session_name=session,
                        level=level,
                        embedding=embedding,
                    )
                )
    await db.flush()


@contextmanager
def _capture(db: AsyncSession) -> Generator[list[tuple[str, dict[str, Any]]]]:
    """Observe real ORM driver statements, including any accidental probe."""
    assert db.bind is not None
    statements: list[tuple[str, dict[str, Any]]] = []

    def record(
        _conn: Connection,
        _cursor: Any,
        sql: str,
        parameters: Any,
        _context: Any,
        _many: bool,
    ) -> None:
        statements.append((sql, dict(cast(dict[str, Any], parameters))))

    event.listen(db.bind.sync_engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(db.bind.sync_engine, "before_cursor_execute", record)


async def _query(
    db: AsyncSession,
    filters: dict[str, Any] | None = None,
    distance: float | None = None,
    top_k: int = 10,
    observer: str = "p3",
) -> list[models.Document]:
    """Exercise the mandatory pgvector path through its public preembedded caller."""
    return list(
        await crud.query_documents(
            db,
            "same-ws",
            "needle",
            observer=observer,
            observed="p4",
            embedding=QUERY,
            filters=filters
            if filters is not None
            else {"observer_id": observer, "observed_id": "p4"},
            max_distance=distance,
            top_k=top_k,
        )
    )


@pytest.mark.parametrize(
    "target", [None, "observer", "observed", "workspace", "document", "session"]
)
@pytest.mark.parametrize("app_role", [False, True], ids=["owner", "app-rls"])
async def test_distinct_pair_liveness_and_collisions(
    pair_db: AsyncSession,
    pair_app_role: str,
    monkeypatch: pytest.MonkeyPatch,
    target: str | None,
    app_role: bool,
) -> None:
    """Keep each parent's tombstone local to its tenant and workspace."""
    await _seed(pair_db)
    if target is not None:
        model = {
            "observer": models.Peer,
            "observed": models.Peer,
            "workspace": models.Workspace,
            "document": models.Document,
            "session": models.Session,
        }[target]
        stmt = update(model).where(model.tenant_id == "pair-a")
        if target == "workspace":
            stmt = stmt.where(models.Workspace.name == "same-ws")
        else:
            stmt = stmt.where(model.workspace_name == "same-ws")
            if target in ("observer", "observed"):
                stmt = stmt.where(
                    models.Peer.name == ("p3" if target == "observer" else "p4")
                )
            elif target == "session":
                stmt = stmt.where(models.Session.name == "s1")
            else:
                stmt = stmt.where(
                    models.Document.content == "session",
                    models.Document.observer == "p3",
                )
        await pair_db.execute(stmt.values(deleted_at=datetime.now(UTC)))
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    if app_role:
        pair_db.info[CROSS_TENANT] = False
        await pair_db.execute(text(f"SET LOCAL ROLE {pair_app_role}"))
        role = (
            await pair_db.execute(
                text(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user"
                )
            )
        ).one()
        assert role == (False, False)
        assert await pair_db.scalar(
            text(
                "SELECT relrowsecurity AND relforcerowsecurity FROM pg_class WHERE oid='public.documents'::regclass"
            )
        )
    for tenant in ("pair-a", "pair-b"):
        await pair_db.execute(
            text("SELECT set_config('app.tenant', :tenant, true)"), {"tenant": tenant}
        )
        token = tenant_context.set(tenant)
        try:
            expected: set[str] = (
                PAIR_LABELS
                if tenant == "pair-b" or target is None
                else (
                    PAIR_LABELS - {"session"}
                    if target in ("document", "session")
                    else set()
                )
            )
            rows = await _query(pair_db)
            assert {row.content for row in rows} == expected
            assert all(
                row.tenant_id == tenant
                and row.workspace_name == "same-ws"
                and (row.observer, row.observed) == ("p3", "p4")
                for row in rows
            )
            sibling = await crud.query_documents(
                pair_db,
                "other-ws",
                "needle",
                observer="p3",
                observed="p4",
                embedding=QUERY,
                top_k=10,
            )
            assert {row.content for row in sibling} == PAIR_LABELS
        finally:
            tenant_context.reset(token)


async def test_actual_vector_sql_has_constant_peer_guards_and_no_probe(
    pair_db: AsyncSession,
) -> None:
    """Inspect actual executed source SQL, not an independently built helper string."""
    await _seed(pair_db)
    token = tenant_context.set("pair-a")
    try:
        with _capture(pair_db) as statements:
            assert len(await _query(pair_db)) == 4
        main = [
            (sql, params)
            for sql, params in statements
            if "ORDER BY public.documents.embedding <=>" in sql
        ]
        assert len(main) == 1
        sql, params = main[0]
        assert "public.documents.observer NOT IN" not in sql
        assert "public.documents.observed NOT IN" not in sql
        assert sql.count("NOT IN") == 1
        assert "public.documents.session_name IS NULL OR" in sql
        assert "public.documents.deleted_at IS NULL" in sql
        assert sql.count("EXISTS (SELECT peers_") == 2
        peer_selects = sql.split("EXISTS (SELECT peers_")[1:]
        peer_names: list[str] = []
        for body in peer_selects:
            body = body.split("))", 1)[0]
            assert "public.documents." not in body
            assert ".deleted_at IS NULL" in body
            for field, value in (
                ("tenant_id", "pair-a"),
                ("workspace_name", "same-ws"),
            ):
                bind = re.search(rf"\.{field} = %\(([^)]+)\)s", body)
                assert bind is not None
                assert params[bind.group(1)] == value
            name = re.search(r"\.name = %\(([^)]+)\)s", body)
            assert name is not None
            peer_names.append(params[name.group(1)])
        assert peer_names == ["p3", "p4"]
        assert any("public.documents.observer =" in statement for statement, _ in main)
        assert any("public.documents.observed =" in statement for statement, _ in main)
        assert all(
            "FROM public.peers" not in statement or "FROM public.documents" in statement
            for statement, _ in statements
        )
    finally:
        tenant_context.reset(token)


@pytest.mark.parametrize(
    "filters,distance,expected",
    [
        ({"level": "explicit"}, None, {"session", "other-session"}),
        ({"session_id": "s1"}, None, {"session"}),
        ({"session_name": "s2"}, None, {"other-session", "far"}),
        ({"session_id": ["s1", "s2"]}, None, {"session", "other-session", "far"}),
        ({"session_id": {"in": ["s1"]}}, None, {"session"}),
        ({"session_id": []}, None, set[str]()),
        ({"session_id": None}, None, {"manual"}),
        ({"level": "explicit", "session_id": ["s1"]}, 0.1, {"session"}),
        (None, 0.1, {"session", "manual", "other-session"}),
        (None, 1.0, PAIR_LABELS),
        ({"observer_id": "p4"}, None, set[str]()),
    ],
)
async def test_vector_filters_and_distance_survive(
    pair_db: AsyncSession,
    filters: dict[str, Any] | None,
    distance: float | None,
    expected: set[str],
) -> None:
    """Keep level, session, allowlist, distance and mandatory-pair intersections."""
    await _seed(pair_db)
    token = tenant_context.set("pair-a")
    try:
        assert {
            row.content for row in await _query(pair_db, filters, distance)
        } == expected
    finally:
        tenant_context.reset(token)


@pytest.mark.parametrize("top_k", [1, 5, 10])
async def test_rest_default_and_limit_controls(
    pair_db: AsyncSession, top_k: int
) -> None:
    """Respect the REST default and limits without claiming exact ANN neighbors."""
    await _seed(pair_db)
    assert schemas.ConclusionQuery(query="needle").top_k == 10
    token = tenant_context.set("pair-a")
    try:
        rows = await _query(pair_db, top_k=top_k)
        assert len(rows) == min(top_k, len(PAIR_LABELS))
        assert {row.content for row in rows} <= PAIR_LABELS
    finally:
        tenant_context.reset(token)


async def test_scoped_pair_and_generic_list_keep_own_peer_semantics(
    pair_db: AsyncSession,
) -> None:
    """A constant guard must not leak reversed pairs or scope/session exclusions."""
    await _seed(pair_db)
    token = tenant_context.set("pair-a")
    try:
        scoped = await _query(pair_db, {"session_id": ["s1"]}, observer=SCOPE_PEER)
        assert [row.content for row in scoped] == ["scope-session"]
        await pair_db.execute(
            update(models.Peer)
            .where(
                models.Peer.tenant_id == "pair-a",
                models.Peer.workspace_name == "same-ws",
                models.Peer.name == "p3",
            )
            .values(deleted_at=datetime.now(UTC))
        )
        with _capture(pair_db) as statements:
            rows = (
                await pair_db.scalars(crud.get_documents_with_filters("same-ws"))
            ).all()
        assert {row.content for row in rows} == {"scope-session", "scope-manual"}
        sql = next(sql for sql, _ in statements if "FROM public.documents" in sql)
        assert "public.documents.observer NOT IN" in sql
        assert "public.documents.observed NOT IN" in sql
        assert sql.count("NOT IN") == 3
        assert len(await _query(pair_db, observer=SCOPE_PEER)) == 2
        await pair_db.execute(
            update(models.Peer)
            .where(
                models.Peer.tenant_id == "pair-a",
                models.Peer.workspace_name == "same-ws",
                models.Peer.name == SCOPE_PEER,
            )
            .values(deleted_at=datetime.now(UTC))
        )
        assert await _query(pair_db, observer=SCOPE_PEER) == []
    finally:
        tenant_context.reset(token)


@pytest.mark.parametrize("use_alias", [False, True])
async def test_pair_option_enforces_outer_equalities_with_aliases(
    pair_db: AsyncSession, use_alias: bool
) -> None:
    """Even an otherwise unpinned enclosing query is restricted to the live pair."""
    await _seed(pair_db)
    document = aliased(models.Document) if use_alias else models.Document
    stmt = (
        select(document)
        .join(
            models.Peer,
            (models.Peer.tenant_id == document.tenant_id)
            & (models.Peer.workspace_name == document.workspace_name)
            & (models.Peer.name == document.observer),
        )
        .where(
            live_document_row(
                "same-ws", document, tenant_id="pair-a", peer_pair=("p3", "p4")
            )
        )
    )
    assert {row.content for row in (await pair_db.scalars(stmt)).all()} == PAIR_LABELS


@pytest.mark.parametrize(
    "observer,observed,expected",
    [("p3", "p4", True), ("missing", "p4", False), ("p3", "missing", False)],
)
async def test_positive_pair_guard_fails_closed_for_missing_peer(
    pair_db: AsyncSession, observer: str, observed: str, expected: bool
) -> None:
    """Project a synthetic orphan row without disabling any real foreign key."""
    await _seed(pair_db)
    row = select(
        literal("pair-a").label("tenant_id"),
        literal("same-ws").label("workspace_name"),
        literal(observer).label("observer"),
        literal(observed).label("observed"),
        sql_cast(literal(None), DateTime(timezone=True)).label("deleted_at"),
        literal(None).label("session_name"),
    ).subquery()
    document = aliased(models.Document, row, adapt_on_names=True)
    stmt = select(
        live_document_row(
            "same-ws", document, tenant_id="pair-a", peer_pair=(observer, observed)
        )
    ).select_from(row)
    assert await pair_db.scalar(stmt) is expected


async def test_unbound_and_conflicting_rls_tenant_fail_closed(
    pair_db: AsyncSession, pair_app_role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not substitute an instance tenant or escape a conflicting RLS scope."""
    await _seed(pair_db)
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="no tenant bound"):
            await _query(pair_db)
    finally:
        tenant_context.reset(token)
    await pair_db.execute(text(f"SET LOCAL ROLE {pair_app_role}"))
    pair_db.info[CROSS_TENANT] = False
    token = tenant_context.set("pair-a")
    try:
        for guc in ("", "pair-b"):
            await pair_db.execute(
                text("SELECT set_config('app.tenant', :tenant, true)"), {"tenant": guc}
            )
            assert await _query(pair_db) == []
    finally:
        tenant_context.reset(token)


async def test_committed_parent_deletion_is_seen_on_next_read(
    pair_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A separate writer commit cannot leave a cached guard on the next statement."""
    tenant_a, tenant_b = "committed-a-" + uuid4().hex, "committed-b-" + uuid4().hex
    write_engine = create_async_engine(
        pair_engine.url, connect_args={"prepare_threshold": None}
    )
    read_engine = create_async_engine(
        pair_engine.url,
        isolation_level="AUTOCOMMIT",
        connect_args={"prepare_threshold": None},
    )
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    token = tenant_context.set(tenant_a)
    try:
        async with AsyncSession(write_engine, info={CROSS_TENANT: True}) as writer:
            await _seed(writer, tenant_a, tenant_b)
            await writer.commit()
            async with AsyncSession(read_engine, info={CROSS_TENANT: True}) as reader:
                assert {row.content for row in await _query(reader)} == PAIR_LABELS
                await writer.execute(
                    update(models.Session)
                    .where(
                        models.Session.tenant_id == tenant_a,
                        models.Session.workspace_name == "same-ws",
                        models.Session.name == "s1",
                    )
                    .values(deleted_at=datetime.now(UTC))
                )
                await writer.commit()
                assert {row.content for row in await _query(reader)} == PAIR_LABELS - {
                    "session"
                }
                await writer.execute(
                    update(models.Peer)
                    .where(
                        models.Peer.tenant_id == tenant_a,
                        models.Peer.workspace_name == "same-ws",
                        models.Peer.name == "p4",
                    )
                    .values(deleted_at=datetime.now(UTC))
                )
                await writer.commit()
                with _capture(reader) as statements:
                    assert await _query(reader) == []
                assert (
                    len(
                        [
                            sql
                            for sql, _ in statements
                            if "ORDER BY public.documents.embedding <=>" in sql
                        ]
                    )
                    == 1
                )
    finally:
        tenant_context.reset(token)
        await read_engine.dispose()
        await write_engine.dispose()
