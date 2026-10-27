"""The tenant predicate hook lets the planner prune partitions at plan time.

The RLS policy compares tenant_id to current_setting('app.tenant'). Postgres
prunes partitions with that predicate only at executor startup. Thus, to plan a
statement, it opens and locks each partition and each partition index. The hook
(models._add_tenant_predicate) adds tenant_id = <bound tenant>. With that value
the planner prunes at plan time, and the statement locks one partition.

The test builds two databases from the models, HASH partitioned 4 and 16 ways,
with the RLS policy of test_rls_isolation. It runs hot statements under a
non-owner role in one transaction, then counts the locks of the backend. With
the hook, the count is small and the same for the two databases.
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.pool import NullPool
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
)

from src import crud, models, schemas
from src.cache.client import cache
from src.config import settings
from src.db import (
    CROSS_TENANT,
    Base,
    ServiceReadSessionLocal,
    ServiceSessionLocal,
    tenant_context,
)
from src.db import connect_args as app_connect_args
from src.utils.search import (
    _semantic_search_pgvector,  # pyright: ignore[reportPrivateUsage]
)
from tests.conftest import (
    _drop_database,  # pyright: ignore[reportPrivateUsage]
    _get_test_db_url,  # pyright: ignore[reportPrivateUsage]
)

_SMALL, _LARGE = 4, 16
_TENANTS = ("tenant-a", "tenant-b")
_VECTOR = [0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS

# The relation locks of this backend in the schema of the models.
_LOCK_COUNT = text(
    "SELECT count(*) FROM pg_locks l JOIN pg_class c ON c.oid = l.relation "
    + "WHERE l.pid = pg_backend_pid() AND c.relnamespace = 'public'::regnamespace"
)


def _partitioned_tables() -> list[str]:
    return [
        table.name
        for table in Base.metadata.sorted_tables
        if table.kwargs.get("postgresql_partition_by")
    ]


def _seed(url: str) -> dict[str, str]:
    """Seed each tenant with one workspace of data. Return the id of a source document."""
    source_ids: dict[str, str] = {}
    engine = create_engine(url)
    try:
        for tenant in _TENANTS:
            token = tenant_context.set(tenant)
            try:
                with OrmSession(engine) as db:
                    db.add(models.Tenant(tenant_id=tenant))
                    db.flush()
                    db.add(models.Workspace(name="w"))
                    db.flush()
                    db.add(models.Peer(name="p", workspace_name="w"))
                    db.flush()
                    db.add(models.Session(name="s", workspace_name="w"))
                    db.add(
                        models.Collection(
                            observer="p", observed="p", workspace_name="w"
                        )
                    )
                    db.flush()
                    db.add(
                        models.SessionPeer(
                            workspace_name="w", session_name="s", peer_name="p"
                        )
                    )
                    messages = [
                        models.Message(
                            workspace_name="w",
                            session_name="s",
                            peer_name="p",
                            content=f"{tenant} message {seq}",
                            seq_in_session=seq,
                        )
                        for seq in (1, 2, 3)
                    ]
                    db.add_all(messages)
                    db.flush()
                    db.add_all(
                        models.MessageEmbedding(
                            content=message.content,
                            embedding=_VECTOR,
                            message_id=message.public_id,
                            workspace_name="w",
                            session_name="s",
                            peer_name="p",
                        )
                        for message in messages
                    )
                    source = models.Document(
                        id=generate_nanoid(),
                        content=f"{tenant} explicit",
                        observer="p",
                        observed="p",
                        workspace_name="w",
                        session_name="s",
                    )
                    db.add(source)
                    db.flush()
                    db.add(
                        models.Document(
                            content=f"{tenant} deductive",
                            level="deductive",
                            observer="p",
                            observed="p",
                            workspace_name="w",
                            session_name="s",
                            source_ids=[source.id],
                        )
                    )
                    db.commit()
                    source_ids[tenant] = source.id
            finally:
                tenant_context.reset(token)
    finally:
        engine.dispose()
    return source_ids


def _build_database(db_url: URL, partitions: int, role: str) -> dict[str, str]:
    """Create the partitioned schema with RLS and a non-owner role, then seed it."""
    create_database(db_url)
    url = db_url.render_as_string(hide_password=False)
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            for extension in ("vector", "pg_trgm", "btree_gin"):
                conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            for table in _partitioned_tables():
                for remainder in range(partitions):
                    conn.execute(
                        text(
                            f"CREATE TABLE {table}_p{remainder} PARTITION OF {table} "
                            + f"FOR VALUES WITH (MODULUS {partitions}, REMAINDER {remainder})"
                        )
                    )
                conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
                conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
                conn.execute(
                    text(
                        f"CREATE POLICY tenant_isolation ON {table} "
                        + "USING (tenant_id = current_setting('app.tenant', true)) "
                        + "WITH CHECK (tenant_id = current_setting('app.tenant', true))"
                    )
                )
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
            conn.execute(text(f"CREATE ROLE {role} NOLOGIN"))
            conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
            conn.execute(
                text(
                    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                    + f"IN SCHEMA public TO {role}"
                )
            )
            conn.execute(
                text(f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {role}")
            )
    finally:
        engine.dispose()
    return _seed(url)


class _PartitionedDb:
    def __init__(self, url: str, role: str, source_ids: dict[str, str]) -> None:
        self.url: str = url
        self.role: str = role
        self.source_ids: dict[str, str] = source_ids


def _drop_role(db_url: URL, role: str) -> None:
    # A role is global to the cluster. Drop it after its database.
    admin = create_engine(
        db_url.set(database="postgres").render_as_string(hide_password=False),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with admin.connect() as conn:
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
    finally:
        admin.dispose()


@pytest.fixture(scope="module")
def partitioned_dbs(worker_id: str) -> Generator[dict[int, _PartitionedDb]]:
    dbs: dict[int, _PartitionedDb] = {}
    built: list[tuple[URL, str]] = []
    try:
        for partitions, tag in ((_SMALL, "prunesmall"), (_LARGE, "prunelarge")):
            db_url = _get_test_db_url(worker_id, tag=tag)
            role = f"{db_url.database}_app"
            _drop_database(db_url)
            built.append((db_url, role))
            source_ids = _build_database(db_url, partitions, role)
            dbs[partitions] = _PartitionedDb(
                db_url.render_as_string(hide_password=False), role, source_ids
            )
        yield dbs
    finally:
        for db_url, role in built:
            _drop_database(db_url)
            _drop_role(db_url, role)


@pytest.fixture
async def async_engines(
    partitioned_dbs: dict[int, _PartitionedDb],
) -> AsyncGenerator[dict[int, AsyncEngine]]:
    # The connect arguments of the app: psycopg with prepare_threshold=None.
    # NullPool gives each case a new backend. Postgres runs foreign key checks
    # as cached plans, and on the sixth run on one backend it also builds a
    # generic plan one time. That plan locks each partition of the table.
    engines = {
        n: create_async_engine(
            db.url, connect_args=app_connect_args, poolclass=NullPool
        )
        for n, db in partitioned_dbs.items()
    }
    try:
        yield engines
    finally:
        for engine in engines.values():
            await engine.dispose()


_Case = Callable[[AsyncSession, str], Awaitable[list[Any]]]


async def _session_messages(db: AsyncSession, _source_id: str) -> list[Any]:
    stmt = await crud.get_messages("w", "s", message_count_limit=10)
    return [m.content for m in (await db.execute(stmt)).scalars()]


async def _conclusions_by_source(db: AsyncSession, source_id: str) -> list[Any]:
    stmt = crud.get_documents_with_filters("w", filters={"source_ids": source_id})
    return [d.content for d in (await db.execute(stmt)).scalars()]


async def _child_observations(db: AsyncSession, source_id: str) -> list[Any]:
    stmt = crud.get_child_observations("w", source_id)
    return [d.content for d in (await db.execute(stmt)).scalars()]


async def _semantic_message_search(db: AsyncSession, _source_id: str) -> list[Any]:
    rows = await _semantic_search_pgvector(db, "w", _VECTOR, limit=5)
    return sorted(m.content for m in rows)


async def _message_ingest(db: AsyncSession, _source_id: str) -> list[Any]:
    created = await crud.create_messages(
        db, [schemas.MessageCreate(content="new", peer_id="p")], "w", "s"
    )
    return [(m.content, m.seq_in_session) for m in created]


async def _conclusion_delete(db: AsyncSession, source_id: str) -> list[Any]:
    await crud.delete_document(db, "w", source_id, observer="p", observed="p")
    deleted = await db.scalars(
        select(models.Document.content).where(models.Document.deleted_at.is_not(None))
    )
    return list(deleted)


async def _session_delete(db: AsyncSession, _source_id: str) -> list[Any]:
    deleted = await crud.delete_session(db, "w", "s")
    return [deleted.messages_deleted, deleted.conclusions_deleted]


_CASES: dict[str, _Case] = {
    "session messages read": _session_messages,
    "conclusions read with a source filter": _conclusions_by_source,
    "reasoning chain read": _child_observations,
    "message_embeddings search": _semantic_message_search,
    "message ingest write": _message_ingest,
    "conclusion soft delete": _conclusion_delete,
    "session delete": _session_delete,
}


async def _run_case(
    engine: AsyncEngine,
    db_info: _PartitionedDb,
    case: _Case,
    tenant: str = _TENANTS[0],
) -> tuple[int, list[Any]]:
    """Run one case under RLS in one transaction. Return its lock count and result."""
    await cache.clear()
    token = tenant_context.set(tenant)
    try:
        async with AsyncSession(engine, autoflush=False, expire_on_commit=False) as db:
            await db.execute(text(f"SET LOCAL ROLE {db_info.role}"))
            await db.execute(
                text("SELECT set_config('app.tenant', :tenant, true)"),
                {"tenant": tenant},
            )
            # The crud writes commit. Flush instead, to count the locks before the end.
            db.commit = db.flush
            result = await case(db, db_info.source_ids[tenant])
            locks = int((await db.execute(_LOCK_COUNT)).scalar_one())
            await db.rollback()
    finally:
        tenant_context.reset(token)
    return locks, result


async def test_hot_statements_lock_few_relations_at_any_partition_count(
    partitioned_dbs: dict[int, _PartitionedDb],
    async_engines: dict[int, AsyncEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: dict[tuple[str, bool, int], int] = {}
    for hook in (False, True):
        # MULTI_TENANT off makes the hook inert: the statements have RLS only.
        monkeypatch.setattr(settings, "MULTI_TENANT", hook)
        for name, case in _CASES.items():
            results: list[list[Any]] = []
            for n in (_SMALL, _LARGE):
                locks, result = await _run_case(
                    async_engines[n], partitioned_dbs[n], case
                )
                counts[(name, hook, n)] = locks
                results.append(result)
            # The two databases have the same rows for tenant-a.
            assert results[0] == results[1], name

    report = "\n".join(
        f"{name}: RLS only {counts[(name, False, _SMALL)]}/{counts[(name, False, _LARGE)]}, "
        + f"with the hook {counts[(name, True, _SMALL)]}/{counts[(name, True, _LARGE)]} "
        + f"(locks at {_SMALL}/{_LARGE} partitions)"
        for name in _CASES
    )
    print(report)
    for name in _CASES:
        small, large = counts[(name, True, _SMALL)], counts[(name, True, _LARGE)]
        # With the hook, the count does not change with the partition count.
        assert small == large, report
        # Without the hook, planning locks each partition of the tables it reads.
        assert counts[(name, False, _LARGE)] > counts[(name, False, _SMALL)], report
        # The hook count is less than the RLS-only count even at few partitions.
        assert large < counts[(name, False, _SMALL)], report


async def test_hook_does_not_change_the_rows(
    partitioned_dbs: dict[int, _PartitionedDb],
    async_engines: dict[int, AsyncEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, case in _CASES.items():
        results: list[list[Any]] = []
        for hook in (False, True):
            monkeypatch.setattr(settings, "MULTI_TENANT", hook)
            results.append(
                (await _run_case(async_engines[_LARGE], partitioned_dbs[_LARGE], case))[
                    1
                ]
            )
        assert results[0] == results[1], name
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    # The soft delete changes only the one document that it names.
    _, deleted = await _run_case(
        async_engines[_LARGE], partitioned_dbs[_LARGE], _conclusion_delete
    )
    assert deleted == [f"{_TENANTS[0]} explicit"]
    # Each tenant sees only its rows, newest first. The statement cache keeps
    # the tenant as a parameter, not as a value of the first tenant.
    for tenant in _TENANTS:
        _, messages = await _run_case(
            async_engines[_LARGE], partitioned_dbs[_LARGE], _session_messages, tenant
        )
        assert messages == [f"{tenant} message {seq}" for seq in (3, 2, 1)]


async def test_loaded_objects_pickle(
    partitioned_dbs: dict[int, _PartitionedDb],
    async_engines: dict[int, AsyncEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A loaded object keeps no option of the hook. The criteria function is
    # local to the hook, and pickle does not serialize a local function.
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    async def pickled_messages(db: AsyncSession, _source_id: str) -> list[Any]:
        stmt = await crud.get_messages("w", "s", message_count_limit=10)
        rows = (await db.execute(stmt)).scalars()
        return [pickle.loads(pickle.dumps(m)).content for m in rows]

    _, messages = await _run_case(
        async_engines[_LARGE], partitioned_dbs[_LARGE], pickled_messages
    )
    assert messages == [f"{_TENANTS[0]} message {seq}" for seq in (3, 2, 1)]


async def _all_message_tenants(db: AsyncSession) -> list[str]:
    rows = await db.execute(
        select(models.Message.tenant_id).group_by(models.Message.tenant_id)
    )
    return sorted(rows.scalars())


async def test_cross_tenant_sessions_see_every_tenant(
    async_engines: dict[int, AsyncEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    engine = async_engines[_LARGE]
    every_tenant = sorted(_TENANTS)

    # The test connects as a superuser. RLS does not apply to it, as to the
    # service role (FORCE ROW LEVEL SECURITY applies to the owner, not to these).
    async with engine.connect() as conn:
        bypasses_rls = await conn.scalar(
            text(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles "
                + "WHERE rolname = current_user"
            )
        )
    assert bypasses_rls is True

    # A service session: no tenant is bound.
    async with AsyncSession(engine, info={CROSS_TENANT: True}) as db:
        assert await _all_message_tenants(db) == every_tenant

    # A service session in a scope with a bound tenant: the hook does not apply.
    token = tenant_context.set(_TENANTS[0])
    try:
        async with AsyncSession(engine, info={CROSS_TENANT: True}) as db:
            assert await _all_message_tenants(db) == every_tenant
            count = await db.scalar(select(func.count()).select_from(models.Document))
            assert count == 2 * len(_TENANTS)
    finally:
        tenant_context.reset(token)

    # A session with no tenant bound does not raise, and the hook does not apply.
    async with AsyncSession(engine) as db:
        assert await _all_message_tenants(db) == every_tenant

    # The service session factories carry the marker.
    for factory in (ServiceSessionLocal, ServiceReadSessionLocal):
        assert factory.kw["info"][CROSS_TENANT] is True


def test_mixin_covers_the_partitioned_tables() -> None:
    mixin_tables = {
        mapper.local_table.name
        for mapper in Base.registry.mappers
        if issubclass(mapper.class_, models.TenantPartitioned)
    }
    assert mixin_tables == set(_partitioned_tables())
