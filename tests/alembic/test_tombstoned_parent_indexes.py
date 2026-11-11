"""Execute parent-index migrations on plain and full model-partitioned schemas."""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.orm import Session

from src import models
from src.config import settings
from src.db import Base
from tests.alembic import test_pipeline
from tests.conftest import CONNECTION_URI

REVISION = "4f9c2a7e8d31"
DOWN_REVISION = "a7e3c1d9f2b4"
INDEXES = {
    "peers": "ix_peers_tenant_workspace_deleted",
    "sessions": "ix_sessions_tenant_workspace_deleted",
}


@pytest.fixture(scope="module")
def alembic_database() -> Iterator[str]:
    """Create and drop only this module's randomly named throwaway database."""
    name = "parent_indexes_" + uuid4().hex
    maintenance = create_engine(
        CONNECTION_URI.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with maintenance.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield CONNECTION_URI.set(database=name).render_as_string(hide_password=False)
    finally:
        with maintenance.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        maintenance.dispose()


@pytest.fixture(autouse=True)
def configure_alembic_settings(
    alembic_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind env.py to the owned database without changing any retained runtime."""
    monkeypatch.setattr(settings.DB, "CONNECTION_URI", alembic_database)
    monkeypatch.setattr(settings.DB, "OWNER_CONNECTION_URI", None)
    monkeypatch.setattr(settings, "MULTI_TENANT", False)


def _prepare(cfg: Config, engine: Engine, partitioned: bool) -> None:
    """Use migrated plain tables or the complete declarative partitioned schema."""
    if not partitioned:
        command.upgrade(cfg, DOWN_REVISION)
        return
    with engine.begin() as conn:
        for extension in ("vector", "pg_trgm", "btree_gin"):
            conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
    Base.metadata.create_all(engine)
    schema = settings.DB.SCHEMA
    with engine.begin() as conn:
        for table in Base.metadata.tables.values():
            if not table.dialect_options["postgresql"].get("partition_by"):
                continue
            for remainder in range(2):
                conn.execute(
                    text(
                        f'CREATE TABLE "{schema}"."{table.name}_p{remainder}"'
                        + f' PARTITION OF "{schema}"."{table.name}"'
                        + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                    )
                )
        for name in INDEXES.values():
            conn.execute(text(f'DROP INDEX "{schema}".{name}'))
    command.stamp(cfg, DOWN_REVISION)


def _catalog_indexes(engine: Engine) -> dict[str, tuple[int, bool, str, str | None]]:
    """Snapshot every schema index, including unchanged legacy indexes."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT c.relname, c.oid::bigint, i.indisvalid,"
                + " pg_get_indexdef(i.indexrelid), pg_get_expr(i.indpred, i.indrelid)"
                + " FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid"
                + " JOIN pg_namespace n ON n.oid = c.relnamespace"
                + " WHERE n.nspname = :schema"
            ),
            {"schema": settings.DB.SCHEMA},
        ).all()
    return {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}


def _states(engine: Engine) -> dict[str, tuple[int, bool, str, str | None]]:
    """Select the revision's parent and child indexes from the real catalogs."""
    return {
        name: state
        for name, state in _catalog_indexes(engine).items()
        if any(
            name == parent or name.startswith(parent + "_")
            for parent in INDEXES.values()
        )
    }


@pytest.fixture
def executed_statements() -> Iterator[list[str]]:
    """Observe SQL actually sent by Alembic-owned engines, without mocking DDL."""
    statements: list[str] = []

    def record(
        _conn: Connection,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", record)


def _schema_state(
    engine: Engine,
) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    """Snapshot columns and constraints to prove the revision adds only indexes."""
    with engine.connect() as conn:
        columns = conn.execute(
            text(
                "SELECT table_name, column_name, data_type, is_nullable, column_default"
                + " FROM information_schema.columns WHERE table_schema = :schema"
                + " ORDER BY table_name, ordinal_position"
            ),
            {"schema": settings.DB.SCHEMA},
        ).all()
        constraints = conn.execute(
            text(
                "SELECT c.conrelid::bigint, c.conname, pg_get_constraintdef(c.oid)"
                + " FROM pg_constraint c JOIN pg_namespace n ON n.oid = c.connamespace"
                + " WHERE n.nspname = :schema ORDER BY c.conrelid, c.conname"
            ),
            {"schema": settings.DB.SCHEMA},
        ).all()
    return [tuple(row) for row in columns], [tuple(row) for row in constraints]


def _assert_indexes(engine: Engine, partitioned: bool) -> None:
    """Match model keys and predicates and require all child indexes attached."""
    states = _states(engine)
    expected = set(INDEXES.values())
    if partitioned:
        expected |= {f"{name}_p{p}" for name in INDEXES.values() for p in range(2)}
    assert set(states) == expected
    for name, (_, valid, definition, predicate) in states.items():
        assert valid is True, name
        assert "USING btree (tenant_id, workspace_name, name)" in definition
        assert predicate == "(deleted_at IS NOT NULL)"
    for table, name in INDEXES.items():
        model = models.Peer if table == "peers" else models.Session
        declaration = next(i for i in model.__table__.indexes if i.name == name)
        assert [column.name for column in declaration.columns] == [
            "tenant_id",
            "workspace_name",
            "name",
        ]
        assert (
            str(declaration.dialect_options["postgresql"]["where"])
            == "deleted_at IS NOT NULL"
        )
        assert declaration.unique is False
        with engine.connect() as conn:
            children = conn.execute(
                text(
                    "SELECT t.relname, i.indisvalid FROM pg_inherits h"
                    + " JOIN pg_index i ON i.indexrelid = h.inhrelid"
                    + " JOIN pg_class t ON t.oid = i.indrelid"
                    + " WHERE h.inhparent = to_regclass(:name) ORDER BY t.relname"
                ),
                {"name": f'"{settings.DB.SCHEMA}".{name}'},
            ).all()
        assert children == (
            [(f"{table}_p0", True), (f"{table}_p1", True)] if partitioned else []
        )


def test_registered_revision_pipeline(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    """Execute registered before/after hooks in the real migration pipeline."""
    test_pipeline.test_migration_revision(REVISION, alembic_cfg, alembic_engine)


@pytest.mark.parametrize(
    "partitioned", [False, True], ids=["migrated-plain", "model-partitioned"]
)
def test_upgrade_scope_retry_and_downgrade(
    alembic_cfg: Config,
    alembic_engine: Engine,
    partitioned: bool,
    executed_statements: list[str],
) -> None:
    """Build valid indexes, preserve schema/data/old indexes, and reverse only additions."""
    _prepare(alembic_cfg, alembic_engine, partitioned)
    with Session(alembic_engine) as db:
        db.merge(models.Tenant(tenant_id="default"))
        db.flush()
        db.add(models.Workspace(tenant_id="default", name="index-workspace"))
        db.flush()
        db.add_all(
            [
                models.Peer(
                    tenant_id="default", workspace_name="index-workspace", name="live"
                ),
                models.Session(
                    tenant_id="default", workspace_name="index-workspace", name="live"
                ),
            ]
        )
        db.commit()
    before = _schema_state(alembic_engine)
    old_indexes = _catalog_indexes(alembic_engine)
    assert _states(alembic_engine) == {}
    executed_statements.clear()
    command.upgrade(alembic_cfg, REVISION)
    _assert_indexes(alembic_engine, partitioned)
    for name in INDEXES.values():
        if partitioned:
            assert any(
                f'CREATE INDEX IF NOT EXISTS "{name}" ON ONLY' in sql
                for sql in executed_statements
            )
            for p in range(2):
                assert any(
                    f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{name}_p{p}"' in sql
                    for sql in executed_statements
                )
                assert any(
                    f'ALTER INDEX "{settings.DB.SCHEMA}"."{name}" ATTACH PARTITION "{settings.DB.SCHEMA}"."{name}_p{p}"'
                    in sql
                    for sql in executed_statements
                )
        else:
            assert any(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS "{name}"' in sql
                for sql in executed_statements
            )
    assert _schema_state(alembic_engine) == before
    states = _states(alembic_engine)
    assert _catalog_indexes(alembic_engine) == old_indexes | states
    command.stamp(alembic_cfg, DOWN_REVISION)
    command.upgrade(alembic_cfg, REVISION)
    assert _states(alembic_engine) == states
    command.downgrade(alembic_cfg, DOWN_REVISION)
    assert _states(alembic_engine) == {}
    assert _schema_state(alembic_engine) == before
    assert _catalog_indexes(alembic_engine) == old_indexes
    with alembic_engine.connect() as conn:
        for table in INDEXES:
            assert conn.execute(
                text(f'SELECT name, deleted_at FROM "{settings.DB.SCHEMA}".{table}')
            ).all() == [("live", None)]


@pytest.mark.parametrize("partitioned", [False, True])
def test_tombstone_sets_keep_tenant_and_workspace_scope(
    alembic_cfg: Config, alembic_engine: Engine, partitioned: bool
) -> None:
    """Exercise colliding parent names with live and tombstoned indexed rows."""
    _prepare(alembic_cfg, alembic_engine, partitioned)
    command.upgrade(alembic_cfg, REVISION)
    with Session(alembic_engine) as db:
        for tenant in ("default", "other"):
            db.merge(models.Tenant(tenant_id=tenant))
        db.flush()
        for tenant in ("default", "other"):
            for workspace in ("shared", "second"):
                db.add(models.Workspace(tenant_id=tenant, name=workspace))
        db.flush()
        for model in (models.Peer, models.Session):
            for tenant in ("default", "other"):
                for workspace in ("shared", "second"):
                    db.add(
                        model(
                            tenant_id=tenant,
                            workspace_name=workspace,
                            name="same-parent",
                            deleted_at=datetime.now(UTC)
                            if (tenant, workspace) == ("default", "shared")
                            else None,
                        )
                    )
        db.commit()
    with alembic_engine.connect() as conn:
        for table in INDEXES:
            for tenant in ("default", "other"):
                for workspace in ("shared", "second"):
                    names = conn.scalars(
                        text(
                            f'SELECT name FROM "{settings.DB.SCHEMA}".{table}'
                            + " WHERE tenant_id = :tenant AND workspace_name = :workspace"
                            + " AND deleted_at IS NOT NULL"
                        ),
                        {"tenant": tenant, "workspace": workspace},
                    ).all()
                    assert names == (
                        ["same-parent"]
                        if (tenant, workspace) == ("default", "shared")
                        else []
                    )
    _assert_indexes(alembic_engine, partitioned)


@pytest.mark.parametrize("table", INDEXES)
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize(
    "invalid", [False, True], ids=["valid-leftover", "invalid-leftover"]
)
def test_resume_interrupted_build(
    alembic_cfg: Config,
    alembic_engine: Engine,
    table: str,
    partitioned: bool,
    invalid: bool,
) -> None:
    """Keep valid indexes, rebuild invalid leftovers, and finish missing attachments."""
    _prepare(alembic_cfg, alembic_engine, partitioned)
    schema = settings.DB.SCHEMA
    name = INDEXES[table]
    predicate = "(tenant_id, workspace_name, name) WHERE deleted_at IS NOT NULL"
    with alembic_engine.begin() as conn:
        if partitioned:
            conn.execute(
                text(f'CREATE INDEX {name} ON ONLY "{schema}".{table} {predicate}')
            )
            for p in range(2):
                conn.execute(
                    text(
                        f'CREATE INDEX {name}_p{p} ON "{schema}".{table}_p{p} {predicate}'
                    )
                )
            conn.execute(
                text(
                    f'ALTER INDEX "{schema}".{name} ATTACH PARTITION "{schema}".{name}_p0'
                )
            )
        else:
            conn.execute(text(f'CREATE INDEX {name} ON "{schema}".{table} {predicate}'))
        leftover = f"{name}_p1" if partitioned else name
        if invalid:
            conn.execute(
                text(
                    "UPDATE pg_index SET indisvalid = false WHERE indexrelid = to_regclass(:name)"
                ),
                {"name": f'"{schema}".{leftover}'},
            )
    before = _states(alembic_engine)
    command.upgrade(alembic_cfg, REVISION)
    _assert_indexes(alembic_engine, partitioned)
    after = _states(alembic_engine)
    assert (before[leftover][0] != after[leftover][0]) is invalid
    if partitioned:
        assert before[name][0] == after[name][0]
        assert before[f"{name}_p0"][0] == after[f"{name}_p0"][0]
    command.downgrade(alembic_cfg, DOWN_REVISION)
    assert _states(alembic_engine) == {}
