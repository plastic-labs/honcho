"""migrations.utils.create_index_online, through revision 647b773f05ac.

test_pipeline.py runs revisions on a caller-owned connection, which takes the
transactional CREATE INDEX branch. These run ``command.upgrade`` the way
``src.migrate`` does, so the revision takes the concurrent paths: CREATE INDEX
CONCURRENTLY on a plain table, and ON ONLY + per partition + ATTACH on a
partitioned one.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings

REVISION = "647b773f05ac"
DOWN_REVISION = "c075274557cd"
INDEXES = {
    "ix_documents_workspace_recent": "documents",
    "ix_documents_collection_derived": "documents",
    "ix_messages_session_tokens": "messages",
}
DOCUMENT_COLUMNS = (
    "id text NOT NULL, workspace_name text, observer text, observed text,"
    + " times_derived int, created_at timestamptz, deleted_at timestamptz"
)
MESSAGE_COLUMNS = (
    "id bigint NOT NULL, workspace_name text, session_name text, token_count int"
)


def _valid(alembic_engine: Engine, name: str) -> bool | None:
    with alembic_engine.connect() as conn:
        return conn.scalar(
            text(
                "SELECT i.indisvalid FROM pg_index i"
                + " JOIN pg_class c ON c.oid = i.indexrelid"
                + " JOIN pg_namespace n ON n.oid = c.relnamespace"
                + " WHERE c.relname = :name AND n.nspname = :schema"
            ),
            {"name": name, "schema": settings.DB.SCHEMA},
        )


def _oid(alembic_engine: Engine, name: str) -> int | None:
    with alembic_engine.connect() as conn:
        return conn.scalar(
            text("SELECT to_regclass(:name)::oid::bigint"),
            {"name": f'"{settings.DB.SCHEMA}".{name}'},
        )


def _attached(alembic_engine: Engine, name: str) -> list[tuple[str, bool]]:
    """(partition, indisvalid) of each partition index attached to the parent index."""
    with alembic_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT t.relname, i.indisvalid FROM pg_inherits h"
                + " JOIN pg_index i ON i.indexrelid = h.inhrelid"
                + " JOIN pg_class t ON t.oid = i.indrelid"
                + " WHERE h.inhparent = to_regclass(:parent) ORDER BY t.relname"
            ),
            {"parent": f'"{settings.DB.SCHEMA}".{name}'},
        )
        return [(r[0], r[1]) for r in rows]


def _partition(alembic_engine: Engine) -> None:
    """Replace documents and messages with hash-partitioned tables, as the multi-tenant schema has."""
    schema = settings.DB.SCHEMA
    with alembic_engine.begin() as conn:
        for table, columns in (
            ("documents", DOCUMENT_COLUMNS),
            ("messages", MESSAGE_COLUMNS),
        ):
            conn.execute(text(f'DROP TABLE "{schema}".{table} CASCADE'))
            conn.execute(
                text(
                    f'CREATE TABLE "{schema}".{table} (tenant_id text NOT NULL, {columns},'
                    + " PRIMARY KEY (tenant_id, id)) PARTITION BY HASH (tenant_id)"
                )
            )
            for remainder in range(2):
                conn.execute(
                    text(
                        f'CREATE TABLE "{schema}".{table}_p{remainder}'
                        + f' PARTITION OF "{schema}".{table}'
                        + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                    )
                )


def test_plain_tables_build_concurrently(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    assert all(_valid(alembic_engine, name) is None for name in INDEXES)

    command.upgrade(alembic_cfg, REVISION)

    assert all(_valid(alembic_engine, name) is True for name in INDEXES)

    command.downgrade(alembic_cfg, DOWN_REVISION)

    assert all(_valid(alembic_engine, name) is None for name in INDEXES)


def test_partitioned_tables_build_each_partition_and_attach(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition(alembic_engine)

    command.upgrade(alembic_cfg, REVISION)

    for name, table in INDEXES.items():
        assert _valid(alembic_engine, name) is True
        assert _attached(alembic_engine, name) == [
            (f"{table}_p0", True),
            (f"{table}_p1", True),
        ]

    command.downgrade(alembic_cfg, DOWN_REVISION)

    for name in INDEXES:
        assert _valid(alembic_engine, name) is None
        assert _valid(alembic_engine, f"{name}_p0") is None
        assert _valid(alembic_engine, f"{name}_p1") is None


@pytest.mark.parametrize(
    "p1_invalid", [True, False], ids=["p1-invalid", "p1-unattached"]
)
def test_partitioned_build_that_stopped_part_way_is_finished(
    alembic_cfg: Config, alembic_engine: Engine, p1_invalid: bool
) -> None:
    """A stopped build leaves an INVALID parent, an attached p0, and a p1 index
    that is INVALID (stopped in its build) or valid but not attached (stopped
    before ATTACH). The revision keeps p0, rebuilds or reuses p1, and attaches it."""
    schema = settings.DB.SCHEMA
    name, columns = (
        "ix_messages_session_tokens",
        "tenant_id, workspace_name, session_name, id",
    )
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition(alembic_engine)
    with alembic_engine.begin() as conn:
        conn.execute(
            text(
                f'CREATE INDEX {name} ON ONLY "{schema}".messages ({columns}) INCLUDE (token_count)'
            )
        )
        for remainder in range(2):
            conn.execute(
                text(
                    f'CREATE INDEX {name}_p{remainder} ON "{schema}".messages_p{remainder}'
                    + f" ({columns}) INCLUDE (token_count)"
                )
            )
        conn.execute(
            text(f'ALTER INDEX "{schema}".{name} ATTACH PARTITION "{schema}".{name}_p0')
        )
        if p1_invalid:
            # What a CREATE INDEX CONCURRENTLY that failed part-way leaves behind.
            conn.execute(
                text(
                    "UPDATE pg_index SET indisvalid = false"
                    + f" WHERE indexrelid = '\"{schema}\".{name}_p1'::regclass"
                )
            )
    p1_before = _oid(alembic_engine, f"{name}_p1")
    assert _valid(alembic_engine, name) is False
    assert _attached(alembic_engine, name) == [("messages_p0", True)]

    command.upgrade(alembic_cfg, REVISION)

    assert _valid(alembic_engine, name) is True
    assert _attached(alembic_engine, name) == [
        ("messages_p0", True),
        ("messages_p1", True),
    ]
    # An INVALID p1 is dropped and built again; a valid one is attached as it is.
    assert (_oid(alembic_engine, f"{name}_p1") != p1_before) is p1_invalid
