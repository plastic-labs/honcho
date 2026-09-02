# honcho/scripts/bootstrap_shared_schema.py
"""Bootstrap the shared-tenant partitioned schema on a fresh shared database."""

# region ai
# This is deliberately NOT an Alembic migration. As a chained revision it would
# run after the per-tenant migration history that already creates these tables
# (non-partitioned): ``alembic upgrade head`` would collide (DuplicateTable)
# and, worse, it would run on existing single-tenant instances via
# ``init_db()`` and error there too. Instead this is a standalone bootstrap
# that the migration/consolidation track runs explicitly against the fresh
# shared database that per-tenant data is consolidated into. How that database
# is provisioned and version-stamped is owned by the migration track.
#
# The DDL is generated from the declarative models (``src.models.Base.metadata``):
# the models are the single source of truth, the compiler renders ``PARTITION
# BY`` from each table's ``postgresql_partition_by`` option, and this stays in
# lockstep with the data model. Only the per-table HASH partitions (which the
# model layer does not enumerate) are created explicitly, and the set of
# partitioned tables is derived from the models — never hand-listed — so a newly
# partitioned model can't silently ship a parent with zero partitions (which
# would fail every insert).
#
# Statements run in AUTOCOMMIT: the partitioned parents times ``PARTITION_COUNT``
# partitions, plus their composite FKs (each child FK locks every partition of
# the referenced parent), accumulate tens of thousands of locks, which overflows
# ``max_locks_per_transaction`` if run in a single transaction. Committing per
# statement releases each partition's locks incrementally.
#
# Prerequisite: the ``vector`` extension must already be installed (for the
# embedding columns and their HNSW indexes).
# endregion

import os
import sys

# ai: Add the project root to the path (this script is run from the scripts directory).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from sqlalchemy import Connection, Table, text  # noqa: E402
from sqlalchemy.schema import CreateIndex, CreateTable  # noqa: E402

# region ai
# Importing Base from src.models (rather than src.db) also registers every model
# on Base.metadata as a side effect — that populated metadata is the schema this
# script builds.
# endregion
from src.models import Base  # noqa: E402  # pyright: ignore

# region ai
# HASH(tenant_id) partition count. Sized so each partition stays around a couple
# GB — well under a node's memory, with headroom for growth. Contention is not the
# driver (write volume is low); this is a maintenance/pruning choice.
# endregion
PARTITION_COUNT = 128


def partitioned_tables() -> set[str]:
    """Names of the tables declared with HASH(tenant_id) partitioning."""

    # region ai
    # Derived from the models, never hand-listed: a hand list drifts out of
    # sync, and a partitioned parent with no partitions fails every insert.
    # endregion
    return {
        table.name
        for table in Base.metadata.tables.values()
        if table.dialect_options["postgresql"].get("partition_by")
    }


def _qualified(table: Table) -> str:
    return f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'


def _create_hash_partitions(conn: Connection, table: Table) -> None:
    schema_prefix = f'"{table.schema}".' if table.schema else ""
    for remainder in range(PARTITION_COUNT):
        conn.execute(
            text(
                f'CREATE TABLE {schema_prefix}"{table.name}_p{remainder:03d}"'
                + f" PARTITION OF {_qualified(table)}"
                + f" FOR VALUES WITH (MODULUS {PARTITION_COUNT}, REMAINDER {remainder})"
            )
        )


def bootstrap_shared_schema(conn: Connection) -> None:
    """Create the shared partitioned schema on ``conn``."""

    # region ai
    # ``conn`` must be in AUTOCOMMIT (see the module docstring on lock
    # accumulation).
    # endregion
    schema = Base.metadata.schema
    if schema and schema != "public":
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    partitioned = partitioned_tables()
    # region ai
    # Create tables in FK-dependency order (tenants first); a partitioned parent's
    # HASH partitions are created immediately after it.
    # endregion
    for table in Base.metadata.sorted_tables:
        conn.execute(CreateTable(table))
        if table.name in partitioned:
            _create_hash_partitions(conn, table)
    # region ai
    # Indexes after every partition exists, so each partitioned index cascades to
    # all partitions.
    # endregion
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            conn.execute(CreateIndex(index))


if __name__ == "__main__":
    # region ai
    # The app's async engine exposes a sync engine; CONNECTION_URI must point at
    # the fresh shared database.
    # endregion
    from src.db import engine  # noqa: E402

    connection = engine.sync_engine.connect().execution_options(
        isolation_level="AUTOCOMMIT"
    )
    with connection as conn:
        bootstrap_shared_schema(conn)
    print("Shared-tenant schema bootstrapped.")
