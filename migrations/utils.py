from collections.abc import Generator
from contextlib import contextmanager

import sqlalchemy as sa
from alembic import op
from sqlalchemy.exc import DBAPIError

from src.config import settings

# The statement_timeout of one concurrent index build (a partition, or a plain table).
BUILD_TIMEOUT = "60min"


def get_schema() -> str:
    return settings.DB.SCHEMA


def table_exists(table_name: str, inspector: sa.Inspector | None = None) -> bool:
    """Check if a table exists in the database."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    return inspector.has_table(table_name, schema=schema)


def column_exists(
    table_name: str, column_name: str, inspector: sa.Inspector | None = None
) -> bool:
    """Check if a column exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    existing_columns = [
        col["name"] for col in inspector.get_columns(table_name, schema=schema)
    ]
    return column_name in existing_columns


def fk_exists(
    table_name: str, fk_name: str, inspector: sa.Inspector | None = None
) -> bool:
    """Check if a foreign key exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
    return any(fk.get("name") == fk_name for fk in foreign_keys)


def index_exists(
    table_name: str, index_name: str, inspector: sa.Inspector | None = None
) -> bool:
    """Check if an index exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    indexes = inspector.get_indexes(table_name, schema=schema)
    return any(idx["name"] == index_name for idx in indexes)


def constraint_exists(
    table_name: str,
    constraint_name: str,
    type: str,
    inspector: sa.Inspector | None = None,
) -> bool:
    """Check if a constraint exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    if type == "check":
        constraints = inspector.get_check_constraints(table_name, schema=schema)
    elif type == "unique":
        constraints = inspector.get_unique_constraints(table_name, schema=schema)
    elif type == "primary":
        constraint = inspector.get_pk_constraint(table_name, schema=schema)
        return constraint["name"] == constraint_name
    elif type == "foreignkey":
        constraints = inspector.get_foreign_keys(table_name, schema=schema)
    else:
        raise ValueError(f"Invalid constraint type: {type}")
    return any(constraint["name"] == constraint_name for constraint in constraints)


def make_column_non_nullable_safe(table_name: str, column_name: str) -> None:
    """
    Make a column non-nullable using a non-blocking approach to minimize lock duration.

    WARNING: Only use this if you can guarantee that:
        1. No NULL values currently exist in the column
        2. The application code is already writing non-NULL values to this column or
        3. The column has never accepted NULLs in practice

    This uses a 4-step process to avoid long exclusive locks:
    1. Add CHECK constraint with NOT VALID (instant, no scan)
    2. Validate the constraint (scans but allows concurrent read/writes to the table)
    3. Set column NOT NULL (fast since we've validated the constraint)
    4. Drop the redundant CHECK constraint

    Args:
        table_name: The name of the table
        column_name: The name of the column to make non-nullable
    """
    schema = get_schema()
    conn = op.get_bind()
    constraint_name = f"{table_name}_{column_name}_not_null"

    # Step 1: Check if the column is already non-nullable
    inspector = sa.inspect(op.get_bind())
    columns = inspector.get_columns(table_name, schema=schema)
    column_info = next((col for col in columns if col["name"] == column_name), None)
    if column_info is None:
        raise ValueError(f"Column {table_name}.{column_name} does not exist")
    if not column_info["nullable"]:
        print(f"Column {table_name}.{column_name} is already non-nullable, skipping...")
        return

    # Step 2: Add CHECK constraint without validation (instant)
    # Note: op.create_check_constraint() doesn't support NOT VALID, so use raw SQL

    # Get the identifier preparer for safe quoting
    dialect = conn.dialect
    preparer = dialect.identifier_preparer

    quoted_schema = preparer.quote(schema)
    quoted_table = preparer.quote(table_name)
    quoted_constraint = preparer.quote(constraint_name)
    quoted_column = preparer.quote(column_name)

    # Step 2: Add CHECK constraint without validation (instant)
    # Note: op.create_check_constraint() doesn't support NOT VALID, so use raw SQL
    if not constraint_exists(table_name, constraint_name, "check"):
        conn.execute(
            sa.text(
                f"""
                ALTER TABLE {quoted_schema}.{quoted_table}
                ADD CONSTRAINT {quoted_constraint}
                CHECK ({quoted_column} IS NOT NULL)
                NOT VALID
                """
            )
        )

    # Step 3: Validate constraint (scans but allows concurrent operations)
    conn.execute(
        sa.text(
            f"""
            ALTER TABLE {quoted_schema}.{quoted_table}
            VALIDATE CONSTRAINT {quoted_constraint}
            """
        )
    )

    # Step 4: Set NOT NULL (fast with validated constraint)
    op.alter_column(
        table_name,
        column_name,
        nullable=False,
        schema=schema,
    )

    # Step 5: Drop the redundant CHECK constraint
    conn.execute(
        sa.text(
            f"""
          ALTER TABLE {quoted_schema}.{quoted_table}
          DROP CONSTRAINT {quoted_constraint}
          """
        )
    )


def caller_owns_transaction() -> bool:
    """True when env.py runs inside a transaction the caller opened (the alembic test pipeline)."""
    config = op.get_context().config
    return config is not None and not config.attributes.get("owns_transaction", True)


def index_valid(name: str) -> bool | None:
    """The index's indisvalid, or None if it doesn't exist."""
    return op.get_bind().scalar(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i"
            + " JOIN pg_class c ON c.oid = i.indexrelid"
            + " JOIN pg_namespace n ON n.oid = c.relnamespace"
            + " WHERE c.relname = :name AND n.nspname = :schema"
        ),
        {"name": name, "schema": get_schema()},
    )


def _is_partitioned(table: str) -> bool:
    return (
        op.get_bind().scalar(
            sa.text("SELECT relkind FROM pg_class WHERE oid = to_regclass(:table)"),
            {"table": f'"{get_schema()}"."{table}"'},
        )
        == "p"
    )


def _partitions(table: str) -> list[str]:
    return list(
        op.get_bind().scalars(
            sa.text(
                "SELECT c.relname FROM pg_inherits h JOIN pg_class c ON c.oid = h.inhrelid"
                + " WHERE h.inhparent = to_regclass(:table) ORDER BY c.relname"
            ),
            {"table": f'"{get_schema()}"."{table}"'},
        )
    )


def _attached_partitions(index: str) -> set[str]:
    """The partitions whose index is attached to the partitioned index."""
    return set(
        op.get_bind().scalars(
            sa.text(
                "SELECT t.relname FROM pg_inherits h"
                + " JOIN pg_index i ON i.indexrelid = h.inhrelid"
                + " JOIN pg_class t ON t.oid = i.indrelid"
                + " WHERE h.inhparent = to_regclass(:index)"
            ),
            {"index": f'"{get_schema()}"."{index}"'},
        )
    )


def _partition_index_name(index: str, table: str, partition: str) -> str:
    name = f"{index}_{partition.removeprefix(table + '_')}"
    assert len(name) <= 63, f"index name {name} is longer than 63 bytes"
    return name


@contextmanager
def build_timeouts(timeout: str = BUILD_TIMEOUT) -> Generator[None]:
    """No lock_timeout and a long statement_timeout, for one concurrent build.

    A concurrent build waits for every transaction that holds an older snapshot,
    so the migration's short lock_timeout would fail it. The statement_timeout
    makes a stuck idle-in-transaction session fail the build instead of hanging it.
    """
    connection = op.get_bind()
    lock_timeout = connection.scalar(sa.text("SHOW lock_timeout"))
    statement_timeout = connection.scalar(sa.text("SHOW statement_timeout"))
    connection.execute(sa.text("SET lock_timeout = 0"))
    connection.execute(sa.text(f"SET statement_timeout = '{timeout}'"))
    try:
        yield
    finally:
        try:
            for name, value in (
                ("lock_timeout", lock_timeout),
                ("statement_timeout", statement_timeout),
            ):
                connection.execute(
                    sa.text("SELECT set_config(:name, :value, false)"),
                    {"name": name, "value": value},
                )
        except DBAPIError:
            # A dropped connection can't be restored; let the build's own error surface.
            if not connection.invalidated:
                raise


def create_index_online(
    name: str,
    table: str,
    columns: str,
    *,
    include: str | None = None,
    where: str | None = None,
) -> None:
    """Create an index on a plain or a partitioned table, with no long write block.

    `columns`, `include` and `where` are SQL, e.g. columns="tenant_id, created_at DESC".

    A plain table gets CREATE INDEX CONCURRENTLY. Postgres refuses CONCURRENTLY on
    a partitioned table, so there the steps are: CREATE INDEX ... ON ONLY the
    parent (instant, INVALID), CREATE INDEX CONCURRENTLY on each partition, and
    ALTER INDEX ... ATTACH PARTITION for each one. The parent turns valid when the
    last partition is attached. The concurrent builds run under build_timeouts().
    The other steps are short but lock: ON ONLY takes SHARE on the parent table
    (writes wait), and ATTACH takes ACCESS EXCLUSIVE on the partition's new index.
    They keep the migration's lock_timeout, so src.migrate retries them.

    It is safe to run again: a valid index and an attached partition are kept, and
    an INVALID index (a build that stopped part-way) is dropped and built again.
    To build ahead of a release, run the same steps by hand; the revision then
    finds them done. Where the caller owns the transaction (the alembic test
    pipeline), it is one transactional CREATE INDEX.
    """
    schema = get_schema()
    tail = (f" INCLUDE ({include})" if include else "") + (
        f" WHERE {where}" if where else ""
    )

    def create(index: str, target: str, concurrently: bool) -> None:
        how = "CONCURRENTLY " if concurrently else ""
        op.execute(
            f'CREATE INDEX {how}IF NOT EXISTS "{index}" ON {target} ({columns}){tail}'
        )

    parent = f'"{schema}"."{table}"'
    if caller_owns_transaction():
        if index_valid(name) is False:
            op.execute(f'DROP INDEX "{schema}"."{name}"')
        create(name, parent, concurrently=False)
        return

    if not _is_partitioned(table):
        with op.get_context().autocommit_block(), build_timeouts():
            if index_valid(name) is False:
                op.execute(f'DROP INDEX CONCURRENTLY "{schema}"."{name}"')
            create(name, parent, concurrently=True)
        return

    if index_valid(name):
        return
    with op.get_context().autocommit_block():
        create(name, f"ONLY {parent}", concurrently=False)
        attached = _attached_partitions(name)
        for partition in _partitions(table):
            if partition in attached:
                continue
            child = _partition_index_name(name, table, partition)
            with build_timeouts():
                if index_valid(child) is False:
                    op.execute(f'DROP INDEX CONCURRENTLY "{schema}"."{child}"')
                create(child, f'"{schema}"."{partition}"', concurrently=True)
            op.execute(
                f'ALTER INDEX "{schema}"."{name}" ATTACH PARTITION "{schema}"."{child}"'
            )


def drop_index_online(name: str, table: str) -> None:
    """Drop an index that create_index_online made.

    A partitioned index can't be dropped CONCURRENTLY: there it is one DROP INDEX,
    which also drops the attached partition indexes. Partition indexes that a
    stopped build left unattached are dropped by name.
    """
    schema = get_schema()
    if caller_owns_transaction() or _is_partitioned(table):
        op.execute(f'DROP INDEX IF EXISTS "{schema}"."{name}"')
        for partition in _partitions(table):
            child = _partition_index_name(name, table, partition)
            op.execute(f'DROP INDEX IF EXISTS "{schema}"."{child}"')
        return
    with op.get_context().autocommit_block(), build_timeouts():
        op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{schema}"."{name}"')
