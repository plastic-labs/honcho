import sqlalchemy as sa
from alembic import op

from src.config import settings


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

    print(f"Adding CHECK constraint without validation: {quoted_constraint}")

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
