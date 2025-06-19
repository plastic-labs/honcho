from os import getenv
from typing import Optional

import sqlalchemy as sa
from alembic import op

from src.config import settings


def get_schema() -> str:
    return settings.DB.SCHEMA


def table_exists(table_name: str, inspector: Optional[sa.Inspector] = None) -> bool:
    """Check if a table exists in the database."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    return inspector.has_table(table_name, schema=schema)


def column_exists(
    table_name: str, column_name: str, inspector: Optional[sa.Inspector] = None
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
    table_name: str, fk_name: str, inspector: Optional[sa.Inspector] = None
) -> bool:
    """Check if a foreign key exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
    return any(fk.get("name") == fk_name for fk in foreign_keys)


def index_exists(
    table_name: str, index_name: str, inspector: Optional[sa.Inspector] = None
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
    inspector: Optional[sa.Inspector] = None,
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
    else:
        raise ValueError(f"Invalid constraint type: {type}")
    return any(constraint["name"] == constraint_name for constraint in constraints)
