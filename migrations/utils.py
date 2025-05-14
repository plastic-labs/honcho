from os import getenv
from typing import Optional

from alembic import op
import sqlalchemy as sa

def get_schema() -> str:
    return getenv("DATABASE_SCHEMA", "public")

def column_exists(table_name: str, column_name: str, inspector: Optional[sa.Inspector] = None) -> bool:
    """Check if a column exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    existing_columns = [col["name"] for col in inspector.get_columns(table_name, schema=schema)]
    return column_name in existing_columns

def fk_exists(table_name: str, fk_name: str, inspector: Optional[sa.Inspector] = None) -> bool:
    """Check if a foreign key exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
    return any(fk.get("name") == fk_name for fk in foreign_keys)

def index_exists(table_name: str, index_name: str, inspector: Optional[sa.Inspector] = None) -> bool:
    """Check if an index exists in a table."""
    if inspector is None:
        inspector = sa.inspect(op.get_bind())
    schema = get_schema()
    indexes = inspector.get_indexes(table_name, schema=schema)
    return any(idx["name"] == index_name for idx in indexes)