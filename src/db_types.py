"""Cross-dialect SQLAlchemy type helpers for SQLite compatibility.

These utilities allow the same model definitions to work on both
PostgreSQL (production) and SQLite (local development).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, Index, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, TypeEngine

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect


class JSONBCompat(TypeDecorator[Any]):
    """Use JSONB on PostgreSQL, JSON on SQLite and other dialects."""

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = JSON
    cache_ok: bool | None = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class VectorType(TypeDecorator[Any]):
    """Use pgvector Vector on PostgreSQL, JSON-serialized TEXT on SQLite.

    On SQLite the embedding is stored as a JSON text string and is never
    used for cosine-distance queries — vector search is routed through the
    external vector store (e.g. lancedb) when running in SQLite mode.
    """

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = Text
    cache_ok: bool | None = True

    def __init__(self, dimensions: int = 1536) -> None:
        super().__init__()
        self.dimensions: int = dimensions

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]

            return dialect.type_descriptor(Vector(self.dimensions))  # pyright: ignore[reportUnknownVariableType]
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if dialect.name != "postgresql" and value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if dialect.name != "postgresql" and value is not None:
            return json.loads(value)
        return value


def pg_check(sqltext: str, name: str) -> CheckConstraint | None:
    """Return a CheckConstraint only when NOT running SQLite.

    Use with ``*filter(None, [...])`` unpacking in ``__table_args__`` to
    cleanly drop the constraint on SQLite without changing model structure.
    """
    from src.config import is_sqlite

    if is_sqlite():
        return None
    return CheckConstraint(sqltext, name=name)


def pg_only_index(name: str, *args: Any, **kwargs: Any) -> Index | None:
    """Return an Index only when NOT running SQLite.

    Use with ``*filter(None, [...])`` unpacking in ``__table_args__``.
    Needed for indexes whose column expressions are PostgreSQL-specific SQL
    (e.g. ``to_tsvector(...)``). Indexes that use only ``postgresql_``-prefixed
    kwargs don't need this — SQLAlchemy already ignores those on other dialects.
    """
    from src.config import is_sqlite

    if is_sqlite():
        return None
    return Index(name, *args, **kwargs)


def upsert_insert(model: Any) -> Any:
    """Return a dialect-appropriate Insert that supports on_conflict_do_update.

    PostgreSQL uses ``sqlalchemy.dialects.postgresql.insert``;
    SQLite uses ``sqlalchemy.dialects.sqlite.insert`` (requires SQLite >= 3.24).
    """
    from src.config import is_sqlite

    if is_sqlite():
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(model)
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    return pg_insert(model)
