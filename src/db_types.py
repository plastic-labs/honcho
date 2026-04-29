"""Cross-dialect SQLAlchemy type helpers for SQLite compatibility.

These utilities allow the same model definitions to work on both
PostgreSQL (production) and SQLite (local development).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, CheckConstraint, Index, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.types import TypeDecorator, TypeEngine

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect


class JSONBCompat(TypeDecorator[Any]):
    """Use JSONB on PostgreSQL, JSON on SQLite and other dialects.

    Overrides the Comparator to ensure JSON-specific operators (``contains``
    via ``@>``, subscript ``__getitem__``, and ``.astext``) are forwarded
    through SQLAlchemy's JSON type rather than falling back to the base
    ``ColumnOperators`` implementations (which would generate LIKE etc.).
    """

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = JSON
    cache_ok: bool | None = True

    class Comparator(TypeDecorator.Comparator[Any]):  # type: ignore[type-arg]
        def contains(self, other: Any, **kwargs: Any) -> Any:
            # Use JSON's contains (→ @> on PostgreSQL, json_extract on SQLite)
            # instead of base ColumnOperators.contains which generates LIKE.
            from sqlalchemy import JSON as SaJSON, type_coerce

            return type_coerce(self.expr, SaJSON()).contains(other, **kwargs)

        def __getitem__(self, index: Any) -> Any:
            # Delegate to the generic JSON type's subscript so that the
            # returned element has .astext (and other JSON accessor attrs).
            from sqlalchemy import JSON as SaJSON, type_coerce

            return type_coerce(self.expr, SaJSON())[index]

    comparator_factory = Comparator

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class VectorType(TypeDecorator[Any]):
    """Use pgvector Vector on PostgreSQL, JSON-serialized TEXT on SQLite.

    On SQLite the embedding is stored as a JSON text string and is never
    used for cosine-distance queries — vector search is routed through the
    external vector store (e.g. lancedb) when running in SQLite mode.

    Overrides the Comparator to forward ``cosine_distance`` to the real
    pgvector ``Vector`` type so that query expressions like
    ``column.cosine_distance(vec)`` work even though the mapped column type
    is ``VectorType`` rather than the raw pgvector ``Vector``.
    """

    impl: TypeEngine[Any] | type[TypeEngine[Any]] = Text
    cache_ok: bool | None = True

    def __init__(self, dimensions: int = 1536) -> None:
        super().__init__()
        self.dimensions: int = dimensions

    class Comparator(TypeDecorator.Comparator[Any]):  # type: ignore[type-arg]
        def cosine_distance(self, other: Any) -> Any:
            # type_coerce tells SQLAlchemy to treat this expression as a real
            # pgvector Vector without wrapping it in extra SQL, then delegate
            # to pgvector's .cosine_distance() for the <=> operator.
            from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
            from sqlalchemy import type_coerce

            return type_coerce(
                self.expr, Vector(self.type.dimensions)  # pyright: ignore[reportUnknownVariableType]
            ).cosine_distance(other)

    comparator_factory = Comparator

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


class _EmptyJsonObject(FunctionElement):  # type: ignore[type-arg]
    """Dialect-aware server default for JSONBCompat columns."""

    inherit_cache = True


@compiles(_EmptyJsonObject)  # type: ignore[misc]
def _emit_empty_json(element: Any, compiler: Any, **kw: Any) -> str:
    return "'{}'"


@compiles(_EmptyJsonObject, "postgresql")  # type: ignore[misc]
def _emit_empty_jsonb(element: Any, compiler: Any, **kw: Any) -> str:
    return "'{}'::jsonb"


def empty_json_default() -> _EmptyJsonObject:
    """Return ``'{}'::jsonb`` on PostgreSQL and ``'{}'`` on SQLite.

    Use as ``server_default=empty_json_default()`` on JSONBCompat columns so
    that Alembic autogenerate sees the same string as existing migrations on
    PostgreSQL while SQLite CREATE TABLE gets valid syntax.
    """
    return _EmptyJsonObject()


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
