"""Utilities for asserting alembic migration behaviour inside tests."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector

from migrations.utils import get_schema


class MigrationVerifier:
    """Helper to run reusable assertions against the migrated schema."""

    def __init__(self, connection: Connection, revision: str):
        self.conn: Connection = connection
        self.revision: str = revision
        self.schema: str = get_schema()
        self._inspector: Inspector | None = None

    def assert_table_exists(self, table: str, *, exists: bool = True) -> None:
        """Assert that a table exists in the schema"""
        tables = self.get_inspector().get_table_names(schema=self.schema)

        if exists:
            assert table in tables
        else:
            assert table not in tables

    def assert_column_exists(
        self,
        table: str,
        column: str,
        *,
        exists: bool = True,
        nullable: bool | None = None,
    ) -> None:
        """Assert that a column exists in the schema"""
        columns = self.get_inspector().get_columns(table, schema=self.schema)
        col_names = [c["name"] for c in columns]

        if exists:
            assert column in col_names
        else:
            assert column not in col_names

        if nullable is not None:
            if column not in col_names:
                # Column absence was asserted above; nothing further to verify
                return

            column_info = next(col for col in columns if col["name"] == column)
            actual_nullable = column_info.get("nullable", True)
            assert (
                actual_nullable == nullable
            ), f"Column {table}.{column} nullability is {actual_nullable}; expected {nullable}"

    def assert_column_type(self, table: str, column: str, expected_type: type) -> None:
        """Assert that a column has the expected type"""
        columns = self.get_inspector().get_columns(table, schema=self.schema)
        column_info = next((col for col in columns if col["name"] == column), None)
        assert (
            column_info is not None
        ), f"Column {table}.{column} not found after migration {self.revision}"
        actual_type = column_info["type"]
        assert isinstance(
            actual_type, expected_type
        ), f"Column {table}.{column} has type {type(actual_type).__name__}; expected {expected_type.__name__}"

    def assert_no_nulls(self, table: str, column: str) -> None:
        """Assert that a column has no null values"""
        result = self.conn.execute(
            text(
                f'SELECT COUNT(*) FROM "{self.schema}"."{table}" '
                + f'WHERE "{column}" IS NULL'
            )
        )
        count = result.scalar() or 0
        assert (
            count == 0
        ), f"Found {count} NULL values in {table}.{column} after migration {self.revision}"

    def assert_constraint_exists(
        self,
        table: str,
        constraint_name: str,
        constraint_type: str,
        *,
        exists: bool = True,
    ) -> None:
        """Assert that a constraint exists in the schema"""
        names = self.fetch_constraints(table, constraint_type)

        if exists:
            assert constraint_name in names
        else:
            assert constraint_name not in names

    def assert_indexes_exist(self, checks: Sequence[tuple[str, str]]) -> None:
        """Assert that indexes exist in the schema"""
        for table_name, index_name in checks:
            indexes = self.get_inspector().get_indexes(table_name, schema=self.schema)
            names = [idx["name"] for idx in indexes]
            assert (
                index_name in names
            ), f"Index {index_name} not found on {table_name} after migration {self.revision}"

    def assert_indexes_not_exist(self, checks: Sequence[tuple[str, str]]) -> None:
        """Assert that indexes do not exist in the schema"""
        for table_name, index_name in checks:
            indexes = self.get_inspector().get_indexes(table_name, schema=self.schema)
            names = [idx["name"] for idx in indexes]
            assert (
                index_name not in names
            ), f"Index {index_name} still present on {table_name} after migration {self.revision}"

    def get_inspector(self) -> Inspector:
        """Get the inspector for the connection"""
        if self._inspector is None:
            self._inspector = inspect(self.conn)
        return self._inspector

    def fetch_constraints(self, table: str, constraint_type: str) -> list[str | None]:
        """Collect the names of constraints in a table"""
        inspector = self.get_inspector()

        if constraint_type == "unique":
            constraints = inspector.get_unique_constraints(table, schema=self.schema)
        elif constraint_type == "foreign_key":
            constraints = inspector.get_foreign_keys(table, schema=self.schema)
        elif constraint_type == "check":
            constraints = inspector.get_check_constraints(table, schema=self.schema)
        elif constraint_type == "primary_key":
            constraint = inspector.get_pk_constraint(table, schema=self.schema)
            constraints = [constraint] if constraint else []
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        return [c.get("name") for c in constraints if c]


__all__ = ["MigrationVerifier"]
