"""Utilities for asserting migration behaviour inside tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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

    def assert_table_exists(self, table: str) -> None:
        """Assert that a table exists in the schema"""
        tables = self._get_inspector().get_table_names(schema=self.schema)
        assert (
            table in tables
        ), f"Table {table} not found after migration {self.revision}"

    def assert_column_exists(
        self, table: str, column: str, *, nullable: bool | None = None
    ) -> None:
        """Assert that a column exists in the schema"""
        columns = self._get_inspector().get_columns(table, schema=self.schema)
        col_names = [c["name"] for c in columns]
        assert (
            column in col_names
        ), f"Column {table}.{column} not found after migration {self.revision}"

        if nullable is not None:
            column_info = next(col for col in columns if col["name"] == column)
            actual_nullable = column_info.get("nullable", True)
            assert (
                actual_nullable == nullable
            ), f"Column {table}.{column} nullability is {actual_nullable}; expected {nullable}"

    def assert_column_not_exists(self, table: str, column: str) -> None:
        """Assert that a column does not exist in the schema"""
        columns = self._get_inspector().get_columns(table, schema=self.schema)
        col_names = [c["name"] for c in columns]
        assert (
            column not in col_names
        ), f"Column {table}.{column} still exists after migration {self.revision}"

    def assert_column_type(self, table: str, column: str, expected_type: type) -> None:
        """Assert that a column has the expected type"""
        columns = self._get_inspector().get_columns(table, schema=self.schema)
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
        self, table: str, constraint_name: str, constraint_type: str
    ) -> None:
        """Assert that a constraint exists in the schema"""
        names = self.fetch_constraints(table, constraint_type)
        assert (
            constraint_name in names
        ), f"Constraint {constraint_name} ({constraint_type}) not found on {table} after migration {self.revision}"

    def assert_constraint_not_exists(
        self, table: str, constraint_name: str, constraint_type: str
    ) -> None:
        """Assert that a constraint does not exist in the schema"""
        names = self.fetch_constraints(table, constraint_type)
        assert (
            constraint_name not in names
        ), f"Constraint {constraint_name} ({constraint_type}) still present on {table} after migration {self.revision}"

    def assert_indexes_exist(self, checks: Sequence[tuple[str, str]]) -> None:
        """Assert that indexes exist in the schema"""
        for table_name, index_name in checks:
            indexes = self._get_inspector().get_indexes(table_name, schema=self.schema)
            names = [idx["name"] for idx in indexes]
            assert (
                index_name in names
            ), f"Index {index_name} not found on {table_name} after migration {self.revision}"

    def assert_index_exists(self, table: str, index_name: str) -> None:
        """Assert that an index exists in the schema"""
        self.assert_indexes_exist([(table, index_name)])

    def assert_indexes_not_exist(self, checks: Sequence[tuple[str, str]]) -> None:
        """Assert that indexes do not exist in the schema"""
        for table_name, index_name in checks:
            indexes = self._get_inspector().get_indexes(table_name, schema=self.schema)
            names = [idx["name"] for idx in indexes]
            assert (
                index_name not in names
            ), f"Index {index_name} still present on {table_name} after migration {self.revision}"

    def assert_index_not_exists(self, table: str, index_name: str) -> None:
        """Assert that an index does not exist in the schema"""
        self.assert_indexes_not_exist([(table, index_name)])

    def assert_query_returns(
        self, query: str, *, expected_value: Any, error_message: str
    ) -> None:
        """Assert that a query returns the expected value"""
        result = self.conn.execute(text(query))
        value = result.scalar()
        assert (
            value == expected_value
        ), f"{error_message}: expected {expected_value}, got {value} after migration {self.revision}"

    def assert_row_count(
        self, table: str, *, expected: int | None = None, minimum: int | None = None
    ) -> None:
        """Assert that a table has the expected number of rows"""
        result = self.conn.execute(
            text(f'SELECT COUNT(*) FROM "{self.schema}"."{table}"')
        )
        count = result.scalar() or 0
        if expected is not None:
            assert (
                count == expected
            ), f"Table {table} has {count} rows; expected {expected}"
        if minimum is not None:
            assert (
                count >= minimum
            ), f"Table {table} has {count} rows; expected at least {minimum}"

    def _get_inspector(self) -> Inspector:
        """Get the inspector for the connection"""
        if self._inspector is None:
            self._inspector = inspect(self.conn)
        return self._inspector

    def fetch_constraints(self, table: str, constraint_type: str) -> list[str | None]:
        """Collect the names of constraints in a table"""
        inspector = self._get_inspector()

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
