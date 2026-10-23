"""Both published branches must survive at the common migration head."""

from sqlalchemy import DateTime

from tests.alembic.registry import register_after_upgrade
from tests.alembic.revisions.test_c6d9e2f74a10_merge_progress_and_pending_work import (
    verify_merged_schema,
)
from tests.alembic.verifier import MigrationVerifier

REVISION = "f2a6d9c81b04"
TOMBSTONE_TABLES = ("workspaces", "peers", "sessions", "messages")


@register_after_upgrade(REVISION)
def verify_progress_and_tombstone_schema(verifier: MigrationVerifier) -> None:
    verify_merged_schema(verifier)
    inspector = verifier.get_inspector()
    for table in TOMBSTONE_TABLES:
        verifier.assert_column_exists(table, "deleted_at", nullable=True)
        column = next(
            col
            for col in inspector.get_columns(table, schema=verifier.schema)
            if col["name"] == "deleted_at"
        )
        assert isinstance(column["type"], DateTime)
        assert column["type"].timezone
        assert column["default"] is None
