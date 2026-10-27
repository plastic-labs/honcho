"""The merged head must contain both progress and pending-work schema."""

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier

REVISION = "c6d9e2f74a10"
PROGRESS_COLUMNS = (
    "started_at",
    "last_activity_at",
    "stage",
    "current_queue_item_id",
)
GENERATION_TABLES = ("workspaces", "sessions", "collections")
INDEXES = (
    "ix_queue_pending_observers",
    "ix_queue_pending_observed",
    "ix_queue_pending_observer",
    "ix_queue_pending_scope_peer",
    "ix_queue_pending_session_name",
)


@register_after_upgrade(REVISION)
def verify_merged_schema(verifier: MigrationVerifier) -> None:
    for column in PROGRESS_COLUMNS:
        verifier.assert_column_exists("active_queue_sessions", column, nullable=True)
    for table in GENERATION_TABLES:
        verifier.assert_column_exists(table, "deletion_generation", nullable=False)
    states = verifier.conn.execute(
        text(
            "SELECT c.relname, i.indisvalid FROM pg_index i"
            + " JOIN pg_class c ON c.oid = i.indexrelid"
            + " JOIN pg_namespace n ON n.oid = c.relnamespace"
            + " WHERE c.relname = ANY(:names) AND n.nspname = :schema"
        ),
        {"names": list(INDEXES), "schema": verifier.schema},
    ).all()
    assert {name for name, _ in states} == set(INDEXES)
    assert all(valid for _, valid in states)
