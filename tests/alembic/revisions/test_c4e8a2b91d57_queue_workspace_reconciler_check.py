"""Hooks for revision c4e8a2b91d57 (queue_workspace_reconciler_check)."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

CONSTRAINT_NAME = "ck_queue_workspace_null_iff_reconciler"
WORKSPACE_NAME = "lane-check-workspace"

# The row the migration's conformance count would reject until its pre-flight
# cleans it up: a reconciler item from when workspace_name was still NOT NULL,
# and errored rather than cleaned up, so it outlived the retention window.
LEGACY_RECONCILER_UNIT = "reconciler:legacy_sync_vectors"

# The two shapes the equivalence admits, seeded before the upgrade so validation
# has to pass over them untouched.
VALID_RECONCILER_UNIT = "reconciler:sync_vectors"
VALID_REPRESENTATION_UNIT = "representation:lane-check:session:peer"

# Distinct keys for the post-upgrade inserts: the reconciler lane has a partial
# unique index over pending rows, so re-using a seeded key would fail on that
# index instead of proving anything about the CHECK.
ADMITTED_RECONCILER_UNIT = "reconciler:post_upgrade_sync"
ADMITTED_REPRESENTATION_UNIT = "representation:lane-check:post-upgrade:peer"


@register_before_upgrade("c4e8a2b91d57")
def prepare_queue_workspace_reconciler_check(verifier: MigrationVerifier) -> None:
    """Seed a queue holding a row the constraint would reject, plus both legal shapes."""
    verifier.assert_constraint_exists("queue", CONSTRAINT_NAME, "check", exists=False)

    conn = verifier.conn
    schema = verifier.schema

    def enqueue(
        work_unit_key: str,
        task_type: str,
        *,
        workspace_name: str | None,
        processed: bool = False,
    ) -> None:
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."queue" '
                + '("session_id", "work_unit_key", "task_type", "payload", '
                + '"processed", "workspace_name", "message_id") '
                + "VALUES (NULL, :key, :task_type, '{}'::jsonb, :processed, :ws, NULL)"
            ),
            {
                "key": work_unit_key,
                "task_type": task_type,
                "processed": processed,
                "ws": workspace_name,
            },
        )

    # The violating shape the pre-flight exists for. Processed, because that is
    # how one survives: the lane's live rows are cleaned up behind it.
    enqueue(
        LEGACY_RECONCILER_UNIT,
        "reconciler",
        workspace_name=WORKSPACE_NAME,
        processed=True,
    )
    enqueue(VALID_RECONCILER_UNIT, "reconciler", workspace_name=None)
    enqueue(VALID_REPRESENTATION_UNIT, "representation", workspace_name=WORKSPACE_NAME)


@register_after_upgrade("c4e8a2b91d57")
def verify_queue_workspace_reconciler_check(verifier: MigrationVerifier) -> None:
    """Assert the lane invariant is enforced, ships NOT VALID, and holds of the rows already there."""
    conn = verifier.conn
    schema = verifier.schema

    verifier.assert_constraint_exists("queue", CONSTRAINT_NAME, "check")

    # region ai
    # The constraint is expected to be NOT VALID, and that is the fix rather
    # than a gap: VALIDATE CONSTRAINT would full-scan the table while the
    # migration's transaction still holds the ACCESS EXCLUSIVE that ADD
    # CONSTRAINT took, blocking the queue for the scan. The upgrade proves
    # conformance with a count taken before any exclusive lock instead, so this
    # asserts BOTH halves: the catalog flag is off, and the property it would
    # have recorded is nonetheless true of every row.
    # endregion
    is_validated = conn.execute(
        text(
            "SELECT convalidated FROM pg_constraint constraint_row "
            + "JOIN pg_class table_row ON table_row.oid = constraint_row.conrelid "
            + "JOIN pg_namespace schema_row "
            + "ON schema_row.oid = table_row.relnamespace "
            + "WHERE constraint_row.conname = :name "
            + "AND table_row.relname = 'queue' AND schema_row.nspname = :schema"
        ),
        {"name": CONSTRAINT_NAME, "schema": schema},
    ).scalar_one()
    assert is_validated is False, (
        f"{CONSTRAINT_NAME} is VALIDATED, so the upgrade scanned the whole "
        + "queue while holding ACCESS EXCLUSIVE — the lock this migration is "
        + "shaped to avoid. It is meant to ship NOT VALID."
    )

    violating_rows = conn.execute(
        text(
            f'SELECT count(*) FROM "{schema}"."queue" '
            + "WHERE NOT ((workspace_name IS NULL) = (task_type = 'reconciler'))"
        )
    ).scalar_one()
    assert violating_rows == 0, (
        f"{violating_rows} row(s) violate the lane invariant the upgrade's "
        + "pre-flight count is supposed to have proven clean"
    )

    def workspace_of(work_unit_key: str) -> str | None:
        return conn.execute(
            text(
                f'SELECT workspace_name FROM "{schema}"."queue" '
                + "WHERE work_unit_key = :key"
            ),
            {"key": work_unit_key},
        ).scalar_one()

    assert workspace_of(LEGACY_RECONCILER_UNIT) is None, (
        "the pre-flight has to NULL the legacy row's workspace; leaving it "
        + "would trip the upgrade's conformance count and abort the migration"
    )
    assert workspace_of(VALID_REPRESENTATION_UNIT) == WORKSPACE_NAME, (
        "the pre-flight targets the reconciler lane only"
    )
    assert workspace_of(VALID_RECONCILER_UNIT) is None

    def enqueue(
        work_unit_key: str, task_type: str, *, workspace_name: str | None
    ) -> None:
        conn.execute(
            text(
                f'INSERT INTO "{schema}"."queue" '
                + '("session_id", "work_unit_key", "task_type", "payload", '
                + '"processed", "workspace_name", "message_id") '
                + "VALUES (NULL, :key, :task_type, '{}'::jsonb, false, :ws, NULL)"
            ),
            {"key": work_unit_key, "task_type": task_type, "ws": workspace_name},
        )

    def assert_rejected(
        work_unit_key: str, task_type: str, *, workspace_name: str | None
    ) -> None:
        """Attempt one insert inside a savepoint, so the outer transaction survives it."""
        with pytest.raises(IntegrityError) as rejection, conn.begin_nested():
            enqueue(work_unit_key, task_type, workspace_name=workspace_name)
        assert CONSTRAINT_NAME in str(rejection.value), (
            f"expected {CONSTRAINT_NAME} to reject the insert, got "
            + f"{rejection.value}"
        )

    # The reconciler is cross-tenant housekeeping, so a workspace on it is a bug.
    assert_rejected(
        ADMITTED_RECONCILER_UNIT, "reconciler", workspace_name=WORKSPACE_NAME
    )
    # Every other task type belongs to a workspace and has to say which.
    assert_rejected(ADMITTED_REPRESENTATION_UNIT, "representation", workspace_name=None)

    # The constraint is an equivalence, so exactly two shapes get through.
    enqueue(ADMITTED_RECONCILER_UNIT, "reconciler", workspace_name=None)
    enqueue(
        ADMITTED_REPRESENTATION_UNIT, "representation", workspace_name=WORKSPACE_NAME
    )

    admitted_count = conn.execute(
        text(
            f'SELECT COUNT(*) FROM "{schema}"."queue" '
            + "WHERE work_unit_key IN (:reconciler_key, :representation_key)"
        ),
        {
            "reconciler_key": ADMITTED_RECONCILER_UNIT,
            "representation_key": ADMITTED_REPRESENTATION_UNIT,
        },
    ).scalar()
    assert admitted_count == 2, f"both legal shapes must insert, found {admitted_count}"

    conn.execute(
        text(
            f'DELETE FROM "{schema}"."queue" '
            + "WHERE work_unit_key IN (:reconciler_key, :representation_key)"
        ),
        {
            "reconciler_key": ADMITTED_RECONCILER_UNIT,
            "representation_key": ADMITTED_REPRESENTATION_UNIT,
        },
    )
