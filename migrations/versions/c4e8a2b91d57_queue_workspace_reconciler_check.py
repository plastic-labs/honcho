"""constrain queue.workspace_name NULL to the reconciler lane

Revision ID: c4e8a2b91d57
Revises: b7d2f4a81c39
Create Date: 2026-09-10

Encodes the queue's lane invariant in the schema: a NULL ``workspace_name``
means the tenant-less reconciler lane and nothing else. Every other task
type belongs to a workspace and its enqueue always sets one.

The tenant half of the invariant (tenant-scoped rows carry ``tenant_id``
when ``MULTI_TENANT`` is on) is deliberately NOT a CHECK: flag-off rows
legitimately carry NULL ``tenant_id`` for every task type and a constraint
cannot read the deployment flag — that half is asserted at enqueue.

OPERATOR NOTE — the constraint ships ``NOT VALID``. It enforces on every
insert and update from the moment this lands; what ``NOT VALID`` withholds is
only Postgres's own record that the rows already in the table were checked.
This migration proves that separately, with a plain count (see ``upgrade``).
To flip the catalog flag as well, run this by hand against the database at
any convenient moment — it takes SHARE UPDATE EXCLUSIVE, which blocks other
DDL but not reads or writes, and it is safe to run more than once::

    ALTER TABLE <schema>.queue VALIDATE CONSTRAINT ck_queue_workspace_null_iff_reconciler;

Nothing in the application depends on it having been run.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import constraint_exists, get_schema

# revision identifiers, used by Alembic.
revision: str = "c4e8a2b91d57"
down_revision: str | None = "b7d2f4a81c39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

CONSTRAINT_NAME = "ck_queue_workspace_null_iff_reconciler"


def upgrade() -> None:
    if constraint_exists("queue", CONSTRAINT_NAME, "check"):
        return
    # region ai
    # Legacy cleanup first: reconciler rows enqueued while workspace_name was
    # still NOT NULL carry a real workspace and would fail validation (errored
    # rows outlive the retention-window cleanup). NULLing the workspace matches
    # what every current writer produces for the lane.
    # endregion
    op.execute(
        f"""
        UPDATE {schema}.queue
        SET workspace_name = NULL
        WHERE task_type = 'reconciler' AND workspace_name IS NOT NULL
        """
    )
    # region ai
    # Conformance is proven by an ordinary count, not by VALIDATE CONSTRAINT.
    # VALIDATE full-scans the table, and alembic runs the whole upgrade inside
    # one transaction (migrations/env.py), so the ACCESS EXCLUSIVE that ADD
    # CONSTRAINT takes would be held until that transaction commits — straight
    # through the scan, with every enqueue and claim blocked behind it. Reading
    # the same rows BEFORE any exclusive lock is taken costs the same scan under
    # ACCESS SHARE, concurrent with live traffic, and aborts the upgrade just as
    # loudly if a row disagrees. The constraint then ships NOT VALID: full
    # enforcement on new writes, no scan under an exclusive lock, and the
    # catalog flag left for an out-of-band VALIDATE (see the module docstring).
    #
    # The WHERE mirrors CHECK semantics exactly — a CHECK rejects only rows the
    # expression evaluates FALSE for, so a NULL-valued expression must not count
    # here either.
    # endregion
    violating_rows = (
        op.get_bind()
        .execute(
            sa.text(
                f"""
                SELECT count(*) FROM {schema}.queue
                WHERE NOT ((workspace_name IS NULL) = (task_type = 'reconciler'))
                """
            )
        )
        .scalar_one()
    )
    if violating_rows:
        raise RuntimeError(
            f"{violating_rows} queue row(s) violate the lane invariant "
            + "((workspace_name IS NULL) = (task_type = 'reconciler')) after the "
            + "pre-flight cleanup. The pre-flight only repairs reconciler rows "
            + "carrying a workspace; a row failing the other direction means a "
            + "writer enqueued a workspace-scoped task without one. Find it "
            + "before re-running this migration."
        )

    # region ai
    # ADD CONSTRAINT ... NOT VALID is catalog-only (no scan), but it still takes
    # ACCESS EXCLUSIVE, and a lock request that WAITS queues every subsequent
    # enqueue and claim behind it — one long-running reader would stall the
    # queue for as long as it runs. lock_timeout bounds that: the deploy fails
    # fast and is retried, rather than taking the service down while it waits.
    # endregion
    op.execute("SET LOCAL lock_timeout = '3s'")
    op.execute(
        f"""
        ALTER TABLE {schema}.queue
        ADD CONSTRAINT {CONSTRAINT_NAME}
        CHECK ((workspace_name IS NULL) = (task_type = 'reconciler'))
        NOT VALID
        """
    )
    op.execute("RESET lock_timeout")


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {schema}.queue DROP CONSTRAINT IF EXISTS {CONSTRAINT_NAME}"
    )
