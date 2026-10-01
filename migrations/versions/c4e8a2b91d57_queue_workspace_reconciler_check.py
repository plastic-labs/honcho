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

The constraint is added ``NOT VALID`` and validated in a second step, each in
its own autocommit block so neither holds a lock across the other. ``upgrade``
explains why the blocks are load-bearing rather than decorative. Both steps are
re-runnable, so a run that dies between them completes on the next one.
"""

from collections.abc import Sequence

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
    if not constraint_exists("queue", CONSTRAINT_NAME, "check"):
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
        # NOT VALID + VALIDATE, each in its OWN autocommit block — the split is
        # worthless without them. The plain form of ADD CONSTRAINT takes ACCESS
        # EXCLUSIVE and full-scans the table to validate, blocking every enqueue
        # and claim for the scan; splitting it is supposed to leave only the
        # scan-free half under that lock. But Postgres releases locks at
        # TRANSACTION end, and alembic runs the whole upgrade in one transaction
        # (migrations/env.py), so a plain split holds the ACCESS EXCLUSIVE taken
        # here straight through the VALIDATE below — exactly the lock it exists
        # to avoid. Each block commits as it finishes, so the exclusive lock is
        # released before the scan starts and VALIDATE runs under SHARE UPDATE
        # EXCLUSIVE, which blocks other DDL but not reads or writes.
        #
        # The blocks also commit whatever migrations preceded this one in the
        # same upgrade: a failure here leaves them applied and the database on an
        # earlier revision, to be fixed forward and re-run rather than rolled
        # back. Both halves of this migration are re-runnable by design.
        #
        # lock_timeout (SET, not SET LOCAL — there is no transaction inside the
        # block) bounds the wait for the ACCESS EXCLUSIVE: the statement itself
        # is catalog-only and instant, but a lock request that WAITS queues every
        # enqueue and claim behind it, so one long-running reader would stall the
        # queue. Failing the deploy and retrying is the cheaper outcome.
        # endregion
        with op.get_context().autocommit_block():
            op.execute("SET lock_timeout = '3s'")
            op.execute(
                f"""
                ALTER TABLE {schema}.queue
                ADD CONSTRAINT {CONSTRAINT_NAME}
                CHECK ((workspace_name IS NULL) = (task_type = 'reconciler'))
                NOT VALID
                """
            )
            op.execute("RESET lock_timeout")

    # region ai
    # Unconditional, and outside the guard above: VALIDATE on an
    # already-validated constraint is a catalog no-op (no rescan), so running it
    # every time is what makes a run that died between the two blocks finish the
    # job instead of silently leaving the constraint unvalidated forever. No
    # lock_timeout here — SHARE UPDATE EXCLUSIVE does not conflict with the ROW
    # EXCLUSIVE that writers take, so waiting for it cannot stall the queue.
    # endregion
    with op.get_context().autocommit_block():
        op.execute(f"ALTER TABLE {schema}.queue VALIDATE CONSTRAINT {CONSTRAINT_NAME}")


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {schema}.queue DROP CONSTRAINT IF EXISTS {CONSTRAINT_NAME}"
    )
