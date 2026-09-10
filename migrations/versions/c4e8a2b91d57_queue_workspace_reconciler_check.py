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
        op.execute(
            f"""
            ALTER TABLE {schema}.queue
            ADD CONSTRAINT {CONSTRAINT_NAME}
            CHECK ((workspace_name IS NULL) = (task_type = 'reconciler'))
            """
        )


def downgrade() -> None:
    op.execute(
        f"ALTER TABLE {schema}.queue DROP CONSTRAINT IF EXISTS {CONSTRAINT_NAME}"
    )
