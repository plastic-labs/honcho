"""add webhooks table

Revision ID: 88b0fb10906f
Revises: 05486ce795d5
Create Date: 2025-07-25 16:12:11.015327

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    column_exists,
    constraint_exists,
    index_exists,
    table_exists,
)
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "88b0fb10906f"
down_revision: str | None = "05486ce795d5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    # 1. Add webhook_endpoints table
    op.create_table(
        "webhook_endpoints",
        sa.Column(
            "id",
            sa.TEXT(),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "workspace_name",
            sa.TEXT(),
            sa.ForeignKey("workspaces.name"),
            nullable=False,
        ),
        sa.Column("url", sa.TEXT(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("length(url) <= 2048", name="webhook_endpoint_url_length"),
        schema=schema,
    )

    op.create_index(
        op.f("idx_webhook_endpoints_workspace_lookup"),
        "webhook_endpoints",
        ["workspace_name"],
        unique=False,
        schema=schema,
    )

    # 2. Add columns to queue table
    op.add_column(
        "queue",
        sa.Column("task_type", sa.TEXT(), nullable=True),
        schema=schema,
    )

    op.add_column(
        "queue",
        sa.Column("work_unit_key", sa.Text(), nullable=True),
        schema=schema,
    )

    # 2.5 Backfill work_unit_key for unprocessed queue items
    op.execute(
        sa.text(
            f"""
            UPDATE {schema}.queue q
            SET work_unit_key =
                CASE
                    WHEN q.payload->>'task_type' IN ('representation', 'summary') THEN
                        (q.payload->>'task_type') || ':' ||
                        COALESCE(s.workspace_name, q.payload->>'workspace_name', 'None') || ':' ||
                        COALESCE(q.payload->>'session_name', 'None') || ':' ||
                        COALESCE(q.payload->>'sender_name', 'None') || ':' ||
                        COALESCE(q.payload->>'target_name', 'None')
                    ELSE
                        'representation:' ||
                        COALESCE(s.workspace_name, q.payload->>'workspace_name', 'None') || ':' ||
                        COALESCE(q.payload->>'session_name', 'None') || ':' ||
                        COALESCE(q.payload->>'sender_name', 'None') || ':' ||
                        COALESCE(q.payload->>'target_name', 'None')
                END
            FROM {schema}.sessions s
            WHERE s.id = q.session_id AND q.processed = false AND q.work_unit_key IS NULL
            """
        )
    )

    # 3. Alter active queue sessions table
    op.add_column(
        "active_queue_sessions",
        sa.Column("work_unit_key", sa.Text(), index=True),
        schema=schema,
    )

    # Add unique constraint for work_unit_key
    op.create_unique_constraint(
        "unique_work_unit_key",
        "active_queue_sessions",
        ["work_unit_key"],
        schema=schema,
    )

    inspector = sa.inspect(op.get_bind())

    if constraint_exists(
        "active_queue_sessions", "unique_active_queue_session", "unique", inspector
    ):
        op.drop_constraint(
            "unique_active_queue_session", "active_queue_sessions", schema=schema
        )

    if column_exists("active_queue_sessions", "session_id", inspector):
        op.drop_column("active_queue_sessions", "session_id", schema=schema)

    if column_exists("active_queue_sessions", "sender_name", inspector):
        op.drop_column("active_queue_sessions", "sender_name", schema=schema)

    if column_exists("active_queue_sessions", "target_name", inspector):
        op.drop_column("active_queue_sessions", "target_name", schema=schema)

    if column_exists("active_queue_sessions", "task_type", inspector):
        op.drop_column("active_queue_sessions", "task_type", schema=schema)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if table_exists("webhook_endpoints", inspector):
        if index_exists(
            "webhook_endpoints", "idx_webhook_endpoints_workspace_lookup", inspector
        ):
            op.drop_index(
                op.f("idx_webhook_endpoints_workspace_lookup"),
                table_name="webhook_endpoints",
                schema=schema,
            )

        op.drop_table("webhook_endpoints", schema=schema)

    if column_exists("queue", "task_type", inspector):
        op.drop_column("queue", "task_type", schema=schema)

    if column_exists("queue", "work_unit_key", inspector):
        op.drop_column("queue", "work_unit_key", schema=schema)

    # Drop unique constraint first if it exists
    if constraint_exists(
        "active_queue_sessions", "unique_work_unit_key", "unique", inspector
    ):
        op.drop_constraint(
            "unique_work_unit_key", "active_queue_sessions", schema=schema
        )

    if column_exists("active_queue_sessions", "work_unit_key", inspector):
        op.drop_column("active_queue_sessions", "work_unit_key", schema=schema)

    if column_exists("active_queue_sessions", "work_unit_data", inspector):
        op.drop_column("active_queue_sessions", "work_unit_data", schema=schema)
