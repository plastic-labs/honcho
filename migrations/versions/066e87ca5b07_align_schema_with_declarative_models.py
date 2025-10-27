"""align_schema_with_declarative_models

Revision ID: 066e87ca5b07
Revises: bb6fb3a7a643
Create Date: 2025-10-27 12:36:51.614959

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    column_exists,
    fk_exists,
    get_schema,
    make_column_non_nullable_safe,
)

# revision identifiers, used by Alembic.
revision: str = "066e87ca5b07"
down_revision: str | None = "bb6fb3a7a643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    conn = op.get_bind()

    # Make peers.workspace_name non-nullable
    if column_exists("peers", "workspace_name"):
        make_column_non_nullable_safe("peers", "workspace_name")

    # Make sessions.workspace_name non-nullable
    if column_exists("sessions", "workspace_name"):
        make_column_non_nullable_safe("sessions", "workspace_name")

    # Make active_queue_sessions.work_unit_key non-nullable
    if column_exists("active_queue_sessions", "work_unit_key"):
        make_column_non_nullable_safe("active_queue_sessions", "work_unit_key")

    # Add FK constraint on queue.session_id to sessions.id
    if not fk_exists("queue", "fk_queue_session_id"):
        # Add constraint without validation (fast, doesn't scan)
        conn.execute(
            sa.text(
                f"""
                ALTER TABLE {schema}.queue
                ADD CONSTRAINT fk_queue_session_id
                FOREIGN KEY (session_id)
                REFERENCES {schema}.sessions(id)
                NOT VALID
                """
            )
        )
        # Validate constraint (scans but allows concurrent reads)
        conn.execute(
            sa.text(
                f"ALTER TABLE {schema}.queue VALIDATE CONSTRAINT fk_queue_session_id"
            )
        )

    # Make documents.embedding non-nullable
    if column_exists("documents", "embedding"):
        make_column_non_nullable_safe("documents", "embedding")


def downgrade() -> None:
    # Make documents.embedding nullable
    if column_exists("documents", "embedding"):
        op.alter_column(
            "documents",
            "embedding",
            nullable=True,
            schema=schema,
        )

    # Drop FK constraint on queue.session_id
    if fk_exists("queue", "fk_queue_session_id"):
        op.drop_constraint(
            "fk_queue_session_id",
            "queue",
            type_="foreignkey",
            schema=schema,
        )

    # Make active_queue_sessions.work_unit_key nullable
    if column_exists("active_queue_sessions", "work_unit_key"):
        op.alter_column(
            "active_queue_sessions",
            "work_unit_key",
            nullable=True,
            schema=schema,
        )

    # Make sessions.workspace_name nullable
    if column_exists("sessions", "workspace_name"):
        op.alter_column(
            "sessions",
            "workspace_name",
            nullable=True,
            schema=schema,
        )

    # Make peers.workspace_name nullable
    if column_exists("peers", "workspace_name"):
        op.alter_column(
            "peers",
            "workspace_name",
            nullable=True,
            schema=schema,
        )
