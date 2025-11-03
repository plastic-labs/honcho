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
    constraint_exists,
    fk_exists,
    get_schema,
    index_exists,
    make_column_non_nullable_safe,
)

# revision identifiers, used by Alembic.
revision: str = "066e87ca5b07"
down_revision: str | None = "bb6fb3a7a643"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """
    The application code has been previously updated to ensure none of the following columns have NULL values but the actual DB schema is out of sync with our SQLAlchemy model definitions.
    This migration fixes this by making the columns non-nullable using a non-blocking approach to minimize lock duration.
    """
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

    # Make documents.embedding non-nullable
    if column_exists("documents", "embedding"):
        make_column_non_nullable_safe("documents", "embedding")

    # Add primary key constraint to message_embeddings.id
    if column_exists("message_embeddings", "id") and not constraint_exists(
        "message_embeddings", "pk_message_embeddings", "primary"
    ):
        conn.execute(
            sa.text(
                f"""
                ALTER TABLE {schema}.message_embeddings
                ADD CONSTRAINT pk_message_embeddings
                PRIMARY KEY (id)
                """
            )
        )

    # Rename indexes on peers table
    inspector = sa.inspect(conn)
    index_renames = [
        ("peers", "ix_users_created_at", "ix_peers_created_at"),
        ("peers", "ix_users_name", "ix_peers_name"),
        ("workspaces", "ix_apps_created_at", "ix_workspaces_created_at"),
        ("workspaces", "ix_apps_name", "ix_workspaces_name"),
    ]
    for table_name, old_name, new_name in index_renames:
        if index_exists(table_name, old_name, inspector):
            conn.execute(
                sa.text(f"ALTER INDEX {schema}.{old_name} RENAME TO {new_name}")
            )

    # Drop redundant indexes
    if index_exists("workspaces", "ix_apps_public_id", inspector):
        op.drop_index("ix_apps_public_id", table_name="workspaces", schema=schema)
    if index_exists("peers", "ix_users_public_id", inspector):
        op.drop_index("ix_users_public_id", table_name="peers", schema=schema)
    if index_exists("sessions", "ix_sessions_public_id", inspector):
        op.drop_index("ix_sessions_public_id", table_name="sessions", schema=schema)
    if index_exists("documents", "ix_documents_public_id", inspector):
        op.drop_index("ix_documents_public_id", table_name="documents", schema=schema)
    if index_exists("collections", "ix_collections_public_id", inspector):
        op.drop_index(
            "ix_collections_public_id", table_name="collections", schema=schema
        )

    # Drop redundant unique constraints
    if constraint_exists("workspaces", "uq_apps_public_id", "unique", inspector):
        op.drop_constraint(
            "uq_apps_public_id", "workspaces", type_="unique", schema=schema
        )

    if constraint_exists("peers", "uq_users_public_id", "unique", inspector):
        op.drop_constraint("uq_users_public_id", "peers", type_="unique", schema=schema)

    if constraint_exists("sessions", "uq_sessions_public_id", "unique", inspector):
        op.drop_constraint(
            "uq_sessions_public_id", "sessions", type_="unique", schema=schema
        )

    if constraint_exists(
        "collections", "uq_collections_public_id", "unique", inspector
    ):
        op.drop_constraint(
            "uq_collections_public_id", "collections", type_="unique", schema=schema
        )

    if constraint_exists("documents", "uq_documents_public_id", "unique", inspector):
        op.drop_constraint(
            "uq_documents_public_id", "documents", type_="unique", schema=schema
        )

    # Drop unnecessary index on active queue
    if index_exists(
        "active_queue_sessions",
        f"ix_{schema}_active_queue_sessions_work_unit_key",
        inspector,
    ):
        op.drop_index(
            f"ix_{schema}_active_queue_sessions_work_unit_key",
            table_name="active_queue_sessions",
            schema=schema,
        )

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

    # Create missing indexes
    if not index_exists("peers", "ix_peers_workspace_name", inspector):
        op.create_index(
            "ix_peers_workspace_name", "peers", ["workspace_name"], schema=schema
        )

    if not index_exists("collections", "ix_collections_workspace_name", inspector):
        op.create_index(
            "ix_collections_workspace_name",
            "collections",
            ["workspace_name"],
            schema=schema,
        )

    if not index_exists("documents", "ix_documents_workspace_name", inspector):
        op.create_index(
            "ix_documents_workspace_name",
            "documents",
            ["workspace_name"],
            schema=schema,
        )

    if not fk_exists("session_peers", "fk_session_peers_workspace_name", inspector):
        # Add constraint without validation (fast, doesn't scan)
        conn.execute(
            sa.text(
                f"""
                ALTER TABLE {schema}.session_peers
                ADD CONSTRAINT fk_session_peers_workspace_name
                FOREIGN KEY (workspace_name)
                REFERENCES {schema}.workspaces(name)
                NOT VALID
                """
            )
        )
        # Validate constraint (scans but allows concurrent reads)
        conn.execute(
            sa.text(
                f"ALTER TABLE {schema}.session_peers VALIDATE CONSTRAINT fk_session_peers_workspace_name"
            )
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if fk_exists("session_peers", "fk_session_peers_workspace_name", inspector):
        op.drop_constraint(
            "fk_session_peers_workspace_name",
            table_name="session_peers",
            type_="foreignkey",
            schema=schema,
        )

    if index_exists("documents", "ix_documents_workspace_name", inspector):
        op.drop_index(
            "ix_documents_workspace_name", table_name="documents", schema=schema
        )

    if index_exists("peers", "ix_peers_workspace_name", inspector):
        op.drop_index("ix_peers_workspace_name", table_name="peers", schema=schema)
    if index_exists("collections", "ix_collections_workspace_name", inspector):
        op.drop_index(
            "ix_collections_workspace_name", table_name="collections", schema=schema
        )

    # First, drop the FK constraint (we'll recreate it later if needed)
    if fk_exists("queue", "fk_queue_session_id"):
        op.drop_constraint(
            "fk_queue_session_id",
            "queue",
            type_="foreignkey",
            schema=schema,
        )

    if not index_exists(
        "active_queue_sessions",
        f"ix_{schema}_active_queue_sessions_work_unit_key",
        inspector,
    ):
        op.create_index(
            f"ix_{schema}_active_queue_sessions_work_unit_key",
            table_name="active_queue_sessions",
            columns=["work_unit_key"],
            schema=schema,
        )

    # Recreate the redundant unique constraints
    if not constraint_exists("sessions", "uq_sessions_public_id", "unique", inspector):
        op.create_unique_constraint(
            "uq_sessions_public_id", "sessions", ["id"], schema=schema
        )
    if not constraint_exists("peers", "uq_users_public_id", "unique", inspector):
        op.create_unique_constraint(
            "uq_users_public_id", "peers", ["id"], schema=schema
        )
    if not constraint_exists("workspaces", "uq_apps_public_id", "unique", inspector):
        op.create_unique_constraint(
            "uq_apps_public_id", "workspaces", ["id"], schema=schema
        )

    if not constraint_exists(
        "collections", "uq_collections_public_id", "unique", inspector
    ):
        op.create_unique_constraint(
            "uq_collections_public_id", "collections", ["id"], schema=schema
        )
    if not constraint_exists(
        "documents", "uq_documents_public_id", "unique", inspector
    ):
        op.create_unique_constraint(
            "uq_documents_public_id", "documents", ["id"], schema=schema
        )

    # Recreate the redundant indexes
    if not index_exists("sessions", "ix_sessions_public_id", inspector):
        op.create_index("ix_sessions_public_id", "sessions", ["id"], schema=schema)
    if not index_exists("peers", "ix_users_public_id", inspector):
        op.create_index("ix_users_public_id", "peers", ["id"], schema=schema)
    if not index_exists("workspaces", "ix_apps_public_id", inspector):
        op.create_index("ix_apps_public_id", "workspaces", ["id"], schema=schema)
    if not index_exists("documents", "ix_documents_public_id", inspector):
        op.create_index("ix_documents_public_id", "documents", ["id"], schema=schema)
    if not index_exists("collections", "ix_collections_public_id", inspector):
        op.create_index(
            "ix_collections_public_id",
            "collections",
            ["id"],
            schema=schema,
        )

    # Rename indexes on peers table back to original names
    index_renames = [
        ("peers", "ix_peers_created_at", "ix_users_created_at"),
        ("peers", "ix_peers_name", "ix_users_name"),
        ("workspaces", "ix_workspaces_created_at", "ix_apps_created_at"),
        ("workspaces", "ix_workspaces_name", "ix_apps_name"),
    ]
    for table_name, new_name, old_name in index_renames:
        if index_exists(table_name, new_name, inspector):
            conn.execute(
                sa.text(f"ALTER INDEX {schema}.{new_name} RENAME TO {old_name}")
            )

    # Drop primary key constraint from message_embeddings.id
    if constraint_exists("message_embeddings", "pk_message_embeddings", "primary"):
        op.drop_constraint(
            "pk_message_embeddings", "message_embeddings", "primary", schema=schema
        )

    # Make documents.embedding nullable
    if column_exists("documents", "embedding"):
        op.alter_column(
            "documents",
            "embedding",
            nullable=True,
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
