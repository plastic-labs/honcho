"""add document_source_messages table

Edge table linking an explicit conclusion to the messages the deriver cited
as its evidence. DDL only: citations are model-reported per derivation, so
there is nothing to backfill for rows written before this revision. Those
rows keep their batch-level ``internal_metadata.message_ids`` and surface
``source_message_ids`` as null.

Revision ID: c4d8e2f6a1b3
Revises: a7c3e9f1b2d4
Create Date: 2026-09-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, table_exists

# revision identifiers, used by Alembic.
revision: str = "c4d8e2f6a1b3"
down_revision: str | None = "a7c3e9f1b2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not table_exists("document_source_messages", inspector):
        op.create_table(
            "document_source_messages",
            sa.Column("derived_id", sa.TEXT, nullable=False),
            sa.Column("message_id", sa.TEXT, nullable=False),
            sa.Column("position", sa.Integer, nullable=False, server_default="0"),
            sa.Column("workspace_name", sa.TEXT, nullable=False),
            sa.PrimaryKeyConstraint("derived_id", "message_id"),
            sa.ForeignKeyConstraint(
                ["derived_id"],
                [f"{schema}.documents.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["message_id"],
                [f"{schema}.messages.public_id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["workspace_name"],
                [f"{schema}.workspaces.name"],
            ),
            sa.CheckConstraint("length(message_id) = 21", name="message_id_length"),
            sa.CheckConstraint(
                "message_id ~ '^[A-Za-z0-9_-]+$'", name="message_id_format"
            ),
            schema=schema,
        )
        # Reverse traversal ("which conclusions cite this message?"); also
        # serves the ON DELETE CASCADE from messages.
        op.create_index(
            "ix_document_source_messages_message_id",
            "document_source_messages",
            ["message_id", "workspace_name"],
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_exists("document_source_messages", inspector):
        op.drop_table("document_source_messages", schema=schema)
