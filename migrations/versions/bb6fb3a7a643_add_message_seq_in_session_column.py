"""add message_seq_in_session column

Revision ID: bb6fb3a7a643
Revises: 76ffba56fe8c
Create Date: 2025-10-20 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, constraint_exists, get_schema, index_exists

# revision identifiers, used by Alembic.
revision: str = "bb6fb3a7a643"
down_revision: str | None = "76ffba56fe8c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    if not column_exists("messages", "message_seq_in_session"):
        op.add_column(
            "messages",
            sa.Column("message_seq_in_session", sa.BigInteger(), nullable=True),
            schema=schema,
        )

        op.execute(
            f"""
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY workspace_name, session_name
                           ORDER BY id
                       ) AS seq
                FROM {schema}.messages
            )
            UPDATE {schema}.messages AS m
            SET message_seq_in_session = ranked.seq
            FROM ranked
            WHERE m.id = ranked.id
            """
        )
        op.alter_column(
            "messages",
            "message_seq_in_session",
            nullable=False,
            schema=schema,
        )

    if not index_exists("messages", "ix_messages_message_seq_in_session"):
        op.create_index(
            "ix_messages_message_seq_in_session",
            "messages",
            ["message_seq_in_session"],
            schema=schema,
        )

    if not constraint_exists("messages", "uq_messages_session_seq", "unique"):
        op.create_unique_constraint(
            "uq_messages_session_seq",
            "messages",
            ["workspace_name", "session_name", "message_seq_in_session"],
            schema=schema,
        )


def downgrade() -> None:
    schema = get_schema()

    if constraint_exists("messages", "uq_messages_session_seq", "unique"):
        op.drop_constraint(
            "uq_messages_session_seq",
            "messages",
            type_="unique",
            schema=schema,
        )

    if index_exists("messages", "ix_messages_message_seq_in_session"):
        op.drop_index(
            "ix_messages_message_seq_in_session",
            table_name="messages",
            schema=schema,
        )

    if column_exists("messages", "message_seq_in_session"):
        op.drop_column("messages", "message_seq_in_session", schema=schema)
