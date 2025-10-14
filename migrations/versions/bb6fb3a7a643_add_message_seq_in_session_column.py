"""add seq_in_session column to messages table

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

BATCH_SIZE = 10_000


def upgrade() -> None:
    if not column_exists("messages", "seq_in_session"):
        op.add_column(
            "messages",
            sa.Column("seq_in_session", sa.BigInteger(), nullable=True),
            schema=schema,
        )

        conn = op.get_bind()
        preparer = conn.dialect.identifier_preparer
        messages_table = sa.Table("messages", sa.MetaData(), schema=schema)
        qualified_messages = preparer.format_table(messages_table)
        id_col = preparer.quote("id")
        workspace_col = preparer.quote("workspace_name")
        session_col = preparer.quote("session_name")
        seq_col = preparer.quote("seq_in_session")

        distinct_sessions = conn.execute(
            sa.text(
                f"""
                SELECT DISTINCT
                    {workspace_col} AS workspace_name,
                    {session_col} AS session_name
                FROM {qualified_messages}
                """
            )
        )

        update_stmt = sa.text(
            f"""
            WITH params AS (
                SELECT
                    :workspace_name AS workspace_name,
                    :session_name AS session_name,
                    COALESCE(
                        (
                            SELECT MAX({seq_col})
                            FROM {qualified_messages}
                            WHERE {workspace_col} = :workspace_name
                              AND {session_col} = :session_name
                        ),
                        0
                    ) AS offset
            ),
            batch AS (
                SELECT
                    m.{id_col} AS id,
                    ROW_NUMBER() OVER (ORDER BY m.{id_col}) AS rn,
                    params.offset
                FROM {qualified_messages} AS m
                JOIN params ON TRUE
                WHERE m.{workspace_col} = params.workspace_name
                  AND m.{session_col} = params.session_name
                  AND m.{seq_col} IS NULL
                ORDER BY m.{id_col}
                LIMIT :batch_size
            )
            UPDATE {qualified_messages} AS m
            SET {seq_col} = batch.rn + batch.offset
            FROM batch
            WHERE m.{id_col} = batch.id
            """
        )

        for workspace_name, session_name in distinct_sessions:
            while True:
                result = conn.execute(
                    update_stmt,
                    {
                        "workspace_name": workspace_name,
                        "session_name": session_name,
                        "batch_size": BATCH_SIZE,
                    },
                )
                updated_rows = result.rowcount or 0
                result.close()
                if updated_rows == 0:
                    break
        distinct_sessions.close()

        op.alter_column(
            "messages",
            "seq_in_session",
            nullable=False,
            schema=schema,
        )

    if not index_exists("messages", "ix_messages_seq_in_session"):
        op.create_index(
            "ix_messages_seq_in_session",
            "messages",
            ["seq_in_session"],
            schema=schema,
        )

    if not constraint_exists("messages", "uq_messages_session_seq", "unique"):
        op.create_unique_constraint(
            "uq_messages_session_seq",
            "messages",
            ["workspace_name", "session_name", "seq_in_session"],
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

    if index_exists("messages", "ix_messages_seq_in_session"):
        op.drop_index(
            "ix_messages_seq_in_session",
            table_name="messages",
            schema=schema,
        )

    if column_exists("messages", "seq_in_session"):
        op.drop_column("messages", "seq_in_session", schema=schema)
