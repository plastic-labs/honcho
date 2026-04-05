"""make session_name required on messages

Revision ID: 05486ce795d5
Revises: 917195d9b5e9
Create Date: 2025-07-21 15:34:05.616578

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from nanoid import generate as generate_nanoid

from migrations.utils import constraint_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "05486ce795d5"
down_revision: str | None = "917195d9b5e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    """Make session_name required on messages and message_embeddings tables.

    This migration:
    1. Identifies messages without a session_name
    2. Creates a default session for each workspace that has messages without a session_name (using peer_name as session_name)
    3. Assigns messages without a session_name to the default sessions
    4. Makes session_name non-nullable on messages table
    5. Makes session_name non-nullable on message_embeddings table
    6. Updates the collections table name_length check constraint from 512 to 1025 to support peer-namespaced collections
    """
    conn = op.get_bind()

    # Step 1: Check if there are any orphaned messages
    orphaned_count = conn.execute(
        sa.text(f"SELECT COUNT(*) FROM {schema}.messages WHERE session_name IS NULL")
    ).scalar()

    print(f"Found {orphaned_count} orphaned messages without session_name")

    if orphaned_count and orphaned_count > 0:
        # Step 2: Get unique workspace-peer combinations that have orphaned messages
        workspace_peer_result = conn.execute(
            sa.text(f"""
                SELECT DISTINCT workspace_name, peer_name
                FROM {schema}.messages
                WHERE session_name IS NULL
            """)
        )
        workspace_peer_combinations = [
            (row[0], row[1]) for row in workspace_peer_result.fetchall()
        ]
        print(
            f"Found {len(workspace_peer_combinations)} workspace-peer combinations with orphaned messages"
        )
        # Step 3: Create individual sessions for each workspace-peer combination
        for workspace_name, peer_name in workspace_peer_combinations:
            default_session_name = peer_name
            default_session_id = generate_nanoid()

            print(
                f"Creating default session '{default_session_name}' for workspace '{workspace_name}' and peer '{peer_name}'"
            )

            # Create the default session (ON CONFLICT DO NOTHING to handle existing sessions)
            op.execute(
                sa.text(f"""
                    INSERT INTO {schema}.sessions (id, name, workspace_name, is_active, metadata, internal_metadata, configuration, created_at)
                    VALUES (
                        '{default_session_id}',
                        '{default_session_name}',
                        '{workspace_name}',
                        true,
                        '{{}}',
                        '{{"migration_note" : "Default session created for orphaned messages from peer {peer_name}"}}',
                        '{{}}',
                        NOW()
                    )
                    ON CONFLICT (name, workspace_name) DO NOTHING
                """)
            )

            # Step 3.5: Create session peer association for this peer (ON CONFLICT DO NOTHING to handle existing associations)
            op.execute(
                sa.text(f"""
                    INSERT INTO {schema}.session_peers (workspace_name, session_name, peer_name, configuration, internal_metadata, joined_at, left_at)
                    VALUES (
                        '{workspace_name}',
                        '{default_session_name}',
                        '{peer_name}',
                        '{{}}',
                        '{{}}',
                        NOW(),
                        NULL
                    )
                    ON CONFLICT (workspace_name, session_name, peer_name) DO NOTHING
                """)
            )
            print(
                f"Created session peer association for peer '{peer_name}' in default session '{default_session_name}'"
            )

            # Step 4: Assign orphaned messages for this peer to the default session in batches
            batch_size = 5000
            while True:
                result = conn.execute(
                    sa.text(f"""
                        WITH batch AS (
                            SELECT id
                            FROM {schema}.messages
                            WHERE workspace_name = :workspace_name
                            AND peer_name = :peer_name
                            AND session_name IS NULL
                            ORDER BY id
                            LIMIT :batch_size
                        )
                        UPDATE {schema}.messages m
                        SET session_name = :default_session_name
                        FROM batch
                        WHERE m.id = batch.id
                    """),
                    {
                        "workspace_name": workspace_name,
                        "peer_name": peer_name,
                        "default_session_name": default_session_name,
                        "batch_size": batch_size,
                    },
                )
                if result.rowcount == 0:
                    break

            # Step 4.5: Handle orphaned message embeddings for this peer in batches
            batch_size = 5000
            while True:
                result = conn.execute(
                    sa.text(f"""
                        WITH batch AS (
                            SELECT id
                            FROM {schema}.message_embeddings
                            WHERE workspace_name = :workspace_name
                            AND peer_name = :peer_name
                            AND session_name IS NULL
                            ORDER BY id
                            LIMIT :batch_size
                        )
                        UPDATE {schema}.message_embeddings me
                        SET session_name = :default_session_name
                        FROM batch
                        WHERE me.id = batch.id
                    """),
                    {
                        "workspace_name": workspace_name,
                        "peer_name": peer_name,
                        "default_session_name": default_session_name,
                        "batch_size": batch_size,
                    },
                )
                if result.rowcount == 0:
                    break

    # Step 5: Sanity check that no orphaned messages remain
    remaining_orphaned = conn.execute(
        sa.text(f"SELECT COUNT(*) FROM {schema}.messages WHERE session_name IS NULL")
    ).scalar()

    if remaining_orphaned and remaining_orphaned > 0:
        raise Exception(
            f"Still have {remaining_orphaned} orphaned messages after migration"
        )

    # Step 6: Make session_name NOT NULL on messages table
    print("Making session_name NOT NULL on messages table")
    op.alter_column("messages", "session_name", nullable=False, schema=schema)

    # Step 7: Sanity check that no orphaned message embeddings remain
    remaining_orphaned_embeddings = conn.execute(
        sa.text(
            f"SELECT COUNT(*) FROM {schema}.message_embeddings WHERE session_name IS NULL"
        )
    ).scalar()

    if remaining_orphaned_embeddings and remaining_orphaned_embeddings > 0:
        raise Exception(
            f"Still have {remaining_orphaned_embeddings} orphaned message embeddings after migration"
        )

    # Step 8: Make session_name NOT NULL on message_embeddings table
    print("Making session_name NOT NULL on message_embeddings table")
    op.alter_column("message_embeddings", "session_name", nullable=False, schema=schema)

    # Step 9: Update the collections table name_length check constraint from 512 to 1025
    print("Updating collections table name_length check constraint from 512 to 1025")

    # Check for both old name and naming-convention-generated name
    if constraint_exists("collections", "name_length", "check"):
        op.drop_constraint("name_length", "collections", schema=schema)
    elif constraint_exists("collections", "ck_collections_name_length", "check"):
        op.drop_constraint("ck_collections_name_length", "collections", schema=schema)

    op.create_check_constraint(
        "name_length", "collections", "length(name) <= 1025", schema=schema
    )


def downgrade() -> None:
    """Reverse the migration by making session_name nullable again.

    Note: This will not restore the original orphaned messages to their previous state, nor will it
    delete the default sessions created during the upgrade.
    """
    # Step 1: Revert the collections table name_length check constraint from 1025 back to 512
    print(
        "Reverting collections table name_length check constraint from 1025 back to 512"
    )

    # Check for both old name and naming-convention-generated name
    if constraint_exists("collections", "name_length", "check"):
        op.drop_constraint("name_length", "collections", schema=schema)
    elif constraint_exists("collections", "ck_collections_name_length", "check"):
        op.drop_constraint("ck_collections_name_length", "collections", schema=schema)

    op.create_check_constraint(
        "name_length", "collections", "length(name) <= 512", schema=schema
    )

    print("Making session_name nullable again on message_embeddings table")
    op.alter_column("message_embeddings", "session_name", nullable=True, schema=schema)

    print("Making session_name nullable again on messages table")
    op.alter_column("messages", "session_name", nullable=True, schema=schema)
