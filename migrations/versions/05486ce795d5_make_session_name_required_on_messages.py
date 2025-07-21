"""make session_name required on messages

Revision ID: 05486ce795d5
Revises: 917195d9b5e9
Create Date: 2025-07-21 15:34:05.616578

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from nanoid import generate as generate_nanoid

from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "05486ce795d5"
down_revision: str | None = "917195d9b5e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = settings.DB.SCHEMA


def upgrade() -> None:
    """Make session_name required on messages table.

    This migration:
    1. Identifies messages without a session_name
    2. Creates a default session for each workspace that has messages without a session_name
    3. Assigns messages without a session_name to the default sessions
    4. Makes session_name non-nullable
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
            default_session_name = f"default_session_{peer_name}"
            default_session_id = generate_nanoid()

            print(
                f"Creating default session '{default_session_name}' for workspace '{workspace_name}' and peer '{peer_name}'"
            )

            # Create the default session
            op.execute(
                sa.text(f"""
                    INSERT INTO {schema}.sessions (id, name, workspace_name, is_active, metadata, internal_metadata, configuration, created_at)
                    VALUES (
                        '{default_session_id}',
                        '{default_session_name}',
                        '{workspace_name}',
                        true,
                        '{{}}',
                        '{{"migration_note": "Default session created for orphaned messages from peer {peer_name}"}}',
                        '{{}}',
                        NOW()
                    )
                """)
            )

            # Step 3.5: Create session peer association for this peer
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
                """)
            )
            print(
                f"Created session peer association for peer '{peer_name}' in default session '{default_session_name}'"
            )

            # Step 4: Assign orphaned messages for this peer to the default session
            op.execute(
                sa.text(f"""
                    UPDATE {schema}.messages 
                    SET session_name = '{default_session_name}'
                    WHERE workspace_name = '{workspace_name}' 
                    AND peer_name = '{peer_name}'
                    AND session_name IS NULL
                """)
            )

    # Step 5: Sanity check that no orphaned messages remain
    remaining_orphaned = conn.execute(
        sa.text(f"SELECT COUNT(*) FROM {schema}.messages WHERE session_name IS NULL")
    ).scalar()

    if remaining_orphaned and remaining_orphaned > 0:
        raise Exception(
            f"Still have {remaining_orphaned} orphaned messages after migration"
        )

    # Step 6: Make session_name NOT NULL
    print("Making session_name NOT NULL")
    op.alter_column("messages", "session_name", nullable=False, schema=schema)


def downgrade() -> None:
    """Reverse the migration by making session_name nullable again.

    Note: This will not restore the original orphaned messages to their previous state, nor will it
    delete the default sessions created during the upgrade.
    """
    print("Making session_name nullable again")
    op.alter_column("messages", "session_name", nullable=True, schema=schema)
