"""Hooks for revision 564ba40505c5 (documents session_name column)."""

from __future__ import annotations

from sqlalchemy import Connection, text

from migrations.utils import get_schema
from tests.alembic.constants import DOCUMENT_PUBLIC_ID, SESSION_PUBLIC_ID
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("564ba40505c5")
def seed_document_session_metadata(connection: Connection) -> None:
    """Embed session_name inside internal_metadata prior to the migration."""

    schema = get_schema()
    connection.execute(
        text(
            f"""
            UPDATE "{schema}"."documents"
            SET internal_metadata = jsonb_set(
                COALESCE(internal_metadata, '{{}}'::jsonb),
                '{{session_name}}',
                to_jsonb(CAST(:session_name AS text)),
                true
            )
            WHERE id = :document_id
            """
        ),
        {"session_name": SESSION_PUBLIC_ID, "document_id": DOCUMENT_PUBLIC_ID},
    )


@register_after_upgrade("564ba40505c5")
def verify_document_session_column(verifier: MigrationVerifier) -> None:
    """Ensure the session_name column reflects migrated metadata."""

    verifier.assert_column_exists("documents", "session_name")
    verifier.assert_index_exists("documents", "idx_documents_session_name")
    verifier.assert_constraint_exists(
        "documents", "fk_documents_session_workspace", "foreign_key"
    )

    row = verifier.conn.execute(
        text(
            f'SELECT "session_name" FROM "{verifier.schema}"."documents" '
            + 'WHERE "id" = :document_id'
        ),
        {"document_id": DOCUMENT_PUBLIC_ID},
    ).one()
    assert row.session_name == SESSION_PUBLIC_ID
