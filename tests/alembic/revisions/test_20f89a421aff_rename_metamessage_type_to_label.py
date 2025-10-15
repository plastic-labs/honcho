"""Hooks for revision 20f89a421aff (metamessage label rename)."""

from __future__ import annotations

from sqlalchemy import inspect, text

from tests.alembic.constants import METAMESSAGE_PUBLIC_ID
from tests.alembic.registry import register_after_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_after_upgrade("20f89a421aff")
def verify_metamessage_label(verifier: MigrationVerifier) -> None:
    """Ensure metamessage_type column is renamed to label and indexes follow suit."""

    verifier.assert_column_exists("metamessages", "label", nullable=False)
    verifier.assert_indexes_exist(
        [
            ("metamessages", "idx_metamessages_lookup"),
            ("metamessages", "idx_metamessages_user_lookup"),
            ("metamessages", "idx_metamessages_session_lookup"),
            ("metamessages", "idx_metamessages_message_lookup"),
        ]
    )

    inspector = inspect(verifier.conn)
    columns = {
        col["name"]
        for col in inspector.get_columns("metamessages", schema=verifier.schema)
    }
    assert "metamessage_type" not in columns

    row = verifier.conn.execute(
        text(
            f'SELECT "label" FROM "{verifier.schema}"."metamessages" '
            + 'WHERE "public_id" = :public_id'
        ),
        {"public_id": METAMESSAGE_PUBLIC_ID},
    ).one()
    assert row.label == "seed"
