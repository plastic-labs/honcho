"""Hooks for revision 43d77d0c5846 (add_tenant_derivation_paused)."""

from __future__ import annotations

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEX_NAME = "ix_tenants_derivation_paused"
# Seeded before the upgrade: an existing tenant must come out unpaused, or the
# upgrade itself would change what the fleet claims.
EXISTING_TENANT = "t-existing-before-pause-bit"


@register_before_upgrade("43d77d0c5846")
def prepare_add_tenant_derivation_paused(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("tenants", "derivation_paused", exists=False)
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}"."tenants" ("tenant_id", "tier") '
            + "VALUES (:tid, 'dedicated')"
        ),
        {"tid": EXISTING_TENANT},
    )


@register_after_upgrade("43d77d0c5846")
def verify_add_tenant_derivation_paused(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("tenants", "derivation_paused", nullable=False)
    verifier.assert_indexes_exist([("tenants", INDEX_NAME)])

    paused = verifier.conn.execute(
        text(
            f'SELECT "derivation_paused" FROM "{verifier.schema}"."tenants" '
            + "WHERE tenant_id = :tid"
        ),
        {"tid": EXISTING_TENANT},
    ).scalar_one()
    assert paused is False, "an existing tenant must not come out of the upgrade paused"

    # The default applies to rows inserted without the column, too.
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}"."tenants" ("tenant_id", "tier") '
            + "VALUES (:tid, 'shared')"
        ),
        {"tid": f"{EXISTING_TENANT}-after"},
    )
    inserted = verifier.conn.execute(
        text(
            f'SELECT "derivation_paused" FROM "{verifier.schema}"."tenants" '
            + "WHERE tenant_id = :tid"
        ),
        {"tid": f"{EXISTING_TENANT}-after"},
    ).scalar_one()
    assert inserted is False
