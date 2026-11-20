"""Check the live-sync migration: the dry-run marker, tombstones, sync states."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

DERIVED = "live_sync_derived"
DRY_RUN = "live_sync_dry_run"
# Single-tenant queue rows carry no tenant: still evidence of work queued.
QUEUED = "live_sync_queued"


def _record(verifier: MigrationVerifier, workspace: str, derivation: str) -> None:
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".ingest_records (tenant_id, id,'
            + " workspace_name, source, source_instance, identity_kind, identity,"
            + " session_name, content_sha256, derivation) VALUES ((SELECT tenant_id"
            + f' FROM "{verifier.schema}".workspaces WHERE name = :ws), :id, :ws,'
            + " 'discord', 'g', 'source_id', :id, 'chan', 'hash', :derivation)"
        ),
        {"id": generate_nanoid(), "ws": workspace, "derivation": derivation},
    )


@register_before_upgrade("c4d9e1a7b2f6")
def prepare_live_sync(verifier: MigrationVerifier) -> None:
    """Auto workspaces with queued work, one with only a tenant-less queue row,
    and one still in its dry run."""
    verifier.assert_table_exists("ingest_sync_states", exists=False)
    verifier.assert_column_exists("workspaces", "ingest_derived_at", exists=False)
    for name in (DERIVED, DRY_RUN, QUEUED):
        verifier.conn.execute(
            text(
                f'INSERT INTO "{verifier.schema}".workspaces (id, name, auto_enabled_at)'
                + " VALUES (:id, :ws, now())"
            ),
            {"id": generate_nanoid(), "ws": name},
        )
    _record(verifier, DERIVED, "queued")
    _record(verifier, DRY_RUN, "deferred")
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".queue (work_unit_key, task_type,'
            + " payload, workspace_name) VALUES ('k', 'representation', '{}', :ws)"
        ),
        {"ws": QUEUED},
    )


@register_after_upgrade("c4d9e1a7b2f6")
def verify_live_sync(verifier: MigrationVerifier) -> None:
    """The marker is backfilled only where work was queued; the table and checks hold."""
    rows = verifier.conn.execute(
        text(
            f'SELECT name, ingest_derived_at IS NOT NULL FROM "{verifier.schema}"'
            + ".workspaces WHERE name IN (:a, :b, :c)"
        ),
        {"a": DERIVED, "b": DRY_RUN, "c": QUEUED},
    ).all()
    marked = {name: is_set for name, is_set in rows}
    assert marked == {DERIVED: True, DRY_RUN: False, QUEUED: True}

    table = "ingest_sync_states"
    verifier.assert_table_exists(table)
    inspector = verifier.get_inspector()
    assert {
        c["name"] for c in inspector.get_columns(table, schema=verifier.schema)
    } == {column.name for column in models.IngestSyncState.__table__.columns}
    verifier.assert_constraint_exists(table, "uq_ingest_sync_states_source", "unique")
    verifier.assert_constraint_exists(
        "ingest_records", "ck_ingest_records_tombstone_or_complete", "check"
    )

    # A tombstone needs no session or hash; any other record does.
    tenant = verifier.conn.scalar(
        text(f'SELECT tenant_id FROM "{verifier.schema}".workspaces WHERE name = :ws'),
        {"ws": DRY_RUN},
    )
    insert = text(
        f'INSERT INTO "{verifier.schema}".ingest_records (tenant_id, id,'
        + " workspace_name, source, source_instance, identity_kind, identity,"
        + " derivation) VALUES (:tenant, :id, :ws, 'discord', 'g', 'source_id',"
        + " :id, :derivation)"
    )
    verifier.conn.execute(
        insert,
        {
            "tenant": tenant,
            "id": generate_nanoid(),
            "ws": DRY_RUN,
            "derivation": "deleted",
        },
    )
    savepoint = verifier.conn.begin_nested()
    with pytest.raises(IntegrityError):
        verifier.conn.execute(
            insert,
            {
                "tenant": tenant,
                "id": generate_nanoid(),
                "ws": DRY_RUN,
                "derivation": "deferred",
            },
        )
    savepoint.rollback()
