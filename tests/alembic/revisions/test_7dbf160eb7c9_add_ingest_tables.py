"""Check the ingest tables and the auto flag column in the revision pipeline."""

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

TABLES = {
    "ingest_records": (
        models.IngestRecord,
        "uq_ingest_records_identity",
        [
            "ix_ingest_records_session",
            "ix_ingest_records_pending",
            "ix_ingest_records_deferred",
            "ix_ingest_records_author_binding",
        ],
    ),
    "ingest_session_bindings": (
        models.IngestSessionBinding,
        "uq_ingest_session_bindings_key",
        ["ix_ingest_session_bindings_session"],
    ),
    "ingest_peer_bindings": (
        models.IngestPeerBinding,
        "uq_ingest_peer_bindings_key",
        ["ix_ingest_peer_bindings_peer"],
    ),
}


def seed_workspace(verifier: MigrationVerifier, name: str) -> None:
    """Insert a workspace through the columns every revision since tenancy has."""
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".workspaces (id, name) VALUES (:id, :ws)'
        ),
        {"id": generate_nanoid(), "ws": name},
    )


@register_before_upgrade("7dbf160eb7c9")
def prepare_ingest_tables(verifier: MigrationVerifier) -> None:
    """The tables and the column are absent, and an existing workspace is kept."""
    for table in TABLES:
        verifier.assert_table_exists(table, exists=False)
    verifier.assert_column_exists("workspaces", "auto_enabled_at", exists=False)
    seed_workspace(verifier, "before_ingest_tables")


@register_after_upgrade("7dbf160eb7c9")
def verify_ingest_tables(verifier: MigrationVerifier) -> None:
    """Tables, keys and indexes match the models; the flag column starts empty."""
    verifier.assert_column_exists("workspaces", "auto_enabled_at", nullable=True)
    flags = verifier.conn.execute(
        text(
            f'SELECT auto_enabled_at FROM "{verifier.schema}".workspaces'
            + " WHERE name = 'before_ingest_tables'"
        )
    ).all()
    assert flags == [(None,)]

    inspector = verifier.get_inspector()
    for table, (model, unique_name, index_names) in TABLES.items():
        verifier.assert_table_exists(table)
        assert {
            c["name"] for c in inspector.get_columns(table, schema=verifier.schema)
        } == {column.name for column in model.__table__.columns}
        assert inspector.get_pk_constraint(table, schema=verifier.schema)[
            "constrained_columns"
        ] == ["tenant_id", "id"]
        verifier.assert_constraint_exists(table, unique_name, "unique")
        verifier.assert_indexes_valid([(table, name) for name in index_names])
        # Only the tenants key: no foreign key to sessions, peers or workspaces,
        # so ingest never locks those parent rows (src/models.py).
        foreign_keys = inspector.get_foreign_keys(table, schema=verifier.schema)
        assert [fk["referred_table"] for fk in foreign_keys] == ["tenants"]

    partial = {
        i["name"]: i.get("dialect_options", {}).get("postgresql_where")
        for i in inspector.get_indexes("ingest_records", schema=verifier.schema)
    }
    assert partial["ix_ingest_records_pending"] == "(derivation = 'pending'::text)"
    assert partial["ix_ingest_records_deferred"] == "(derivation = 'deferred'::text)"
    assert partial["ix_ingest_records_author_binding"] == (
        "(author_binding_id IS NOT NULL)"
    )
