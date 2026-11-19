"""Check the ingest suggestions table in the revision pipeline."""

from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("b8c2e4f1a9d3")
def prepare_ingest_suggestions(verifier: MigrationVerifier) -> None:
    """The table is absent."""
    verifier.assert_table_exists("ingest_suggestions", exists=False)


@register_after_upgrade("b8c2e4f1a9d3")
def verify_ingest_suggestions(verifier: MigrationVerifier) -> None:
    """Columns, keys and constraints match the model; only the tenants key."""
    table = "ingest_suggestions"
    verifier.assert_table_exists(table)
    inspector = verifier.get_inspector()
    assert {
        c["name"] for c in inspector.get_columns(table, schema=verifier.schema)
    } == {column.name for column in models.IngestSuggestion.__table__.columns}
    assert inspector.get_pk_constraint(table, schema=verifier.schema)[
        "constrained_columns"
    ] == ["tenant_id", "id"]
    verifier.assert_constraint_exists(table, "uq_ingest_suggestions_subject", "unique")
    foreign_keys = inspector.get_foreign_keys(table, schema=verifier.schema)
    assert [fk["referred_table"] for fk in foreign_keys] == ["tenants"]
