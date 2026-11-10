"""Check partial tombstone indexes and legacy writes in the revision pipeline."""

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from src import models
from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

INDEXES = (
    ("peers", "ix_peers_tenant_workspace_deleted", models.Peer),
    ("sessions", "ix_sessions_tenant_workspace_deleted", models.Session),
)


def seed_legacy_parents(verifier: MigrationVerifier, workspace: str) -> None:
    """Insert using unchanged pre-index columns, without setting tombstones."""
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".workspaces (id, name) VALUES (:id, :ws)'
        ),
        {"id": generate_nanoid(), "ws": workspace},
    )
    for table, _, _ in INDEXES:
        verifier.conn.execute(
            text(
                f'INSERT INTO "{verifier.schema}".{table} (id, name, workspace_name)'
                + " VALUES (:id, 'parent', :ws)"
            ),
            {"id": generate_nanoid(), "ws": workspace},
        )


@register_before_upgrade("4f9c2a7e8d31")
def prepare_tombstoned_parent_indexes(verifier: MigrationVerifier) -> None:
    """Require both indexes to be absent and preserve a legacy-written graph."""
    verifier.assert_indexes_not_exist([(table, name) for table, name, _ in INDEXES])
    seed_legacy_parents(verifier, "before_parent_indexes")


@register_after_upgrade("4f9c2a7e8d31")
def verify_tombstoned_parent_indexes(verifier: MigrationVerifier) -> None:
    """Compare deployed and model index keys, predicates, validity, and old writes."""
    seed_legacy_parents(verifier, "after_parent_indexes")
    verifier.assert_indexes_valid([(table, name) for table, name, _ in INDEXES])
    inspector = verifier.get_inspector()
    for table, name, model in INDEXES:
        index = next(
            i
            for i in inspector.get_indexes(table, schema=verifier.schema)
            if i["name"] == name
        )
        assert index["column_names"] == ["tenant_id", "workspace_name", "name"]
        assert index["unique"] is False
        assert index.get("dialect_options", {}).get("postgresql_where") == (
            "(deleted_at IS NOT NULL)"
        )
        declaration = next(i for i in model.__table__.indexes if i.name == name)
        assert [column.name for column in declaration.columns] == index["column_names"]
        assert (
            str(declaration.dialect_options["postgresql"]["where"])
            == "deleted_at IS NOT NULL"
        )
        rows = verifier.conn.execute(
            text(
                f'SELECT name, deleted_at FROM "{verifier.schema}".{table}'
                + " WHERE workspace_name IN ('before_parent_indexes', 'after_parent_indexes')"
            )
        ).all()
        assert rows == [("parent", None), ("parent", None)]
