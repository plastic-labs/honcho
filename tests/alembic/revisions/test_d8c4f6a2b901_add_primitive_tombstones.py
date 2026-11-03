"""Schema and legacy-writer compatibility for inactive primitive tombstones."""

from nanoid import generate as generate_nanoid
from sqlalchemy import DateTime, text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

TABLES = ("workspaces", "peers", "sessions", "messages")


def _seed_legacy_rows(verifier: MigrationVerifier, workspace: str) -> None:
    """Use only pre-migration columns, both before and after the upgrade."""
    for table, columns, values in (
        ("workspaces", "id, name, deletion_generation", ":uid, :ws, 7"),
        ("peers", "id, name, workspace_name", ":uid, 'author', :ws"),
        (
            "sessions",
            "id, name, workspace_name, deletion_generation",
            ":uid, 'session', :ws, 11",
        ),
        (
            "messages",
            "public_id, workspace_name, session_name, peer_name, content, token_count, seq_in_session",
            ":uid, :ws, 'session', 'author', 'legacy content', 2, 1",
        ),
    ):
        verifier.conn.execute(
            text(
                f'INSERT INTO "{verifier.schema}".{table} ({columns}) VALUES ({values})'
            ),
            {"uid": generate_nanoid(), "ws": workspace},
        )


def _assert_document_deleted_at_index_preserved(verifier: MigrationVerifier) -> None:
    """The partial index from 119a52b73c60 stays valid with its original predicate."""
    row = verifier.conn.execute(
        text(
            "SELECT i.indisvalid, pg_get_expr(i.indpred, i.indrelid) FROM pg_index i"
            + " JOIN pg_class c ON c.oid = i.indexrelid"
            + " JOIN pg_namespace n ON n.oid = c.relnamespace"
            + " WHERE n.nspname = :schema AND c.relname = 'ix_documents_deleted_at'"
        ),
        {"schema": verifier.schema},
    ).one()
    assert tuple(row) == (True, "(deleted_at IS NOT NULL)")


@register_before_upgrade("d8c4f6a2b901")
def prepare_primitive_tombstones(verifier: MigrationVerifier) -> None:
    for table in TABLES:
        verifier.assert_column_exists(table, "deleted_at", exists=False)
    _seed_legacy_rows(verifier, "before_tombstones")
    _assert_document_deleted_at_index_preserved(verifier)


@register_after_upgrade("d8c4f6a2b901")
def verify_primitive_tombstones(verifier: MigrationVerifier) -> None:
    _seed_legacy_rows(verifier, "after_tombstones")
    inspector = verifier.get_inspector()
    for table in TABLES:
        verifier.assert_column_exists(table, "deleted_at", nullable=True)
        column = next(
            col
            for col in inspector.get_columns(table, schema=verifier.schema)
            if col["name"] == "deleted_at"
        )
        assert isinstance(column["type"], DateTime)
        assert column["type"].timezone
        assert column["default"] is None
        workspace_column = "name" if table == "workspaces" else "workspace_name"
        rows = verifier.conn.execute(
            text(
                f'SELECT deleted_at FROM "{verifier.schema}".{table}'
                + f" WHERE {workspace_column} IN ('before_tombstones', 'after_tombstones')"
            )
        ).all()
        assert rows == [(None,), (None,)]
        # The new timestamp is writable without changing old-column reads.
        verifier.conn.execute(
            text(
                f'UPDATE "{verifier.schema}".{table} SET deleted_at = CURRENT_TIMESTAMP'
                + f" WHERE {workspace_column} = 'after_tombstones'"
            )
        )
        assert (
            verifier.conn.scalar(
                text(
                    f'SELECT deleted_at IS NOT NULL FROM "{verifier.schema}".{table}'
                    + f" WHERE {workspace_column} = 'after_tombstones'"
                )
            )
            is True
        )

    # Prior-generation values and old payloads are untouched, not backfilled.
    for table, workspace_column, generation in (
        ("workspaces", "name", 7),
        ("sessions", "workspace_name", 11),
    ):
        assert verifier.conn.scalars(
            text(
                f'SELECT deletion_generation FROM "{verifier.schema}".{table}'
                + f" WHERE {workspace_column} IN ('before_tombstones', 'after_tombstones')"
            )
        ).all() == [generation, generation]
    assert verifier.conn.scalars(
        text(
            f'SELECT content FROM "{verifier.schema}".messages'
            + " WHERE workspace_name IN ('before_tombstones', 'after_tombstones')"
        )
    ).all() == ["legacy content", "legacy content"]

    # Expand never contracts the old document index or queue discovery indexes.
    verifier.assert_indexes_exist(
        [
            ("documents", "ix_documents_deleted_at"),
            *(
                ("queue", f"ix_queue_pending_{key}")
                for key in (
                    "observers",
                    "observed",
                    "observer",
                    "scope_peer",
                    "session_name",
                )
            ),
        ]
    )
    _assert_document_deleted_at_index_preserved(verifier)
