"""Hooks for revision a7c3e9f1b2d4 (document_sources table, DDL only)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = "sources_ws"
PEER_NAME = "sources_peer"

# Documents covering every legacy linkage location; the reconciler drains
# these after deploy, so the migration must leave them untouched.
DOC_COLUMN = generate_nanoid()  # linkage in source_ids column
DOC_META = generate_nanoid()  # linkage in internal_metadata.source_ids
DOC_PREMISE = generate_nanoid()  # linkage in internal_metadata.premise_ids
DOC_GARBAGE = generate_nanoid()  # column mixes valid + malformed entries

SRC_A = generate_nanoid()
SRC_B = generate_nanoid()
SRC_C = generate_nanoid()
SRC_D = generate_nanoid()
SRC_E = generate_nanoid()


@register_before_upgrade("a7c3e9f1b2d4")
def prepare_document_sources(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("document_sources", exists=False)

    schema = verifier.schema
    connection = verifier.conn

    connection.execute(
        text(f'INSERT INTO "{schema}"."workspaces" ("id", "name") VALUES (:id, :n)'),
        {"id": generate_nanoid(), "n": WORKSPACE_NAME},
    )
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."peers" ("id", "name", "workspace_name")
            VALUES (:id, :n, :w)"""
        ),
        {"id": generate_nanoid(), "n": PEER_NAME, "w": WORKSPACE_NAME},
    )
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."collections"
                ("id", "workspace_name", "observer", "observed")
            VALUES (:id, :w, :p, :p)"""
        ),
        {"id": generate_nanoid(), "w": WORKSPACE_NAME, "p": PEER_NAME},
    )

    def seed_doc(doc_id: str, source_ids: str | None, metadata: str) -> None:
        connection.execute(
            text(
                f"""INSERT INTO "{schema}"."documents"
                    ("id", "workspace_name", "observer", "observed", "content",
                     "level", "source_ids", "internal_metadata")
                VALUES (:id, :w, :p, :p, :c, 'deductive',
                        CAST(:s AS jsonb), CAST(:m AS jsonb))"""
            ),
            {
                "id": doc_id,
                "w": WORKSPACE_NAME,
                "p": PEER_NAME,
                "c": f"doc {doc_id}",
                "s": source_ids,
                "m": metadata,
            },
        )

    seed_doc(DOC_COLUMN, f'["{SRC_A}", "{SRC_B}"]', "{}")
    seed_doc(DOC_META, None, f'{{"source_ids": ["{SRC_C}"]}}')
    seed_doc(DOC_PREMISE, None, f'{{"premise_ids": ["{SRC_D}"]}}')
    seed_doc(DOC_GARBAGE, f'["{SRC_E}", "1234", "2024-01-01T00:00:00"]', "{}")


@register_after_upgrade("a7c3e9f1b2d4")
def verify_document_sources(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("document_sources")
    verifier.assert_indexes_exist(
        [
            ("document_sources", "ix_document_sources_source_id"),
            # Kept until the drain completes; the follow-up migration drops it.
            ("documents", "ix_documents_source_ids_gin"),
        ]
    )

    schema = verifier.schema
    connection = verifier.conn

    edge_count = connection.execute(
        text(f'SELECT count(*) FROM "{schema}"."document_sources"')
    ).scalar_one()
    assert edge_count == 0  # backfill is the reconciler's job, not the migration's

    legacy = connection.execute(
        text(
            f"""SELECT "id", "source_ids", "internal_metadata"
            FROM "{schema}"."documents" ORDER BY "id"
            """
        )
    ).all()
    by_id = {row.id: row for row in legacy}
    assert by_id[DOC_COLUMN].source_ids == [SRC_A, SRC_B]
    assert by_id[DOC_META].internal_metadata == {"source_ids": [SRC_C]}
    assert by_id[DOC_PREMISE].internal_metadata == {"premise_ids": [SRC_D]}
    assert by_id[DOC_GARBAGE].source_ids == [SRC_E, "1234", "2024-01-01T00:00:00"]
