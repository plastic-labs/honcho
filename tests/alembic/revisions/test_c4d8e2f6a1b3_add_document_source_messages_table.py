"""Hooks for revision c4d8e2f6a1b3 (document_source_messages table, DDL only)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE_NAME = "cited_ws"
PEER_NAME = "cited_peer"
SESSION_NAME = "cited_session"

DOC_ID = generate_nanoid()
MESSAGE_ID = generate_nanoid()


@register_before_upgrade("c4d8e2f6a1b3")
def prepare_document_source_messages(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("document_source_messages", exists=False)

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
            f"""INSERT INTO "{schema}"."sessions" ("id", "name", "workspace_name")
            VALUES (:id, :n, :w)"""
        ),
        {"id": generate_nanoid(), "n": SESSION_NAME, "w": WORKSPACE_NAME},
    )
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."messages"
                ("public_id", "session_name", "workspace_name", "peer_name",
                 "content", "seq_in_session")
            VALUES (:id, :s, :w, :p, 'hello', 1)"""
        ),
        {"id": MESSAGE_ID, "s": SESSION_NAME, "w": WORKSPACE_NAME, "p": PEER_NAME},
    )
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."collections"
                ("id", "workspace_name", "observer", "observed")
            VALUES (:id, :w, :p, :p)"""
        ),
        {"id": generate_nanoid(), "w": WORKSPACE_NAME, "p": PEER_NAME},
    )
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."documents"
                ("id", "workspace_name", "observer", "observed", "content",
                 "level", "session_name", "internal_metadata")
            VALUES (:id, :w, :p, :p, 'peer said hello', 'explicit', :s,
                    CAST(:m AS jsonb))"""
        ),
        {
            "id": DOC_ID,
            "w": WORKSPACE_NAME,
            "p": PEER_NAME,
            "s": SESSION_NAME,
            "m": '{"message_ids": [1]}',
        },
    )


@register_after_upgrade("c4d8e2f6a1b3")
def verify_document_source_messages(verifier: MigrationVerifier) -> None:
    verifier.assert_table_exists("document_source_messages")
    verifier.assert_indexes_exist(
        [("document_source_messages", "ix_document_source_messages_message_id")]
    )

    schema = verifier.schema
    connection = verifier.conn

    # No backfill: pre-existing explicit rows have no citations to recover.
    edge_count = connection.execute(
        text(f'SELECT count(*) FROM "{schema}"."document_source_messages"')
    ).scalar_one()
    assert edge_count == 0
    metadata = connection.execute(
        text(
            f'SELECT "internal_metadata" FROM "{schema}"."documents" WHERE "id" = :id'
        ),
        {"id": DOC_ID},
    ).scalar_one()
    assert metadata == {"message_ids": [1]}

    # Edges follow the cited message: deleting it removes the edge, not the document.
    connection.execute(
        text(
            f"""INSERT INTO "{schema}"."document_source_messages"
                ("derived_id", "message_id", "position", "workspace_name")
            VALUES (:d, :m, 0, :w)"""
        ),
        {"d": DOC_ID, "m": MESSAGE_ID, "w": WORKSPACE_NAME},
    )
    connection.execute(
        text(f'DELETE FROM "{schema}"."messages" WHERE "public_id" = :m'),
        {"m": MESSAGE_ID},
    )
    edge_count = connection.execute(
        text(f'SELECT count(*) FROM "{schema}"."document_source_messages"')
    ).scalar_one()
    assert edge_count == 0
    doc_count = connection.execute(
        text(f'SELECT count(*) FROM "{schema}"."documents" WHERE "id" = :id'),
        {"id": DOC_ID},
    ).scalar_one()
    assert doc_count == 1
