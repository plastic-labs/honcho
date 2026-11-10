"""add tenant id primitive

Revision ID: e5fe7f8bcf62
Revises: c4d8e2f6a1b3
Create Date: 2026-09-03

Adds ``tenant_id`` as a first-class primitive to the data model: a new
``tenants`` table and a ``tenant_id`` column (plus tenant-scoped composite
PKs / uniques / FKs / indexes) on every tenant-scoped table, matching the
declarative models MINUS physical partitioning.

No statement holds a write-blocking lock for longer than a catalog change,
and every step commits on its own, so the order is what keeps integrity.
Each new key is built and enforced before the old one it
replaces is dropped:

1. ``tenants`` and its one row; ``tenant_id`` on every table (catalog-only).
2. Every new index except the two on ``session_peers``, built concurrently. The
   primary-key indexes get a temporary name, ``pk_<table>_tenant``.
3. The new uniques attached to their indexes; the new FKs added ``NOT VALID``
   (they check new writes and cascade at once), then validated.
4. The old FKs, then the old uniques, dropped.
5. The two ``session_peers`` indexes built, then each primary key swapped onto
   its new index in one statement.

The swaps are last, and ``_finished`` reads them: a run that stopped part-way runs
again from the top, and each step skips what is already done.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext

import sqlalchemy as sa
from alembic import op

from migrations.utils import (
    _execute_short_lock,  # pyright: ignore[reportPrivateUsage]
    build_timeouts,
    caller_owns_transaction,
    column_exists,
    create_index_online,
    drop_index_online,
    get_schema,
    index_exists,
    table_exists,
)
from src.models import instance_tenant_id

# revision identifiers, used by Alembic.
revision: str = "e5fe7f8bcf62"
down_revision: str | None = "c4d8e2f6a1b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()

DEFAULT_TENANT_ID = instance_tenant_id()

LEGACY_SESSION_PEERS_INDEX = "uq_session_peers_legacy_ws_session_peer"

# The new primary-key index of a table, until the swap renames it to pk_<table>.
PK_CANDIDATE = "pk_{}_tenant"

# Tables that gain tenant_id + a composite PK + composite FKs (session_peers has
# an all-natural composite PK and is handled with the rest).
TENANT_SCOPED: tuple[str, ...] = (
    "workspaces",
    "peers",
    "sessions",
    "messages",
    "message_embeddings",
    "collections",
    "documents",
    "document_sources",
    "conclusion_source_messages",
    "webhook_endpoints",
    "session_peers",
)

# New primary keys (table -> ordered PK columns). Everything is (tenant_id, id)
# except the association table, whose PK is all-natural.
NEW_PKS: dict[str, list[str]] = {
    "workspaces": ["tenant_id", "id"],
    "peers": ["tenant_id", "id"],
    "sessions": ["tenant_id", "id"],
    "messages": ["tenant_id", "id"],
    "message_embeddings": ["tenant_id", "id"],
    "collections": ["tenant_id", "id"],
    "documents": ["tenant_id", "id"],
    "document_sources": ["tenant_id", "derived_id", "source_id"],
    "conclusion_source_messages": ["tenant_id", "conclusion_id", "message_id"],
    "webhook_endpoints": ["tenant_id", "id"],
    "session_peers": ["tenant_id", "workspace_name", "session_name", "peer_name"],
}

# New unique constraints (table -> list of (name, cols)).
NEW_UNIQUES: dict[str, list[tuple[str, list[str]]]] = {
    "workspaces": [("uq_workspaces_tenant_id_name", ["tenant_id", "name"])],
    "peers": [
        (
            "uq_peers_tenant_id_name_workspace_name",
            ["tenant_id", "name", "workspace_name"],
        )
    ],
    "sessions": [
        (
            "uq_sessions_tenant_id_name_workspace_name",
            ["tenant_id", "name", "workspace_name"],
        )
    ],
    "messages": [
        ("uq_messages_tenant_id_public_id", ["tenant_id", "public_id"]),
        (
            "uq_messages_tenant_id_ws_session_seq",
            ["tenant_id", "workspace_name", "session_name", "seq_in_session"],
        ),
    ],
    "collections": [
        (
            "uq_collections_tenant_id_observer_observed_ws",
            ["tenant_id", "observer", "observed", "workspace_name"],
        )
    ],
}

# New foreign keys: (name, source_table, [local_cols], ref_table, [ref_cols], ondelete)
# Every tenant-scoped table also FKs its tenant_id -> tenants.tenant_id (added in a
# loop below); the composite FKs below carry the tenant_id into the natural-key refs.
NEW_FKS: list[tuple[str, str, list[str], str, list[str], str | None]] = [
    # peers
    (
        "fk_peers_ws_tenant_workspaces",
        "peers",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    # sessions
    (
        "fk_sessions_ws_tenant_workspaces",
        "sessions",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    # messages
    (
        "fk_messages_session_ws_tenant_sessions",
        "messages",
        ["session_name", "workspace_name", "tenant_id"],
        "sessions",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_messages_peer_ws_tenant_peers",
        "messages",
        ["peer_name", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    # message_embeddings
    (
        "fk_msg_emb_tenant_message_messages",
        "message_embeddings",
        ["tenant_id", "message_id"],
        "messages",
        ["tenant_id", "public_id"],
        "CASCADE",
    ),
    (
        "fk_msg_emb_ws_tenant_workspaces",
        "message_embeddings",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    (
        "fk_msg_emb_session_ws_tenant_sessions",
        "message_embeddings",
        ["session_name", "workspace_name", "tenant_id"],
        "sessions",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_msg_emb_peer_ws_tenant_peers",
        "message_embeddings",
        ["peer_name", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    # collections
    (
        "fk_collections_ws_tenant_workspaces",
        "collections",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    (
        "fk_collections_observer_ws_tenant_peers",
        "collections",
        ["observer", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_collections_observed_ws_tenant_peers",
        "collections",
        ["observed", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    # documents
    (
        "fk_documents_ws_tenant_workspaces",
        "documents",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    (
        "fk_documents_collection_tenant_collections",
        "documents",
        ["observer", "observed", "workspace_name", "tenant_id"],
        "collections",
        ["observer", "observed", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_documents_observer_ws_tenant_peers",
        "documents",
        ["observer", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_documents_observed_ws_tenant_peers",
        "documents",
        ["observed", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_documents_session_ws_tenant_sessions",
        "documents",
        ["session_name", "workspace_name", "tenant_id"],
        "sessions",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    # document_sources (created by a7c3e9f1b2d4 keyed on documents.id alone; the
    # composite PK above means the edge must now carry tenant_id to reach its
    # document)
    (
        "fk_document_sources_derived_tenant_documents",
        "document_sources",
        ["derived_id", "tenant_id"],
        "documents",
        ["id", "tenant_id"],
        "CASCADE",
    ),
    (
        "fk_document_sources_ws_tenant_workspaces",
        "document_sources",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    # conclusion_source_messages (created by c4d8e2f6a1b3 keyed on documents.id
    # and messages.public_id alone; same reason as document_sources)
    (
        "fk_conclusion_source_messages_conclusion_tenant_documents",
        "conclusion_source_messages",
        ["conclusion_id", "tenant_id"],
        "documents",
        ["id", "tenant_id"],
        "CASCADE",
    ),
    (
        "fk_conclusion_source_messages_message_tenant_messages",
        "conclusion_source_messages",
        ["message_id", "tenant_id"],
        "messages",
        ["public_id", "tenant_id"],
        "CASCADE",
    ),
    (
        "fk_conclusion_source_messages_ws_tenant_workspaces",
        "conclusion_source_messages",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    # webhook_endpoints
    (
        "fk_webhook_ws_tenant_workspaces",
        "webhook_endpoints",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    # session_peers
    (
        "fk_session_peers_ws_tenant_workspaces",
        "session_peers",
        ["workspace_name", "tenant_id"],
        "workspaces",
        ["name", "tenant_id"],
        None,
    ),
    (
        "fk_session_peers_session_ws_tenant_sessions",
        "session_peers",
        ["session_name", "workspace_name", "tenant_id"],
        "sessions",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
    (
        "fk_session_peers_peer_ws_tenant_peers",
        "session_peers",
        ["peer_name", "workspace_name", "tenant_id"],
        "peers",
        ["name", "workspace_name", "tenant_id"],
        None,
    ),
]

# region ai
# downgrade() cannot recover OLD_UNIQUES / OLD_FKS by introspection: by the
# time it runs, upgrade()'s _drop_all_fks / _drop_all_uniques have already
# removed those constraints from the catalog under their original names, so
# there is nothing left on the DB to read the old definitions back from. The
# two lists below are instead the literal pre-migration constraint set,
# captured by migrating a fresh database to a7c3e9f1b2d4 and dumping
# pg_constraint for every TENANT_SCOPED table plus queue. The revisions between
# a7c3e9f1b2d4 and this revision's down_revision (a5f8da8f6d23) add only
# indexes and plain columns, no constraints, so the lists still hold. Keep them
# in lockstep with that constraint set, not with NEW_FKS /
# NEW_UNIQUES above (those are the *new* tenant-aware shape).
# endregion

# Pre-migration unique constraints (table -> list of (name, cols)), exactly as
# they existed on down_revision.
OLD_UNIQUES: dict[str, list[tuple[str, list[str]]]] = {
    "workspaces": [("uq_workspaces_name", ["name"])],
    "peers": [("uq_peers_name_workspace_name", ["name", "workspace_name"])],
    "sessions": [("uq_sessions_name_workspace_name", ["name", "workspace_name"])],
    "messages": [
        ("uq_messages_public_id", ["public_id"]),
        (
            "uq_messages_workspace_name_session_name_seq_in_session",
            ["workspace_name", "session_name", "seq_in_session"],
        ),
    ],
    "collections": [
        (
            "uq_collections_observer_observed_workspace_name",
            ["observer", "observed", "workspace_name"],
        )
    ],
}

# Pre-migration foreign keys: (name, source_table, [local_cols], ref_table,
# [ref_cols], ondelete), exactly as they existed on down_revision. Includes
# queue's FKs: queue is not in TENANT_SCOPED, but upgrade() step 3 drops its
# FKs too (via _drop_all_fks((*TENANT_SCOPED, "queue"))), so downgrade must
# restore them here.
OLD_FKS: list[tuple[str, str, list[str], str, list[str], str | None]] = [
    # peers
    (
        "fk_peers_workspace_name_workspaces",
        "peers",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    # sessions
    (
        "fk_sessions_workspace_name_workspaces",
        "sessions",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    # messages
    (
        "fk_messages_session_name_sessions",
        "messages",
        ["session_name", "workspace_name"],
        "sessions",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_messages_peer_name_peers",
        "messages",
        ["peer_name", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    # message_embeddings
    (
        "fk_message_embeddings_message_id_messages",
        "message_embeddings",
        ["message_id"],
        "messages",
        ["public_id"],
        "CASCADE",
    ),
    (
        "fk_message_embeddings_workspace_name_workspaces",
        "message_embeddings",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    (
        "fk_message_embeddings_session_name_workspace_name_sessions",
        "message_embeddings",
        ["session_name", "workspace_name"],
        "sessions",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_message_embeddings_peer_name_workspace_name_peers",
        "message_embeddings",
        ["peer_name", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    # collections
    (
        "fk_collections_workspace_name_workspaces",
        "collections",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    (
        "fk_collections_observer_workspace_name_peers",
        "collections",
        ["observer", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_collections_observed_workspace_name_peers",
        "collections",
        ["observed", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    # documents
    (
        "fk_documents_workspace_name_workspaces",
        "documents",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    (
        "fk_documents_observer_observed_workspace_name_collections",
        "documents",
        ["observer", "observed", "workspace_name"],
        "collections",
        ["observer", "observed", "workspace_name"],
        None,
    ),
    (
        "fk_documents_observer_workspace_name_peers",
        "documents",
        ["observer", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_documents_observed_workspace_name_peers",
        "documents",
        ["observed", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_documents_session_workspace",
        "documents",
        ["session_name", "workspace_name"],
        "sessions",
        ["name", "workspace_name"],
        None,
    ),
    # document_sources
    (
        "fk_document_sources_derived_id_documents",
        "document_sources",
        ["derived_id"],
        "documents",
        ["id"],
        "CASCADE",
    ),
    (
        "fk_document_sources_workspace_name_workspaces",
        "document_sources",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    # conclusion_source_messages
    (
        "fk_conclusion_source_messages_conclusion_id_documents",
        "conclusion_source_messages",
        ["conclusion_id"],
        "documents",
        ["id"],
        "CASCADE",
    ),
    (
        "fk_conclusion_source_messages_message_id_messages",
        "conclusion_source_messages",
        ["message_id"],
        "messages",
        ["public_id"],
        "CASCADE",
    ),
    (
        "fk_conclusion_source_messages_workspace_name_workspaces",
        "conclusion_source_messages",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    # webhook_endpoints
    (
        "fk_webhook_endpoints_workspace_name_workspaces",
        "webhook_endpoints",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    # session_peers
    (
        "fk_session_peers_workspace_name",
        "session_peers",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
    (
        "fk_session_peers_session_name_workspace_name_sessions",
        "session_peers",
        ["session_name", "workspace_name"],
        "sessions",
        ["name", "workspace_name"],
        None,
    ),
    (
        "fk_session_peers_peer_name_workspace_name_peers",
        "session_peers",
        ["peer_name", "workspace_name"],
        "peers",
        ["name", "workspace_name"],
        None,
    ),
    # queue (not TENANT_SCOPED, but upgrade() drops its FKs alongside the rest)
    ("fk_queue_message_id", "queue", ["message_id"], "messages", ["id"], None),
    ("fk_queue_session_id", "queue", ["session_id"], "sessions", ["id"], None),
    (
        "fk_queue_workspace_name",
        "queue",
        ["workspace_name"],
        "workspaces",
        ["name"],
        None,
    ),
]


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _drop_pk(table: str) -> None:
    insp = _inspector()
    pk = insp.get_pk_constraint(table, schema=schema)
    name = pk.get("name") if pk else None
    if name:
        op.drop_constraint(name, table, type_="primary", schema=schema)


# Secondary indexes: (name, table, columns, using, include, with_). One that
# already exists under its name is kept as it is.
NEW_INDEXES: list[tuple[str, str, str, str | None, str | None, str | None]] = [
    (
        "ix_peers_tenant_workspace",
        "peers",
        "tenant_id, workspace_name",
        None,
        None,
        None,
    ),
    (
        "ix_sessions_tenant_workspace",
        "sessions",
        "tenant_id, workspace_name",
        None,
        None,
        None,
    ),
    (
        "ix_messages_session_lookup",
        "messages",
        "tenant_id, session_name, id",
        None,
        "created_at",
        None,
    ),
    (
        "ix_messages_peer_lookup",
        "messages",
        "tenant_id, workspace_name, peer_name, created_at",
        None,
        None,
        None,
    ),
    (
        "ix_messages_content_gin",
        "messages",
        "to_tsvector('english', content)",
        "gin",
        None,
        None,
    ),
    (
        "ix_message_embeddings_message_tenant",
        "message_embeddings",
        "message_id, tenant_id",
        None,
        None,
        None,
    ),
    (
        "ix_message_embeddings_embedding_hnsw",
        "message_embeddings",
        "embedding vector_cosine_ops",
        "hnsw",
        None,
        "m = 16, ef_construction = 64",
    ),
    (
        "ix_message_embeddings_sync_state_last_sync_at",
        "message_embeddings",
        "sync_state, last_sync_at",
        None,
        None,
        None,
    ),
    (
        "ix_documents_tenant_collection",
        "documents",
        "tenant_id, observer, observed, workspace_name",
        None,
        None,
        None,
    ),
    (
        "ix_documents_embedding_hnsw",
        "documents",
        "embedding vector_cosine_ops",
        "hnsw",
        None,
        "m = 16, ef_construction = 64",
    ),
    ("ix_documents_source_ids_gin", "documents", "source_ids", "gin", None, None),
    (
        "ix_documents_sync_state_last_sync_at",
        "documents",
        "sync_state, last_sync_at",
        None,
        None,
        None,
    ),
    (
        "ix_webhook_endpoints_tenant_workspace",
        "webhook_endpoints",
        "tenant_id, workspace_name",
        None,
        None,
        None,
    ),
]

# Every FK of the new shape: tenant_id -> tenants on each table, then NEW_FKS.
ALL_NEW_FKS: list[tuple[str, str, list[str], str, list[str], str | None]] = [
    *(
        (f"fk_{t}_tenant_id_tenants", t, ["tenant_id"], "tenants", ["tenant_id"], None)
        for t in TENANT_SCOPED
    ),
    *NEW_FKS,
]


def _q(name: str) -> str:
    return f'"{schema}"."{name}"'


def _cols(cols: Sequence[str]) -> str:
    return ", ".join(f'"{c}"' for c in cols)


def _pk(table: str) -> tuple[str | None, list[str]]:
    """The table's primary key: (constraint name, columns in key order)."""
    pk = _inspector().get_pk_constraint(table, schema=schema)
    return pk.get("name"), list(pk.get("constrained_columns") or [])


def _fk_names(table: str) -> set[str]:
    return {
        str(fk["name"])
        for fk in _inspector().get_foreign_keys(table, schema=schema)
        if fk.get("name")
    }


@contextmanager
def _autocommit() -> Generator[None]:
    """Each statement in the block commits on its own, so it holds its lock only
    for its own run. A caller-owned transaction stays one plain transaction."""
    if caller_owns_transaction():
        yield
        return
    with op.get_context().autocommit_block():
        yield


def _short_lock_ms() -> int:
    """Half the server's deadlock_timeout: at least 1 ms, at most one second."""
    return int(
        op.get_bind()
        .execute(
            sa.text(
                "SELECT greatest(least(setting::int / 2, 1000), 1) FROM pg_settings"
                + " WHERE name = 'deadlock_timeout'"
            )
        )
        .scalar_one()
    )


def _alter(table: str, action: str) -> None:
    """One catalog-only ALTER TABLE under a short lock_timeout, retried in place."""
    sql = f"ALTER TABLE {_q(table)} {action}"
    if caller_owns_transaction():
        # ai: a timeout aborts the caller's transaction; a retry would hide it.
        op.execute(sql)
        return
    # region ai
    # An ALTER that touches two tables (a foreign key locks the child, then the
    # parent) can cross a write that takes them in the other order. Postgres
    # breaks that tie after deadlock_timeout by cancelling one side, which can be
    # the application's write. Waiting for less than that makes this statement
    # give up first in the common case, so fewer writes are cancelled; it does
    # not rule a deadlock out. The statement is then retried. SET LOCAL lasts for
    # this one statement's transaction only (the two are sent together), so
    # nothing is left on the connection.
    # endregion
    _execute_short_lock(f"SET LOCAL lock_timeout = '{_short_lock_ms()}ms'; {sql}")


def _finished() -> bool:
    """True when the last step, the primary-key swap, is done on every table.

    Also true on a database that was created with the tenant shape (the shared
    pool's bootstrap) or that ran this revision's earlier, one-transaction body.
    """
    return table_exists("tenants") and all(
        _pk(table)[1] == cols for table, cols in NEW_PKS.items()
    )


def _create_tenants() -> None:
    if not table_exists("tenants"):
        op.create_table(
            "tenants",
            sa.Column("tenant_id", sa.TEXT(), nullable=False),
            sa.Column("vector_correlation_id", sa.TEXT(), nullable=True),
            sa.Column("tier", sa.TEXT(), server_default="dedicated", nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("tenant_id", name="pk_tenants"),
            schema=schema,
        )
    # ai: the key names the tenant's vector-store namespaces, so two tenants
    # holding one would share them; NULL (not yet set) is exempt.
    if not index_exists("tenants", "uq_tenants_vector_correlation_id"):
        op.create_index(
            "uq_tenants_vector_correlation_id",
            "tenants",
            ["vector_correlation_id"],
            unique=True,
            schema=schema,
            postgresql_where=sa.text("vector_correlation_id IS NOT NULL"),
        )
    op.execute(
        sa.text(
            f'INSERT INTO "{schema}"."tenants" (tenant_id, tier) '
            + "VALUES (:tid, 'dedicated') ON CONFLICT DO NOTHING"
        ).bindparams(tid=DEFAULT_TENANT_ID)
    )


def _build_pk_candidate(table: str) -> None:
    # ai: after the swap the index is named pk_<table>; don't build a second.
    if _pk(table)[1] != NEW_PKS[table]:
        create_index_online(
            PK_CANDIDATE.format(table), table, _cols(NEW_PKS[table]), unique=True
        )


def _build_indexes() -> None:
    """Build the new indexes concurrently, except the two on session_peers.

    A valid one is kept on a rerun.
    """
    for table in NEW_PKS:
        if table != "session_peers":
            _build_pk_candidate(table)
    for table, uniques in NEW_UNIQUES.items():
        for name, cols in uniques:
            create_index_online(name, table, _cols(cols), unique=True)
    create_index_online("ix_queue_tenant_id", "queue", "tenant_id")
    for name, table, columns, using, include, with_ in NEW_INDEXES:
        create_index_online(
            name, table, columns, using=using, include=include, with_=with_
        )


def _add_new_constraints() -> None:
    """Attach the new uniques to their indexes and add the new FKs NOT VALID."""
    for table, uniques in NEW_UNIQUES.items():
        existing = {
            uq["name"]
            for uq in _inspector().get_unique_constraints(table, schema=schema)
        }
        for name, _ in uniques:
            if name not in existing:
                _alter(table, f'ADD CONSTRAINT "{name}" UNIQUE USING INDEX "{name}"')
    # region ai
    # NOT VALID skips the scan of existing rows but checks every new write and
    # installs the ON DELETE action, so from here each new FK enforces next to
    # the old one it replaces. An FK needs only a unique index on the referenced
    # columns, not a constraint: the two that reference documents (id, tenant_id)
    # bind to pk_documents_tenant, which the swap later turns into the primary key.
    # endregion
    for name, src, local_cols, ref, ref_cols, ondelete in ALL_NEW_FKS:
        if name not in _fk_names(src):
            _alter(
                src,
                f'ADD CONSTRAINT "{name}" FOREIGN KEY ({_cols(local_cols)})'
                + f" REFERENCES {_q(ref)} ({_cols(ref_cols)})"
                + (f" ON DELETE {ondelete}" if ondelete else "")
                + " NOT VALID",
            )


def _validate_new_fks() -> None:
    # region ai
    # The scan runs under SHARE UPDATE EXCLUSIVE, which reads and writes pass.
    # build_timeouts: a wait for that lock (behind a vacuum) stalls nobody, and a
    # large table's scan must not hit the runner's statement_timeout. Validating
    # a validated constraint is a catalog no-op, so a rerun costs nothing.
    # endregion
    with nullcontext() if caller_owns_transaction() else build_timeouts():
        for name, src, *_ in ALL_NEW_FKS:
            op.execute(f'ALTER TABLE {_q(src)} VALIDATE CONSTRAINT "{name}"')


def _drop_old_constraints() -> None:
    """Drop every FK and unique constraint that is not part of the new shape.

    Old names vary between databases, so this drops by exclusion. queue keeps no
    FK: its old ones point at the keys that are about to change.
    """
    new_fks = {name for name, *_ in ALL_NEW_FKS}
    for table in (*TENANT_SCOPED, "queue"):
        for name in sorted(_fk_names(table) - new_fks):
            _alter(table, f'DROP CONSTRAINT IF EXISTS "{name}"')
    for table in TENANT_SCOPED:
        keep = {name for name, _ in NEW_UNIQUES.get(table, [])}
        for uq in _inspector().get_unique_constraints(table, schema=schema):
            if uq["name"] not in keep:
                _alter(table, f'DROP CONSTRAINT IF EXISTS "{uq["name"]}"')


def _stale_unique_indexes(table: str) -> list[str]:
    """Unique indexes with no constraint that are not part of the new shape."""
    keep = {
        LEGACY_SESSION_PEERS_INDEX,
        PK_CANDIDATE.format(table),
        *(name for name, _ in NEW_UNIQUES.get(table, [])),
    }
    return [
        str(idx["name"])
        for idx in _inspector().get_indexes(table, schema=schema)
        if idx.get("unique")
        and "duplicates_constraint" not in idx
        and idx["name"] not in keep
    ]


def _build_session_peers_indexes() -> None:
    # region ai
    # The previous release upserts session_peers with ON CONFLICT on the old
    # (workspace, session, peer) key, which is the primary key the swap drops.
    # Keep a unique index on it so that image can still add peers to sessions
    # during a rolling deploy and stays a rollback target. Only a DB that runs
    # this body gets it, and those hold a single tenant, so it rejects nothing the
    # new key allows. Not declared in models (see migrations/env.py); a later
    # revision drops it once the compatibility floor passes this release.
    #
    # Both indexes are built as late as possible. Postgres handles that upsert's
    # conflict only on a valid index with exactly its three columns. The legacy
    # index while it builds, and the new four-column key index always, reject the
    # same rows as a plain unique check: two requests that add the same new
    # membership at once can then get a unique violation instead of an update.
    # That holds in the final schema until the old pods are gone; building here
    # keeps it out of every other build and the validation.
    # endregion
    create_index_online(
        LEGACY_SESSION_PEERS_INDEX,
        "session_peers",
        "workspace_name, session_name, peer_name",
        unique=True,
    )
    _build_pk_candidate("session_peers")


def _swap_primary_keys() -> None:
    # region ai
    # One statement per table: the old key is dropped and the new one attached
    # under one short ACCESS EXCLUSIVE lock, so the table always has a primary
    # key. USING INDEX renames pk_<table>_tenant to pk_<table>. Every key column
    # is already NOT NULL, so there is no scan.
    # endregion
    for table in NEW_PKS:
        name, current = _pk(table)
        if current == NEW_PKS[table]:
            continue
        drop = f'DROP CONSTRAINT "{name}", ' if name else ""
        _alter(
            table,
            f'{drop}ADD CONSTRAINT "pk_{table}" PRIMARY KEY'
            + f' USING INDEX "{PK_CANDIDATE.format(table)}"',
        )


def upgrade() -> None:
    # region ai
    # Skip a database that already has the whole tenant shape: the shared pool's
    # schema is built by the bootstrap and stamped past this revision, and a
    # dedicated database may run this again after a later revision failed before
    # the version row was committed. A run that stopped part-way is not skipped.
    # endregion
    if _finished():
        return

    # 1. tenants + the single self-host tenant, then tenant_id on every table. A
    #    constant default fills existing rows without rewriting the table, and
    #    lets the previous release, which never sets tenant_id, keep inserting.
    #    Service tables: tenant_id is plain attribution, nullable, no FK, no RLS.
    _create_tenants()
    with _autocommit():
        for table in TENANT_SCOPED:
            _alter(
                table,
                "ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL"
                + f" DEFAULT '{DEFAULT_TENANT_ID}'",
            )
        for table in ("queue", "active_queue_sessions"):
            _alter(table, "ADD COLUMN IF NOT EXISTS tenant_id TEXT")

    # 2. Every new index but the two on session_peers (step 5), concurrently.
    _build_indexes()

    # 3. New uniques and FKs: enforced first, validated second.
    with _autocommit():
        _add_new_constraints()
        _validate_new_fks()

    # 4. Only now the old FKs and uniques: everything they enforced, a validated
    #    new constraint enforces too.
    with _autocommit():
        _drop_old_constraints()
    for table in TENANT_SCOPED:
        for name in _stale_unique_indexes(table):
            drop_index_online(name, table)

    # 5. The two session_peers indexes, then the primary keys. The swaps are the
    #    last step, and _finished reads them.
    # region ai
    # The session_peers builds come before every swap, not after. A concurrent
    # build can wait for transactions that hold older snapshots. Once messages has
    # its new key, the previous release's queue claim reads messages by id alone,
    # with no index that leads with id: with a backlog that query runs for
    # minutes, and a build that starts behind it waits as long.
    # endregion
    _build_session_peers_indexes()
    with _autocommit():
        _swap_primary_keys()


def downgrade() -> None:
    # Reverse the transform: drop new FKs/uniques/indexes, restore sole-id PKs,
    # drop tenant_id, drop tenants. Dropping the tenant_id column cascades away any
    # residual constraint that references it.
    if not table_exists("tenants"):
        return

    for name, src, *_ in NEW_FKS:
        insp = _inspector()
        if any(
            fk.get("name") == name for fk in insp.get_foreign_keys(src, schema=schema)
        ):
            op.drop_constraint(name, src, type_="foreignkey", schema=schema)
    for tname in TENANT_SCOPED:
        insp = _inspector()
        fk_name = f"fk_{tname}_tenant_id_tenants"
        if any(
            fk.get("name") == fk_name
            for fk in insp.get_foreign_keys(tname, schema=schema)
        ):
            op.drop_constraint(fk_name, tname, type_="foreignkey", schema=schema)

    for tname, uniques in NEW_UNIQUES.items():
        for uname, _cols in uniques:
            insp = _inspector()
            if any(
                uq.get("name") == uname
                for uq in insp.get_unique_constraints(tname, schema=schema)
            ):
                op.drop_constraint(uname, tname, type_="unique", schema=schema)

    # Restore sole-id / natural PKs, then drop tenant_id (with any index on it).
    old_pks = {
        "session_peers": ["workspace_name", "session_name", "peer_name"],
        "document_sources": ["derived_id", "source_id"],
        "conclusion_source_messages": ["conclusion_id", "message_id"],
    }
    if index_exists("session_peers", LEGACY_SESSION_PEERS_INDEX):
        op.drop_index(LEGACY_SESSION_PEERS_INDEX, "session_peers", schema=schema)
    for tname in TENANT_SCOPED:
        _drop_pk(tname)
        cols = old_pks.get(tname, ["id"])
        op.create_primary_key(f"pk_{tname}", tname, cols, schema=schema)

    # Restore the pre-migration uniques and FKs (see the OLD_UNIQUES / OLD_FKS
    # comment above for why these are a recorded list rather than introspected
    # here). Uniques first: several of the FKs below target one of them rather
    # than a primary key (e.g. fk_queue_workspace_name -> workspaces.name).
    for tname, uniques in OLD_UNIQUES.items():
        insp = _inspector()
        existing_uniques = {
            uq.get("name") for uq in insp.get_unique_constraints(tname, schema=schema)
        }
        for uname, cols in uniques:
            if uname not in existing_uniques:
                op.create_unique_constraint(uname, tname, cols, schema=schema)

    for name, src, local_cols, ref, ref_cols, ondelete in OLD_FKS:
        insp = _inspector()
        if not any(
            fk.get("name") == name for fk in insp.get_foreign_keys(src, schema=schema)
        ):
            op.create_foreign_key(
                name,
                src,
                ref,
                local_cols,
                ref_cols,
                ondelete=ondelete,
                source_schema=schema,
                referent_schema=schema,
            )

    for tname in (*TENANT_SCOPED, "queue", "active_queue_sessions"):
        if column_exists(tname, "tenant_id"):
            op.drop_column(tname, "tenant_id", schema=schema)

    op.drop_table("tenants", schema=schema)
