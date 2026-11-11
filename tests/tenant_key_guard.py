"""Catch statements that look up a tenant row by id without its tenant.

The tenant tables key every row on (tenant_id, id), and ids are not unique
across tenants: every dedicated database counted messages.id from 1, and the
bulk copy into the shared pool keeps them. A statement that filters such a
table by id (or message_id) alone reads or writes another tenant's row on a
service session, and on any session it cannot seek the primary key, so it
probes every partition.

The guard watches every ORM-session statement the test suite runs from src/.
A WHERE or JOIN ON clause that compares a guarded column, where the statement
never compares the same table's tenant_id, is a violation. Every session counts: the tests drive
service-session code with plain sessions, so the session a statement runs on
here says nothing about the one it runs on in production. text() statements
are opaque to it.

A site that is safe for another reason, usually because it only ever runs on
a tenant session (where _add_tenant_predicate in src/models.py adds the
tenant), is listed in ALLOWED with the reason.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import greenlet
from sqlalchemy import Column, Delete, Select, Table, Update
from sqlalchemy.orm import ORMExecuteState
from sqlalchemy.sql import visitors
from sqlalchemy.sql.elements import BinaryExpression, ClauseList, Grouping
from sqlalchemy.sql.selectable import Join

from src.db import Base

_SRC = str(Path(__file__).resolve().parents[1] / "src") + "/"

# (path under src/, function) -> why its id-only filter is safe. A new entry
# needs every caller traced to its session: listed here, a function that later
# gains a service_db caller goes cross-tenant without the guard noticing.
_TENANT_SESSION = (
    "tenant session only (tracked_db or the request session); the hook adds"
    " tenant_id to the ORM statement"
)
ALLOWED: dict[tuple[str, str], str] = {
    ("crud/document.py", "create_documents"): _TENANT_SESSION,
    ("crud/document.py", "_apply_document_row_updates"): _TENANT_SESSION,
    ("crud/document.py", "delete_document"): _TENANT_SESSION,
    ("crud/document.py", "delete_documents"): _TENANT_SESSION,
    ("crud/document.py", "delete_document_by_id"): _TENANT_SESSION,
    ("crud/document.py", "create_observations"): _TENANT_SESSION,
    ("crud/document.py", "get_documents_by_ids"): _TENANT_SESSION,
    ("crud/document.py", "query_explicit_documents_after"): _TENANT_SESSION,
    ("crud/document.py", "fetch_documents_by_ids"): _TENANT_SESSION,
    ("crud/message.py", "get_messages_id_range"): _TENANT_SESSION,
    ("crud/message.py", "get_message_seq_in_session"): _TENANT_SESSION,
    ("crud/session.py", "clone_session"): _TENANT_SESSION,
    ("crud/webhook.py", "delete_webhook_endpoint"): _TENANT_SESSION,
    ("deriver/scope_backfill.py", "_hydrate_chunk_embeddings"): _TENANT_SESSION,
    ("deriver/scope_backfill.py", "_copy_chunk"): _TENANT_SESSION,
    ("deriver/scope_backfill.py", "_sync_copies_to_vector_store"): _TENANT_SESSION,
    ("reconciler/embed_now.py", "_claim_and_lease"): _TENANT_SESSION,
    ("reconciler/embed_now.py", "_persist_pgvector"): _TENANT_SESSION,
    ("reconciler/embed_now.py", "_mark_synced"): _TENANT_SESSION,
    ("utils/pagination.py", "_apaginate_cursor"): (
        "the routers' request session only; sqlakeyset keeps the caller's model"
        " in FROM, so the hook adds tenant_id"
    ),
    ("utils/search.py", "matching_messages"): (
        "tenant session only, and the id page comes from"
        " message_text_matches_page(scoped_tenant_id(), ...)"
    ),
    ("crud/deriver.py", "get_queue_status"): (
        "the request's read session only, joining sessions by id: the hook adds"
        " tenant_id for Session, and queue_item_tenant_match() pins the queue rows"
    ),
    ("crud/message.py", "_search_messages_pgvector"): (
        _TENANT_SESSION + "; joins messages to message_embeddings on public_id"
    ),
    ("utils/search.py", "_semantic_search_pgvector"): (
        _TENANT_SESSION + "; joins messages to message_embeddings on public_id"
    ),
    ("deriver/queue_manager.py", "get_queue_item_batch"): (
        "service session, but tenant_clause binds Message.tenant_id to the work"
        " unit's tenant under MULTI_TENANT; flag-off it is true() on a"
        " single-tenant database"
    ),
}


def _guarded_columns() -> dict[str, frozenset[str]]:
    guarded: dict[str, frozenset[str]] = {}
    for table in Base.metadata.sorted_tables:
        pk = [column.name for column in table.primary_key.columns]
        if "tenant_id" not in pk or len(pk) < 2:
            continue
        columns: set[str] = {"id"} if pk == ["tenant_id", "id"] else set()
        if "message_id" in table.columns:
            columns.add("message_id")
        if columns:
            guarded[table.name] = frozenset(columns)
    return guarded


GUARDED = _guarded_columns()


@dataclass(frozen=True)
class Violation:
    table: str
    columns: tuple[str, ...]
    path: str
    line: int
    function: str

    def __str__(self) -> str:
        return (
            f"src/{self.path}:{self.line} ({self.function}): {self.table} filtered"
            + f" by {', '.join(self.columns)} without tenant_id"
        )


def _base_table(selectable: Any) -> Table | None:
    while selectable is not None and not isinstance(selectable, Table):
        selectable = getattr(selectable, "element", None)
    return selectable


def _operand_columns(expression: Any) -> list[Column[Any]]:
    """The columns an operand compares: a column, or each column of a tuple."""
    if isinstance(expression, Column):
        return [expression]
    if isinstance(expression, Grouping):
        return _operand_columns(cast(Any, expression).element)
    if isinstance(expression, ClauseList):
        return [c for clause in expression.clauses for c in _operand_columns(clause)]
    return []


def _compared_columns(clause: Any) -> dict[str, set[str]]:
    """Columns compared anywhere in a WHERE clause, by table (subqueries included).

    Only comparison operands count: a column a subquery merely selects, as in
    exists(select(Peer.id).where(...)), filters nothing.
    """
    compared: dict[str, set[str]] = {}
    if clause is None:
        return compared
    for element in visitors.iterate(clause):
        if not isinstance(element, BinaryExpression):
            continue
        for column in _operand_columns(element.left) + _operand_columns(element.right):
            table = _base_table(column.table)
            if table is not None:
                compared.setdefault(table.name, set()).add(column.name)
    return compared


def _frames() -> list[tuple[str, int, str]]:
    """The calling frames, outermost first, including the async caller's.

    AsyncSession runs the sync session in a child greenlet, so the stack at the
    event starts at greenlet_spawn; the code that awaited the query is on the
    parent greenlet's (suspended) stack.
    """
    frames: list[tuple[str, int, str]] = []
    parent = greenlet.getcurrent().parent
    while parent is not None:
        outer: list[tuple[str, int, str]] = []
        frame = parent.gr_frame
        while frame is not None:
            code = frame.f_code
            outer.append((code.co_filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
        frames[:0] = reversed(outer)
        parent = parent.parent
    frames.extend(
        (f.filename, f.lineno or 0, f.name) for f in traceback.extract_stack()
    )
    return frames


def _call_site() -> tuple[str, int, str] | None:
    """The innermost src/ frame that ran the statement, or None for test code."""
    for filename, line, function in reversed(_frames()):
        if filename.startswith(_SRC) and not filename.endswith(
            ("src/db.py", "src/dependencies.py")
        ):
            return filename[len(_SRC) :], line, function
    return None


def _join_conditions(statement: Any) -> list[Any]:
    """The ON clause of every join in a SELECT's FROM list, nested ones included."""
    if not isinstance(statement, Select):
        return []
    conditions: list[Any] = []
    for from_ in statement.get_final_froms():
        for element in visitors.iterate(from_):
            if isinstance(element, Join) and element.onclause is not None:
                conditions.append(element.onclause)
    return conditions


def find_violations(state: ORMExecuteState) -> list[Violation]:
    statement = state.statement
    if not isinstance(statement, (Select, Update, Delete)):
        return []
    compared = _compared_columns(statement.whereclause)
    for onclause in _join_conditions(statement):
        for table, columns in _compared_columns(onclause).items():
            compared.setdefault(table, set()).update(columns)
    found: list[tuple[str, tuple[str, ...]]] = []
    for table, columns in compared.items():
        keys = GUARDED.get(table, frozenset()) & columns
        if keys and "tenant_id" not in columns:
            found.append((table, tuple(sorted(keys))))
    if not found:
        return []
    site = _call_site()
    if site is None or (site[0], site[2]) in ALLOWED:
        return []
    return [Violation(table, keys, *site) for table, keys in found]
