"""Writes the matrix performs, each recorded in the ledger as it happens."""

import json
import secrets
from dataclasses import dataclass, field
from typing import Any

from tests.compat_matrix.api import HonchoAPI
from tests.compat_matrix.ledger import (
    ConclusionExpect,
    DeductionExpect,
    DerivedExpect,
    Expectation,
    Ledger,
    MessageExpect,
    PeerExpect,
    SessionExpect,
    SummaryExpect,
    WorkspaceExpect,
)
from tests.compat_matrix.stack import Stack

# Delete, suppress, restore and queued work are the operations rule 7 names.
# Suppress and restore do not exist in the API yet; the deletion lifecycle
# work adds them, and their scenarios belong in run_combo next to the deletes.


@dataclass
class Context:
    stack: Stack
    ledger: Ledger
    apis: dict[str, HonchoAPI]
    # Expectations for queued work no worker has run yet. They join the
    # ledger once the next round's first worker has drained the queue.
    carried: list[Expectation] = field(default_factory=list)


def _marker() -> str:
    return f"cmx-{secrets.token_hex(4)}"


def run_combo(ctx: Context, tag: str, writer: str, workers: list[str]) -> str:
    """Write through ``writer``, process with ``workers``, delete, then stop.

    Covers create and read of every primitive with full, empty and null-valued
    metadata; queued representation, summary, dream and deletion work; and
    deletes of a conclusion and of a session whose derived memory must go too.
    """
    api = ctx.apis[writer]
    ws = f"cmx-{tag}"
    origin = f"{tag}: written by {api.label}, processed by {'+'.join(workers)}"

    ws_meta: dict[str, Any] = {
        "tag": tag,
        "nested": {"list": [1, "two", None], "empty": {}},
        "unicode": "café ✓",
        "nothing": None,
    }
    api.workspace(ws, ws_meta)
    api.peer(ws, "alice", {"role": "user", "empty_list": []})
    api.peer(ws, "bob", {})
    api.session(ws, "keep", ["alice", "bob"], {"k": "v"})
    api.session(ws, "doomed", ["alice", "bob"])
    ctx.ledger.add(
        WorkspaceExpect(origin, ws, ws_meta),
        PeerExpect(origin, ws, "alice", {"role": "user", "empty_list": []}),
        PeerExpect(origin, ws, "bob", {}),
        SessionExpect(origin, ws, "keep", {"k": "v"}),
    )

    ctx.stack.up(*workers)

    keep_markers = [_marker() for _ in range(4)]
    keep_messages: list[dict[str, Any]] = [
        {"peer_id": "alice", "content": f"I keep thinking about {m}"}
        for m in keep_markers
    ]
    keep_messages += [
        {"peer_id": "bob", "content": "", "metadata": {}},
        {"peer_id": "bob", "content": "ok", "metadata": {"reply": True, "n": None}},
    ]
    created = api.add_messages(ws, "keep", keep_messages)
    for sent, row in zip(keep_messages, created, strict=True):
        ctx.ledger.add(
            MessageExpect(
                origin,
                ws,
                "keep",
                row["id"],
                {
                    "content": sent["content"],
                    "peer_id": sent["peer_id"],
                    "session_id": "keep",
                    "metadata": sent.get("metadata", {}),
                },
            )
        )

    doomed_markers = [_marker() for _ in range(2)]
    doomed = api.add_messages(
        ws,
        "doomed",
        [{"peer_id": "alice", "content": f"soon gone: {m}"} for m in doomed_markers],
    )

    manual = api.create_conclusions(
        ws,
        [
            {
                "content": f"bob thinks alice likes {_marker()}",
                "observer_id": "bob",
                "observed_id": "alice",
                "session_id": "keep",
            },
            {
                "content": f"a global note {_marker()}",
                "observer_id": "bob",
                "observed_id": "alice",
            },
        ],
    )
    ctx.stack.wait_for_queue()

    api.schedule_dream(ws, "alice", "alice")
    ctx.stack.wait_for_queue()

    api.delete_conclusion(ws, manual[1]["id"])
    api.delete_session(ws, "doomed")
    ctx.stack.wait_for_queue()
    ctx.stack.stop(*workers)

    for row, session in ((manual[0], "keep"), (manual[1], None)):
        ctx.ledger.add(
            ConclusionExpect(
                origin,
                ws,
                row["id"],
                {
                    "content": row["content"],
                    "observer_id": "bob",
                    "observed_id": "alice",
                    "session_id": session,
                    "level": "explicit",
                    "source_ids": [],
                },
                present=row is manual[0],
            )
        )
    ctx.ledger.add(
        *(DerivedExpect(origin, ws, "alice", "alice", m) for m in keep_markers),
        *(
            DerivedExpect(origin, ws, "alice", "alice", m, present=False)
            for m in doomed_markers
        ),
        *(
            MessageExpect(origin, ws, "doomed", row["id"], {}, present=False)
            for row in doomed
        ),
        SessionExpect(origin, ws, "doomed", present=False),
        DeductionExpect(origin, ws, "alice", "alice"),
        SummaryExpect(origin, ws, "keep"),
    )
    return ws


def enqueue_unprocessed(ctx: Context, tag: str, writer: str) -> None:
    """Leave representation work queued by ``writer`` for the next image's worker.

    This is the state an image swap inherits: whatever the outgoing image
    enqueued is processed by the incoming one.
    """
    api = ctx.apis[writer]
    ws = f"cmx-{tag}"
    origin = f"{tag}: queued by {api.label}, left for the other image's worker"
    api.workspace(ws, {})
    api.peer(ws, "alice")
    api.session(ws, "carry", ["alice"])
    markers = [_marker() for _ in range(3)]
    api.add_messages(
        ws, "carry", [{"peer_id": "alice", "content": f"carry {m}"} for m in markers]
    )
    ctx.carried += [DerivedExpect(origin, ws, "alice", "alice", m) for m in markers]


def _legacy_id(n: int) -> str:
    return f"cmxlegacy{n:012d}"


def seed_legacy_rows(ctx: Context, ws: str) -> None:
    """Insert conclusions in every shape the database has held for provenance.

    Pre-3.2.0 writers kept a derived conclusion's premises in the
    ``documents.source_ids`` JSONB column or in ``internal_metadata``; the
    document_sources backfill drains them. These rows are what stays pending
    while the matrix runs with the backfill incomplete.
    """
    explicit = [
        c["id"]
        for c in ctx.apis["old"].list_conclusions(ws)
        if c["level"] == "explicit" and c["observer_id"] == c["observed_id"] == "alice"
    ]
    if len(explicit) < 2:
        raise RuntimeError(f"seed workspace {ws} has fewer than two explicit rows")
    a, b = explicit[:2]
    # (column source_ids, internal_metadata, session, what the API must return)
    shapes: list[tuple[Any, dict[str, Any], str | None, list[str]]] = [
        ([a, b], {}, "keep", [a, b]),
        (None, {"source_ids": [b, a]}, "keep", [b, a]),
        (None, {"premise_ids": [a]}, None, [a]),
        ([], {}, "keep", []),
        ("not-a-list", {}, None, []),
        ([a, "not a valid id", a], {}, None, [a]),
        (None, {}, None, []),
    ]
    origin = "legacy: inserted by SQL in pre-3.2.0 shapes"
    with ctx.stack.connect() as conn:
        for n, (column, metadata, session, expected) in enumerate(shapes):
            row_id = _legacy_id(n)
            content = f"legacy deduction {n}"
            conn.execute(
                "INSERT INTO documents (id, content, level, observer, observed,"
                + " workspace_name, session_name, internal_metadata, source_ids)"
                + " VALUES (%s, %s, 'deductive', 'alice', 'alice', %s, %s,"
                + " %s::jsonb, %s::jsonb)",
                (
                    row_id,
                    content,
                    ws,
                    session,
                    json.dumps(metadata),
                    None if column is None else json.dumps(column),
                ),
            )
            ctx.ledger.add(
                ConclusionExpect(
                    origin,
                    ws,
                    row_id,
                    {
                        "content": content,
                        "level": "deductive",
                        "session_id": session,
                        "source_ids": expected,
                    },
                )
            )
