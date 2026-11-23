import hashlib
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import accumulate, compress
from operator import itemgetter
from typing import Any

from sqlalchemy import (
    BigInteger,
    CompoundSelect,
    and_,
    any_,
    bindparam,
    case,
    extract,
    func,
    literal,
    null,
    select,
    union_all,
)
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased
from sqlalchemy.sql import ColumnElement

from src import models
from src.crud.lifecycle import live_document_row
from src.crud.peer import scope_peer_clause
from src.deriver.scope_backfill import COPIED_FROM_KEY
from src.utils.honchogr_reference import NO_COPIED_FROM, NULL_SESSION, TYPE_DICTIONARY

_TYPE_INDEX = {name: i for i, name in enumerate(TYPE_DICTIONARY)}

# Row kinds of the graph statement.
ROW_NODE, ROW_LINK, ROW_MISSING, ROW_SCOPE = 0, 1, 2, 3

# Classes of a source that is not a node of the response, in WIRE.md's terms.
OUTSIDE, DELETED, UNKNOWN = 0, 1, 2


@dataclass
class PerspectiveSummary:
    observer: str
    observed: str
    conclusions: int
    derived: int
    newest_at: datetime | None
    version: str


@dataclass
class GraphOverview:
    workspace: str
    conclusions: int
    perspectives: list[PerspectiveSummary]
    version: str


def _u32() -> array[int]:
    return array("I")


@dataclass
class GraphColumns:
    """HONCHOGR v2 columns in node order: created_us, then ID bytes."""

    ids: list[str] = field(default_factory=list)
    node_type: array[int] = field(default_factory=_u32)
    node_session: array[int] = field(default_factory=_u32)
    node_perspective: array[int] = field(default_factory=_u32)
    created_us: array[int] = field(default_factory=lambda: array("q"))
    unresolved_outside: array[int] = field(default_factory=_u32)
    unresolved_deleted: array[int] = field(default_factory=_u32)
    unresolved_unknown: array[int] = field(default_factory=_u32)
    times_derived: array[int] = field(default_factory=_u32)
    node_flags: array[int] = field(default_factory=_u32)
    copied_from: array[int] = field(default_factory=_u32)
    csr_offsets: array[int] = field(default_factory=lambda: array("I", [0]))
    csr_targets: array[int] = field(default_factory=_u32)
    session_dict: list[str] = field(default_factory=list)
    perspective_dict: list[tuple[str, str, str]] = field(default_factory=list)


def _version(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _live(
    workspace_name: str, observer: str | None, observed: str | None
) -> list[ColumnElement[bool]]:
    # same liveness as the conclusions list: hides rows under deleted peers and sessions, binds the tenant
    D = models.Document
    if observer is not None and observed is not None:
        return [live_document_row(workspace_name, peer_pair=(observer, observed))]
    conds = [live_document_row(workspace_name)]
    if observer is not None:
        conds.append(D.observer == observer)
    if observed is not None:
        conds.append(D.observed == observed)
    return conds


async def get_graph_overview(db: AsyncSession, workspace_name: str) -> GraphOverview:
    D, C = models.Document, models.Collection
    n = func.count()
    # sum(times_derived) catches dedup reinforcement, which changes no timestamp
    rows = (
        await db.execute(
            select(
                D.observer,
                D.observed,
                n.label("n"),
                func.count().filter(D.level != "explicit").label("derived"),
                func.max(D.created_at).label("newest"),
                func.sum(D.times_derived).label("reinforced"),
            )
            .where(live_document_row(workspace_name))
            .group_by(D.observer, D.observed)
            .order_by(n.desc(), D.observer, D.observed)
        )
    ).all()
    generations = {
        (r.observer, r.observed): r.deletion_generation
        for r in (
            await db.execute(
                select(C.observer, C.observed, C.deletion_generation).where(
                    C.tenant_id == models.scoped_tenant_id(),
                    C.workspace_name == workspace_name,
                )
            )
        ).all()
    }
    perspectives = [
        PerspectiveSummary(
            observer=r.observer,
            observed=r.observed,
            conclusions=r.n,
            derived=r.derived,
            newest_at=r.newest,
            version=_version(
                r.n, r.newest, r.reinforced, generations.get((r.observer, r.observed))
            ),
        )
        for r in rows
    ]
    return GraphOverview(
        workspace=workspace_name,
        conclusions=sum(p.conclusions for p in perspectives),
        perspectives=perspectives,
        version=_version(*(p.version for p in perspectives)),
    )


def _graph_statement(
    workspace_name: str, observer: str | None, observed: str | None
) -> CompoundSelect[Any]:
    D, S = models.Document, models.DocumentSource
    tenant = models.scoped_tenant_id()

    n = (
        select(
            D.id,
            D.observer,
            D.observed,
            D.level,
            D.session_name,
            D.internal_metadata[COPIED_FROM_KEY].astext.label("copied_from"),
            (extract("epoch", D.created_at) * 1_000_000)
            .cast(BigInteger)
            .label("created_us"),
            D.times_derived,
            D.internal_metadata.has_key(COPIED_FROM_KEY).label("is_copy"),
        )
        .where(*_live(workspace_name, observer, observed))
        .cte("n")
        .prefix_with("MATERIALIZED")
    )
    links = (
        select(S.derived_id, S.position, S.source_id)
        .select_from(n)
        .join(S, S.derived_id == n.c.id)
        # n is already tenant-bound; constants on S prune its partitions and let the planner hash-join
        .where(S.tenant_id == tenant, S.workspace_name == workspace_name)
        .cte("l")
    )
    missing = (
        select(links.c.source_id)
        .where(~select(n.c.id).where(n.c.id == links.c.source_id).exists())
        .distinct()
        .subquery("m")
    )
    # Only rows of this tenant and workspace decide a class, so a source the caller cannot read stays unknown.
    d = aliased(models.Document)
    source_class = case(
        (live_document_row(workspace_name, d, tenant_id=tenant), OUTSIDE),
        (d.deleted_at.is_not(None), DELETED),
        else_=UNKNOWN,
    )

    nodes = select(
        literal(ROW_NODE).label("kind"),
        n.c.id,
        n.c.observer,
        n.c.observed,
        n.c.level,
        n.c.session_name,
        n.c.copied_from.label("ref"),
        n.c.created_us,
        n.c.times_derived.label("num"),
        n.c.is_copy.label("flag"),
    )
    link_rows = select(
        literal(ROW_LINK),
        links.c.derived_id,
        null(),
        null(),
        null(),
        null(),
        links.c.source_id,
        null(),
        links.c.position,
        null(),
    )
    missing_rows = (
        select(
            literal(ROW_MISSING),
            null(),
            null(),
            null(),
            null(),
            null(),
            missing.c.source_id,
            null(),
            source_class,
            null(),
        )
        .select_from(missing)
        .outerjoin(
            d,
            and_(
                d.tenant_id == tenant,
                d.workspace_name == workspace_name,
                d.id == missing.c.source_id,
            ),
        )
    )
    scope_rows = select(
        literal(ROW_SCOPE),
        null(),
        models.Peer.name,
        null(),
        null(),
        null(),
        null(),
        null(),
        null(),
        null(),
    ).where(
        models.Peer.tenant_id == tenant,
        models.Peer.workspace_name == workspace_name,
        models.Peer.name.in_(select(n.c.observer)),
        scope_peer_clause(),
    )
    return union_all(nodes, link_rows, missing_rows, scope_rows)


def build_graph_columns(rows: Iterable[Sequence[Any]]) -> GraphColumns:
    """Turn the rows of the graph statement into HONCHOGR v2 columns.

    Raises:
        ValueError: If a conclusion has a level outside the v2 type dictionary.
    """
    nodes: list[Sequence[Any]] = []
    links: list[Sequence[Any]] = []
    missing: list[Sequence[Any]] = []
    scopes: list[Sequence[Any]] = []
    by_kind = (nodes.append, links.append, missing.append, scopes.append)
    for row in rows:
        by_kind[row[0]](row)
    # created_us, then ID; Python orders str by code point, which is also UTF-8 byte order
    nodes.sort(key=itemgetter(7, 1))
    count = len(nodes)
    columns = list(zip(*nodes, strict=True)) or [()] * 10
    ids, observers, observeds, levels, sessions = columns[1:6]
    refs, created_us, times_derived, copies = columns[6:10]

    unknown_levels = set(levels) - _TYPE_INDEX.keys()
    if unknown_levels:
        raise ValueError(
            f"HONCHOGR v2 has no type for conclusion levels {sorted(map(str, unknown_levels))}"
        )
    cols = GraphColumns(ids=list(ids))
    index = dict(zip(cols.ids, range(count), strict=True))
    cols.node_type = array("I", map(_TYPE_INDEX.__getitem__, levels))

    cols.session_dict = sorted(set(sessions) - {None})
    session_index: dict[str | None, int] = {None: NULL_SESSION}
    session_index.update(
        zip(cols.session_dict, range(len(cols.session_dict)), strict=True)
    )
    cols.node_session = array("I", map(session_index.__getitem__, sessions))

    pairs = list(zip(observers, observeds, strict=True))
    ordered_pairs = sorted(set(pairs))
    perspective_index = dict(zip(ordered_pairs, range(len(ordered_pairs)), strict=True))
    cols.node_perspective = array("I", map(perspective_index.__getitem__, pairs))
    scope_observers = {row[2] for row in scopes}
    cols.perspective_dict = [
        (obs, obd, "scope" if obs in scope_observers else "peer")
        for obs, obd in ordered_pairs
    ]

    cols.created_us = array("q", created_us)
    cols.times_derived = array("I", times_derived)
    cols.node_flags = array("I", map(bool, copies))
    cols.copied_from = array("I", [NO_COPIED_FROM]) * count
    for node in compress(range(count), copies):
        cols.copied_from[node] = index.get(refs[node], NO_COPIED_FROM)

    classes = {row[6]: row[8] for row in missing}
    unresolved = tuple(array("I", [0]) * count for _ in range(3))
    edges: list[tuple[int, int, int]] = []
    for row in links:
        node = index[row[1]]
        source = index.get(row[6])
        # unresolved sources are counted, never named
        if source is None:
            unresolved[classes.get(row[6], UNKNOWN)][node] += 1
        else:
            edges.append((node, row[8], source))
    cols.unresolved_outside, cols.unresolved_deleted, cols.unresolved_unknown = (
        unresolved
    )
    # a node's sources follow their position in document_sources
    edges.sort()
    cols.csr_targets = array("I", map(itemgetter(2), edges))
    degree = [0] * count
    for node, _, _ in edges:
        degree[node] += 1
    cols.csr_offsets.extend(accumulate(degree))
    return cols


async def get_graph_rows(
    db: AsyncSession,
    workspace_name: str,
    observer: str | None,
    observed: str | None,
) -> Sequence[Sequence[Any]]:
    # one statement so nodes, links and source classes come from the same snapshot under the autocommit read engine
    # server-side cursors need a transaction, which the autocommit read engine never opens
    result = await db.execute(_graph_statement(workspace_name, observer, observed))
    return result.tuples().all()


async def get_graph_columns(
    db: AsyncSession,
    workspace_name: str,
    observer: str | None,
    observed: str | None,
) -> GraphColumns:
    return build_graph_columns(
        await get_graph_rows(db, workspace_name, observer, observed)
    )


async def get_graph_text(
    db: AsyncSession, workspace_name: str, ids: list[str]
) -> list[tuple[str, str, str | None]]:
    D = models.Document
    rows = await db.execute(
        select(D.id, D.content, D.session_name).where(
            live_document_row(workspace_name),
            D.id == any_(bindparam("ids", ids, type_=ARRAY(TEXT))),
        )
    )
    return [(r.id, r.content, r.session_name) for r in rows]
