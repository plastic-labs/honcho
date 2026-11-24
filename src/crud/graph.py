from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import accumulate
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
from src.crud.lifecycle import live_document_row, live_peer_row, live_workspace_row
from src.crud.peer import scope_peer_clause
from src.deriver.scope_backfill import COPIED_FROM_KEY
from src.exceptions import ResourceNotFoundException, ValidationException

# Row kinds of the graph statement.
ROW_NODE, ROW_LINK, ROW_MISSING, ROW_SCOPE = 0, 1, 2, 3

# Classes of a source that is not a conclusion in the response.
OUTSIDE, DELETED, UNKNOWN = 0, 1, 2

LINEAGE_NOT_APPLICABLE = "not_applicable"
LINEAGE_COMPLETE = "complete"
LINEAGE_PARTIAL = "partial"
LINEAGE_UNKNOWN = "unknown"


@dataclass
class PerspectiveSummary:
    observer: str
    observed: str
    conclusions: int
    derived: int
    newest_at: datetime | None


@dataclass
class GraphOverview:
    workspace: str
    conclusions: int
    perspectives: list[PerspectiveSummary]


@dataclass
class GraphColumns:
    """One entry per conclusion, ordered by creation time, then ID."""

    ids: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    observers: list[str] = field(default_factory=list)
    observeds: list[str] = field(default_factory=list)
    observer_kinds: list[str] = field(default_factory=list)
    sessions: list[str | None] = field(default_factory=list)
    created_us: array[int] = field(default_factory=lambda: array("q"))
    times_derived: array[int] = field(default_factory=lambda: array("I"))
    lineage: list[str] = field(default_factory=list)
    unresolved_outside: array[int] = field(default_factory=lambda: array("I"))
    unresolved_deleted: array[int] = field(default_factory=lambda: array("I"))
    unresolved_unknown: array[int] = field(default_factory=lambda: array("I"))
    is_copy: list[bool] = field(default_factory=list)
    copied_from: list[int | None] = field(default_factory=list)
    source_offsets: array[int] = field(default_factory=lambda: array("i", [0]))
    source_targets: array[int] = field(default_factory=lambda: array("I"))


async def require_graph_names(
    db: AsyncSession, workspace_name: str, peer_names: Iterable[str | None]
) -> None:
    """Raise ResourceNotFoundException for a missing workspace or peer."""
    W, P = models.Workspace, models.Peer
    tenant = models.scoped_tenant_id()
    workspace = await db.scalar(
        select(W.name).where(
            W.tenant_id == tenant, W.name == workspace_name, live_workspace_row()
        )
    )
    if workspace is None:
        raise ResourceNotFoundException(f"Workspace {workspace_name} not found")
    wanted = sorted({name for name in peer_names if name is not None})
    if not wanted:
        return
    found = set(
        await db.scalars(
            select(P.name).where(
                P.tenant_id == tenant,
                P.workspace_name == workspace_name,
                P.name.in_(wanted),
                live_peer_row(),
            )
        )
    )
    for name in wanted:
        if name not in found:
            raise ResourceNotFoundException(
                f"Peer {name} not found in workspace {workspace_name}"
            )


async def get_graph_overview(db: AsyncSession, workspace_name: str) -> GraphOverview:
    D = models.Document
    n = func.count()
    rows = (
        await db.execute(
            select(
                D.observer,
                D.observed,
                n.label("n"),
                func.count().filter(D.level != "explicit").label("derived"),
                func.max(D.created_at).label("newest"),
            )
            .where(live_document_row(workspace_name))
            .group_by(D.observer, D.observed)
            .order_by(n.desc(), D.observer, D.observed)
        )
    ).all()
    perspectives = [
        PerspectiveSummary(
            observer=r.observer,
            observed=r.observed,
            conclusions=r.n,
            derived=r.derived,
            newest_at=r.newest,
        )
        for r in rows
    ]
    return GraphOverview(
        workspace=workspace_name,
        conclusions=sum(p.conclusions for p in perspectives),
        perspectives=perspectives,
    )


def _live(
    workspace_name: str, observer: str | None, observed: str | None
) -> list[ColumnElement[bool]]:
    # the conclusions list's liveness: hides rows under deleted peers and sessions
    D = models.Document
    if observer is not None and observed is not None:
        return [live_document_row(workspace_name, peer_pair=(observer, observed))]
    conds = [live_document_row(workspace_name)]
    if observer is not None:
        conds.append(D.observer == observer)
    if observed is not None:
        conds.append(D.observed == observed)
    return conds


def _graph_statement(
    workspace_name: str, observer: str | None, observed: str | None, limit: int
) -> CompoundSelect[Any]:
    D, S = models.Document, models.DocumentSource
    tenant = models.scoped_tenant_id()

    # one row past the limit tells build_graph_columns the graph is too large
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
        .limit(limit + 1)
        .cte("n")
        .prefix_with("MATERIALIZED")
    )
    links = (
        select(S.derived_id, S.position, S.source_id)
        .select_from(n)
        .join(S, S.derived_id == n.c.id)
        # n is tenant-bound; constants on S prune partitions and allow a hash join
        .where(S.tenant_id == tenant, S.workspace_name == workspace_name)
        .cte("l")
    )
    missing = (
        select(links.c.source_id)
        .where(~select(n.c.id).where(n.c.id == links.c.source_id).exists())
        .distinct()
        .subquery("m")
    )
    # only this tenant's workspace decides a class, so an unreadable source is unknown
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


def _lineage(level: str, unresolved: int, sources: int) -> str:
    if unresolved:
        return LINEAGE_PARTIAL
    if level == "explicit":
        return LINEAGE_NOT_APPLICABLE
    return LINEAGE_COMPLETE if sources else LINEAGE_UNKNOWN


def build_graph_columns(
    rows: Iterable[Sequence[Any]], max_conclusions: int
) -> GraphColumns:
    """Turn the rows of the graph statement into columns.

    Raises:
        ValidationException: If the graph has more than max_conclusions conclusions.
    """
    nodes: list[Sequence[Any]] = []
    links: list[Sequence[Any]] = []
    missing: list[Sequence[Any]] = []
    scopes: list[Sequence[Any]] = []
    by_kind = (nodes.append, links.append, missing.append, scopes.append)
    for row in rows:
        by_kind[row[0]](row)
    if len(nodes) > max_conclusions:
        raise ValidationException(
            f"This graph has more than {max_conclusions:,} conclusions. "
            + "Filter by observer or observed to request a smaller one."
        )
    # Python orders str by code point, which is also UTF-8 byte order
    nodes.sort(key=itemgetter(7, 1))
    count = len(nodes)
    columns = list(zip(*nodes, strict=True)) or [()] * 10
    ids, observers, observeds, levels, sessions = columns[1:6]
    refs, created_us, times_derived, copies = columns[6:10]
    index = dict(zip(ids, range(count), strict=True))
    scope_observers = {row[2] for row in scopes}

    cols = GraphColumns(
        ids=list(ids),
        levels=list(levels),
        observers=list(observers),
        observeds=list(observeds),
        observer_kinds=[
            "scope" if obs in scope_observers else "peer" for obs in observers
        ],
        sessions=list(sessions),
        created_us=array("q", created_us),
        times_derived=array("I", times_derived),
        is_copy=[bool(copy) for copy in copies],
    )
    cols.copied_from = [
        index.get(ref) if copy else None for ref, copy in zip(refs, copies, strict=True)
    ]

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
    # a conclusion's sources follow their position in document_sources
    edges.sort()
    cols.source_targets = array("I", map(itemgetter(2), edges))
    degree = [0] * count
    for node, _, _ in edges:
        degree[node] += 1
    cols.source_offsets.extend(accumulate(degree))
    cols.lineage = [
        _lineage(level, out + gone + unknown, sources)
        for level, out, gone, unknown, sources in zip(
            levels, *unresolved, degree, strict=True
        )
    ]
    return cols


async def get_graph_rows(
    db: AsyncSession,
    workspace_name: str,
    observer: str | None,
    observed: str | None,
    max_conclusions: int,
) -> Sequence[Sequence[Any]]:
    # one statement, so nodes, links and source classes share a snapshot under the
    # autocommit read engine, which never opens the transaction a server-side cursor
    # would need
    statement = _graph_statement(workspace_name, observer, observed, max_conclusions)
    result = await db.execute(statement)
    return result.tuples().all()


async def get_graph_columns(
    db: AsyncSession,
    workspace_name: str,
    observer: str | None,
    observed: str | None,
    max_conclusions: int,
) -> GraphColumns:
    rows = await get_graph_rows(db, workspace_name, observer, observed, max_conclusions)
    return build_graph_columns(rows, max_conclusions)


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
