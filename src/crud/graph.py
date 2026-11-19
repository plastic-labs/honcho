import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Float,
    any_,
    bindparam,
    extract,
    func,
    literal,
    null,
    select,
    union_all,
)
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import ColumnElement

from src import models
from src.crud.lifecycle import live_document_row

LEVELS: tuple[str, ...] = ("explicit", "deductive", "inductive", "contradiction")
_LEVEL_INDEX = {name: i for i, name in enumerate(LEVELS)}


class GraphTooLargeError(Exception):
    """count is a lower bound: the check stops counting one past the limit."""

    def __init__(self, count: int, limit: int):
        super().__init__(f"graph has at least {count} conclusions; limit is {limit}")
        self.count = count
        self.limit = limit


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


@dataclass
class GraphColumns:
    ids: list[str] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    created_at: list[float] = field(default_factory=list)
    times_derived: list[int] = field(default_factory=list)
    perspective: list[int] = field(default_factory=list)
    perspectives: list[tuple[str, str]] = field(default_factory=list)
    source_offsets: list[int] = field(default_factory=list)
    source_index: list[int] = field(default_factory=list)
    unresolved: list[int] = field(default_factory=list)


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


async def get_graph_columns(
    db: AsyncSession,
    workspace_name: str,
    observer: str | None,
    observed: str | None,
    max_nodes: int,
) -> GraphColumns:
    D, S = models.Document, models.DocumentSource
    conds = _live(workspace_name, observer, observed)

    # stops at max_nodes + 1, like the list's 100,001 cap, so an oversized graph costs a bounded scan
    capped = (
        select(literal(1)).select_from(D).where(*conds).limit(max_nodes + 1).subquery()
    )
    count = await db.scalar(select(func.count()).select_from(capped)) or 0
    if count > max_nodes:
        raise GraphTooLargeError(count, max_nodes)

    n = (
        select(D.id, D.observer, D.observed, D.level, D.created_at, D.times_derived)
        .where(*conds)
        .cte("n")
        .prefix_with("MATERIALIZED")
    )
    nodes = select(
        literal(0).label("kind"),
        n.c.id,
        n.c.observer,
        n.c.observed,
        n.c.level,
        extract("epoch", n.c.created_at).cast(Float).label("ts"),
        n.c.times_derived.label("num"),
        null().cast(TEXT).label("source_id"),
    )
    links = (
        select(
            literal(1),
            S.derived_id,
            null(),
            null(),
            null(),
            null(),
            S.position,
            S.source_id,
        )
        .select_from(n)
        .join(S, S.derived_id == n.c.id)
        # n is already tenant-bound; constants on S prune its partitions and let the planner hash-join
        .where(
            S.tenant_id == models.scoped_tenant_id(), S.workspace_name == workspace_name
        )
    )
    # one statement so nodes and links come from the same snapshot under the autocommit read engine
    # server-side cursors need a transaction, which the autocommit read engine never opens
    result = await db.execute(union_all(nodes, links))

    cols = GraphColumns()
    index: dict[str, int] = {}
    perspective_index: dict[tuple[str, str], int] = {}
    links: list[tuple[str, int, str]] = []
    for kind, doc_id, obs, obd, level, ts, num, source_id in result:
        if kind == 1:
            links.append((doc_id, num, source_id))
            continue
        if len(index) >= max_nodes:
            raise GraphTooLargeError(len(index) + 1, max_nodes)
        key = (obs, obd)
        p = perspective_index.get(key)
        if p is None:
            p = perspective_index[key] = len(cols.perspectives)
            cols.perspectives.append(key)
        index[doc_id] = len(cols.ids)
        cols.ids.append(doc_id)
        cols.levels.append(_LEVEL_INDEX.get(level, 255))
        cols.created_at.append(ts)
        cols.times_derived.append(num)
        cols.perspective.append(p)

    by_node: list[list[tuple[int, int]]] = [[] for _ in cols.ids]
    cols.unresolved = [0] * len(cols.ids)
    for derived_id, position, source_id in links:
        node = index.get(derived_id)
        if node is None:
            continue
        source = index.get(source_id)
        # unresolved sources are counted, never named: the id may belong to a deleted conclusion
        if source is None:
            cols.unresolved[node] += 1
        else:
            by_node[node].append((position, source))

    offset = 0
    cols.source_offsets.append(0)
    for sources in by_node:
        sources.sort()
        cols.source_index.extend(s for _, s in sources)
        offset += len(sources)
        cols.source_offsets.append(offset)
    return cols


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
