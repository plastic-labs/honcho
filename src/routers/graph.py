import asyncio
import hashlib
import logging
import time

from fastapi import APIRouter, Body, Depends, Header, Path, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.crud.graph import (
    GraphColumns,
    build_graph_columns,
    get_graph_overview,
    get_graph_rows,
    get_graph_text,
    require_graph_names,
)
from src.dependencies import read_db, tracked_db
from src.security import require_auth
from src.utils.graph_arrow import MEDIA_TYPE, encode_graph

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/graph",
    tags=["graph"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


@router.get("/overview")
async def graph_overview(
    workspace_id: str = Path(...),
    db: AsyncSession = read_db,
):
    """Perspectives and conclusion counts for a workspace graph. Experimental."""
    await require_graph_names(db, workspace_id, ())
    overview = await get_graph_overview(db, workspace_id)
    return {
        "workspace": overview.workspace,
        "conclusions": overview.conclusions,
        "perspectives": [
            {
                "observer": p.observer,
                "observed": p.observed,
                "conclusions": p.conclusions,
                "total_kind": "exact",
                "derived": p.derived,
                "newest_at": p.newest_at.isoformat() if p.newest_at else None,
            }
            for p in overview.perspectives
        ],
    }


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Weak comparison against an If-None-Match list, as RFC 9110 13.1.2 asks."""
    if if_none_match is None:
        return False
    if if_none_match.strip() == "*":
        return True
    opaque = etag.removeprefix("W/")
    return any(
        tag.strip().removeprefix("W/") == opaque for tag in if_none_match.split(",")
    )


def _encode(cols: GraphColumns) -> tuple[bytes, str]:
    body = encode_graph(cols)
    return body, '"' + hashlib.sha256(body).hexdigest()[:32] + '"'


@router.get(
    "",
    response_class=Response,
    responses={
        200: {
            "description": "The graph as an Arrow IPC stream.",
            "content": {MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
        },
        304: {"description": "The graph matches the If-None-Match ETag."},
    },
)
async def graph(
    workspace_id: str = Path(...),
    observer: str | None = Query(None),
    observed: str | None = Query(None),
    if_none_match: str | None = Header(None),
):
    """Structure-only conclusion graph as an Arrow IPC stream. Experimental."""
    limit = settings.GRAPH.MAX_CONCLUSIONS
    started = time.perf_counter()
    # the connection goes back to the pool before the CPU work below
    async with tracked_db("graph", read_only=True) as db:
        await require_graph_names(db, workspace_id, (observer, observed))
        rows = await get_graph_rows(db, workspace_id, observer, observed, limit)
    queried = time.perf_counter()
    # in threads, so other requests on this worker run between GIL switches
    cols = await asyncio.to_thread(build_graph_columns, rows, limit)
    built = time.perf_counter()
    body, etag = await asyncio.to_thread(_encode, cols)
    encoded = time.perf_counter()
    timing = (
        f"db;dur={(queried - started) * 1000:.1f}, "
        + f"build;dur={(built - queried) * 1000:.1f}, "
        + f"encode;dur={(encoded - built) * 1000:.1f}"
    )
    headers = {
        "ETag": etag,
        "Cache-Control": "private, no-cache",
        "Server-Timing": timing,
    }
    logger.info(
        "Built a conclusion graph: %d conclusions, %d links, %d bytes (%s)",
        len(cols.ids),
        len(cols.source_targets),
        len(body),
        timing,
    )
    if etag_matches(if_none_match, etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type=MEDIA_TYPE, headers=headers)


TEXT_MAX_IDS = 20_000


@router.post("/text")
async def graph_text(
    workspace_id: str = Path(...),
    ids: list[str] = Body(..., embed=True, max_length=TEXT_MAX_IDS),
    db: AsyncSession = read_db,
):
    """Text and session for a batch of conclusion ids. Experimental."""
    await require_graph_names(db, workspace_id, ())
    rows = await get_graph_text(db, workspace_id, ids)
    return {"items": [{"id": i, "content": c, "session_id": s} for i, c, s in rows]}
