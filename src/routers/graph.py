import hashlib
import logging
import time

from fastapi import APIRouter, Body, Depends, Header, Path, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.crud.graph import (
    build_graph_columns,
    get_graph_overview,
    get_graph_rows,
    get_graph_text,
)
from src.dependencies import read_db
from src.security import require_auth
from src.utils.graph_wire import CONTENT_TYPE, encode_graph

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
    overview = await get_graph_overview(db, workspace_id)
    return {
        "workspace": overview.workspace,
        "version": overview.version,
        "conclusions": overview.conclusions,
        "perspectives": [
            {
                "observer": p.observer,
                "observed": p.observed,
                "conclusions": p.conclusions,
                "total_kind": "exact",
                "derived": p.derived,
                "newest_at": p.newest_at.isoformat() if p.newest_at else None,
                "version": p.version,
            }
            for p in overview.perspectives
        ],
    }


@router.get("")
async def graph(
    workspace_id: str = Path(...),
    observer: str | None = Query(None),
    observed: str | None = Query(None),
    if_none_match: str | None = Header(None),
    db: AsyncSession = read_db,
):
    """Structure-only conclusion graph as HONCHOGR v2 bytes. Experimental."""
    started = time.perf_counter()
    rows = await get_graph_rows(db, workspace_id, observer, observed)
    queried = time.perf_counter()
    cols = build_graph_columns(rows)
    built = time.perf_counter()
    body = encode_graph(cols)
    encoded = time.perf_counter()
    etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
    timing = (
        f"db;dur={(queried - started) * 1000:.1f}, "
        f"build;dur={(built - queried) * 1000:.1f}, "
        f"encode;dur={(encoded - built) * 1000:.1f}"
    )
    headers = {
        "ETag": etag,
        "Cache-Control": "private, no-cache",
        "Server-Timing": timing,
    }
    logger.info(
        "graph %s %s/%s nodes=%d links=%d bytes=%d %s",
        workspace_id,
        observer,
        observed,
        len(cols.ids),
        len(cols.csr_targets),
        len(body),
        timing,
    )
    if if_none_match == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type=CONTENT_TYPE, headers=headers)


TEXT_MAX_IDS = 20_000


@router.post("/text")
async def graph_text(
    workspace_id: str = Path(...),
    ids: list[str] = Body(..., embed=True, max_length=TEXT_MAX_IDS),
    db: AsyncSession = read_db,
):
    """Text and session for a batch of conclusion ids. Experimental."""
    rows = await get_graph_text(db, workspace_id, ids)
    return {"items": [{"id": i, "content": c, "session_id": s} for i, c, s in rows]}
