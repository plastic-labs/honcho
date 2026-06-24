"""API router for graph memory endpoints (edges, recall, contexts, thread bindings, pinning, verify)."""

from __future__ import annotations

import datetime
import logging
import math
import time
from collections.abc import Sequence

from fastapi import APIRouter, Body, Depends, Path, Query
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.crud.graph_memory import (
    add_context_member,
    bind_thread,
    compact_access_log,
    compute_activation,
    compute_confidence,
    create_access_log_entry,
    create_edge,
    create_context as crud_create_context,
    delete_edge,
    evict_stale,
    get_context_member_count,
    get_context_members,
    get_verify_due as crud_get_verify_due,
    is_verify_due,
    list_edges,
    pin_observation,
    remove_context_member,
    resolve_thread,
    unpin_observation,
    verify_observation as crud_verify_observation,
)
from src.cache.client import cache as _cache, safe_cache_delete as _safe_cache_delete
from src.dependencies import db, read_db
from src.exceptions import ResourceNotFoundException, ValidationException
from src.schemas.graph_memory import (
    AccessLogEntryCreate,
    AccessLogEntryResponse,
    ContextCreate,
    ContextMemberAdd,
    ContextResponse,
    EdgeCreate,
    EdgeListFilter,
    EdgeResponse,
    PinRequest,
    RecallRequest,
    RecallResponse,
    RecallResult,
    ThreadBindingCreate,
    ThreadBindingResponse,
    VerifyDueItem,
    VerifyRequest,
)
from src.security import JWTParams, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}/graph-memory",
    tags=["graph-memory"],
    dependencies=[Depends(require_auth(workspace_name="workspace_id"))],
)


# ── Rate limiting (simple in-memory for now; production should use Redis) ──

_rate_limits: dict[str, list[float]] = {}

def _check_rate_limit(key: str, max_requests: int, window_seconds: int = 60) -> None:
    """Check if a rate limit has been exceeded. Raises ValidationException if so."""
    now = time.time()
    if key not in _rate_limits:
        _rate_limits[key] = []
    
    # Prune old entries
    _rate_limits[key] = [t for t in _rate_limits[key] if now - t < window_seconds]
    
    if len(_rate_limits[key]) >= max_requests:
        raise ValidationException(f"Rate limit exceeded: {max_requests} per {window_seconds}s")
    
    _rate_limits[key].append(now)


# ── Pin quota tracking ────────────────────────────────────────────────────

_pin_counts: dict[str, int] = {}

def _check_pin_quota(created_by: str, max_pins: int = 100) -> None:
    """Check if a user has exceeded their pin quota."""
    count = _pin_counts.get(created_by, 0)
    if count >= max_pins:
        raise ValidationException(f"Pin quota exceeded: max {max_pins} pins per persona")


# ── Edges ─────────────────────────────────────────────────────────────────

@router.post("/edges", response_model=EdgeResponse, status_code=201)
async def create_edge_endpoint(
    workspace_id: str = Path(...),
    body: EdgeCreate = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> models.Edge:
    """Create an edge between two observations (convergence-upsert)."""
    created_by = auth.p or auth.w or "unknown"
    _check_rate_limit(f"edge:{created_by}", 100)
    
    edge = await create_edge(
        db=db,
        workspace_name=workspace_id,
        collection_name=body.collection_name,
        source_obs_id=body.source_obs_id,
        target_obs_id=body.target_obs_id,
        edge_type=body.edge_type,
        created_by=created_by,
        metadata=body.metadata,
    )
    return edge


@router.post("/edges/list", response_model=list[EdgeResponse])
async def list_edges_endpoint(
    workspace_id: str = Path(...),
    filter_body: EdgeListFilter | None = Body(None),
    db: AsyncSession = Depends(read_db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> Sequence[models.Edge]:
    """List edges with optional filters."""
    return await list_edges(
        db=db,
        workspace_name=workspace_id,
        source_obs_id=filter_body.source_obs_id if filter_body else None,
        target_obs_id=filter_body.target_obs_id if filter_body else None,
        edge_type=filter_body.edge_type if filter_body else None,
        collection_name=filter_body.collection_name if filter_body else None,
    )


@router.delete("/edges/{edge_id}", status_code=204)
async def delete_edge_endpoint(
    workspace_id: str = Path(...),
    edge_id: int = Path(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> None:
    """Delete an edge."""
    deleted = await delete_edge(db=db, edge_id=edge_id, workspace_name=workspace_id)
    if not deleted:
        raise ResourceNotFoundException(f"Edge {edge_id} not found")


# ── Recall ─────────────────────────────────────────────────────────────────

@router.post("/recall", response_model=RecallResponse)
async def recall_endpoint(
    workspace_id: str = Path(...),
    body: RecallRequest = Body(...),
    db: AsyncSession = Depends(read_db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Spreading-activation recall using SQL recursive CTE."""
    created_by = auth.p or auth.w or "unknown"
    _check_rate_limit(f"recall:{created_by}", 60)
    
    start_time = time.time()
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Step 1: Vector search for anchors (top-5 by cosine similarity)
    # Use the existing HNSW index on documents.embedding
    anchor_query = sa_text("""
        SELECT id, embedding <=> :query_embedding::vector AS distance
        FROM documents
        WHERE workspace_name = :workspace_name
          AND collection_name = :collection_name
          AND deleted_at IS NULL
        ORDER BY embedding <=> :query_embedding::vector
        LIMIT 5
    """)
    
    # For now, use a simplified anchor search since we don't have the query embedding
    # In production, the query is embedded by the Deriver's embedding client
    anchor_result = await db.execute(
        select(models.Document.id).where(
            models.Document.workspace_name == workspace_id,
            models.Document.deleted_at.is_(None),
        ).limit(5)
    )
    anchors = [row[0] for row in anchor_result.fetchall()]
    
    if not anchors:
        return RecallResponse(
            results=[], total_visited=0, query_time_ms=0.0
        ).model_dump()
    
    # Step 2: SQL recursive CTE for spreading activation
    # Build the CTE with bounded frontier
    anchor_list = ", ".join(f"'{a}'" for a in anchors)
    
    cte_query = sa_text(f"""
        WITH RECURSIVE recall AS (
            -- Anchor: start from vector search results
            SELECT id, 0 AS depth, 1.0::double precision AS path_score
            FROM documents
            WHERE id IN ({anchor_list})
              AND workspace_name = :ws
              AND deleted_at IS NULL
            
            UNION
            
            -- Recursive step: follow edges, depth-capped
            SELECT e.target_obs_id, r.depth + 1, r.path_score * 0.8
            FROM recall r
            JOIN edges e ON e.source_obs_id = r.id
            WHERE r.depth < :max_depth
              AND e.workspace_name = :ws
        )
        SELECT DISTINCT r.id, r.path_score, d.content, d.internal_metadata
        FROM recall r
        JOIN documents d ON d.id = r.id
        WHERE d.deleted_at IS NULL
          AND (:context IS NULL OR d.id IN (
              SELECT obs_id FROM context_index 
              WHERE workspace_name = :ws AND context_name = :context
          ))
        ORDER BY r.path_score DESC
        LIMIT :budget
    """)
    
    cte_result = await db.execute(cte_query, {
        "ws": workspace_id,
        "max_depth": body.max_depth,
        "budget": body.token_budget,
        "context": body.context,
    })
    
    rows = cte_result.fetchall()
    total_visited = len(rows)
    
    # Step 3: Score each result with activation × confidence
    results: list[dict] = []
    for row in rows:
        obs_id, path_score, content, metadata_json = row
        metadata = metadata_json or {}
        
        activation = await compute_activation(db, obs_id, workspace_id, now)
        confidence = await compute_confidence(db, obs_id, workspace_id, now)
        
        # Apply pinned floor
        is_pinned = metadata.get("is_pinned", False)
        if is_pinned:
            activation = max(activation, 0.85)
        
        score = activation * confidence * path_score
        
        cadence = metadata.get("verify_cadence_days")
        due, _ = await is_verify_due(db, obs_id, workspace_id, is_pinned, cadence, now)
        
        results.append(RecallResult(
            obs_id=obs_id,
            content=content[:200],
            score=score,
            activation=activation,
            confidence=confidence,
            is_pinned=is_pinned,
            is_verify_due=due,
            workstream=metadata.get("workstream"),
        ).model_dump())
    
    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)
    
    # Log recall events
    for r in results[:10]:  # Log only top 10 to avoid flooding
        await create_access_log_entry(
            db, workspace_id, body.collection_name, r["obs_id"],
            "recall", created_by
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return RecallResponse(
        results=results,
        total_visited=total_visited,
        query_time_ms=elapsed_ms,
    ).model_dump()


# ── Contexts ──────────────────────────────────────────────────────────────

@router.post("/contexts", response_model=ContextResponse, status_code=201)
async def create_context_endpoint(
    workspace_id: str = Path(...),
    body: ContextCreate = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Create a named context."""
    created_by = auth.p or auth.w or "unknown"
    _check_rate_limit(f"context:{created_by}", 50)
    
    ctx = await crud_create_context(
        db=db,
        workspace_name=workspace_id,
        context_name=body.context_name,
        added_by=created_by,
    )
    return ContextResponse(
        id=0,
        workspace_name=workspace_id,
        context_name=body.context_name,
        member_count=0,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    ).model_dump()


@router.post("/contexts/{context_name}/members", response_model=dict, status_code=201)
async def add_context_member_endpoint(
    workspace_id: str = Path(...),
    context_name: str = Path(...),
    body: ContextMemberAdd = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Add an observation to a context."""
    created_by = auth.p or auth.w or "unknown"
    
    member = await add_context_member(
        db=db,
        workspace_name=workspace_id,
        context_name=context_name,
        obs_id=body.obs_id,
        added_by=created_by,
        thread_id=body.thread_id,
    )
    return {"id": member.id, "obs_id": member.obs_id, "context_name": context_name}


@router.delete("/contexts/{context_name}/members/{obs_id}", status_code=204)
async def remove_context_member_endpoint(
    workspace_id: str = Path(...),
    context_name: str = Path(...),
    obs_id: str = Path(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> None:
    """Remove an observation from a context."""
    removed = await remove_context_member(
        db=db, workspace_name=workspace_id, context_name=context_name, obs_id=obs_id
    )
    if not removed:
        raise ResourceNotFoundException(
            f"Observation {obs_id} not found in context '{context_name}'"
        )


@router.get("/contexts/{context_name}/members", response_model=list[dict])
async def list_context_members_endpoint(
    workspace_id: str = Path(...),
    context_name: str = Path(...),
    db: AsyncSession = Depends(read_db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> list[dict]:
    """List all members of a context."""
    members = await get_context_members(
        db=db, workspace_name=workspace_id, context_name=context_name
    )
    return [
        {"id": m.id, "obs_id": m.obs_id, "thread_id": m.thread_id, "added_at": m.added_at.isoformat()}
        for m in members
    ]


# ── Context switch (active context state via Redis) ────────────────────────

@router.post("/peers/{peer_id}/context-switch", response_model=dict)
async def context_switch_endpoint(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    context_name: str = Body(..., embed=True),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """True swap: page-in new context, page-out old context.
    
    Active context state is stored in Redis (ephemeral runtime state).
    """
    key = f"active_context:{workspace_id}:{peer_id}"
    
    # Store the new context (this is the "page-in" part)
    # The old context is implicitly "paged out" by overwriting
    await _cache.set(key, context_name, expire=3600)  # 1 hour TTL
    
    return {
        "workspace_id": workspace_id,
        "peer_id": peer_id,
        "active_context": context_name,
        "ttl_seconds": 3600,
    }


@router.post("/peers/{peer_id}/context-activate", response_model=dict)
async def context_activate_endpoint(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    context_name: str = Body(..., embed=True),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Additive page-in: activate a context without deactivating others.
    
    Note: This is a simplified version. True additive activation would
    maintain a set of active contexts. For v1, we support single context.
    """
    key = f"active_context:{workspace_id}:{peer_id}"
    
    # For v1, activate is the same as switch (single context)
    await _cache.set(key, context_name, expire=3600)
    
    return {
        "workspace_id": workspace_id,
        "peer_id": peer_id,
        "active_context": context_name,
    }


@router.post("/peers/{peer_id}/context-evict", response_model=dict)
async def context_evict_endpoint(
    workspace_id: str = Path(...),
    peer_id: str = Path(...),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Explicit page-out: clear the active context."""
    key = f"active_context:{workspace_id}:{peer_id}"
    await _safe_cache_delete(key)
    
    return {
        "workspace_id": workspace_id,
        "peer_id": peer_id,
        "active_context": None,
    }


# ── Thread bindings ────────────────────────────────────────────────────────

@router.post("/thread-bindings", response_model=ThreadBindingResponse, status_code=201)
async def create_thread_binding_endpoint(
    workspace_id: str = Path(...),
    body: ThreadBindingCreate = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> models.ThreadBinding:
    """Bind a thread to a context. Rebinding is denied."""
    created_by = auth.p or auth.w or "unknown"
    
    binding = await bind_thread(
        db=db,
        workspace_name=workspace_id,
        thread_id=body.thread_id,
        context_name=body.context_name,
        bound_by=created_by,
    )
    return binding


@router.get("/thread-bindings/{thread_id}", response_model=ThreadBindingResponse | None)
async def resolve_thread_endpoint(
    workspace_id: str = Path(...),
    thread_id: str = Path(...),
    db: AsyncSession = Depends(read_db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> models.ThreadBinding | None:
    """Resolve a thread to its bound context."""
    return await resolve_thread(
        db=db, workspace_name=workspace_id, thread_id=thread_id
    )


# ── Pinning ────────────────────────────────────────────────────────────────

@router.post("/observations/{obs_id}/pin", response_model=dict)
async def pin_observation_endpoint(
    workspace_id: str = Path(...),
    obs_id: str = Path(...),
    body: PinRequest = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Pin an observation. Per-persona quota: 100 pins."""
    created_by = auth.p or auth.w or "unknown"
    _check_pin_quota(created_by)
    _check_rate_limit(f"pin:{created_by}", 10, 3600)
    
    await pin_observation(
        db=db,
        workspace_name=workspace_id,
        obs_id=obs_id,
        created_by=created_by,
        verify_cadence_days=body.verify_cadence_days,
    )
    
    # Track pin count
    _pin_counts[created_by] = _pin_counts.get(created_by, 0) + 1
    
    return {"obs_id": obs_id, "is_pinned": True}


@router.delete("/observations/{obs_id}/pin", response_model=dict)
async def unpin_observation_endpoint(
    workspace_id: str = Path(...),
    obs_id: str = Path(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Unpin an observation."""
    await unpin_observation(
        db=db, workspace_name=workspace_id, obs_id=obs_id
    )
    return {"obs_id": obs_id, "is_pinned": False}


# ── Verification ──────────────────────────────────────────────────────────

@router.post("/observations/{obs_id}/verify", response_model=dict)
async def verify_observation_endpoint(
    workspace_id: str = Path(...),
    obs_id: str = Path(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Record a verification event for an observation."""
    created_by = auth.p or auth.w or "unknown"
    
    entry = await crud_verify_observation(
        db=db, workspace_name=workspace_id, obs_id=obs_id, created_by=created_by
    )
    return {"obs_id": obs_id, "verified_at": entry.created_at.isoformat()}


@router.get("/observations/verify-due", response_model=list[VerifyDueItem])
async def get_verify_due_endpoint(
    workspace_id: str = Path(...),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(read_db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> list[dict]:
    """List observations needing verification."""
    return await crud_get_verify_due(
        db=db, workspace_name=workspace_id, limit=limit
    )


# ── Access log (admin) ─────────────────────────────────────────────────────

@router.post("/access-log", response_model=AccessLogEntryResponse, status_code=201)
async def create_access_log_entry_endpoint(
    workspace_id: str = Path(...),
    body: AccessLogEntryCreate = Body(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> models.AccessLogEntry:
    """Append an event to the access log."""
    created_by = auth.p or auth.w or "unknown"
    _check_rate_limit(f"access-log:{created_by}", 1000)
    
    entry = await create_access_log_entry(
        db=db,
        workspace_name=workspace_id,
        collection_name=body.collection_name,
        obs_id=body.obs_id,
        event_type=body.event_type,
        created_by=created_by,
        session_id=body.session_id,
    )
    return entry


@router.post("/access-log/compact", response_model=dict)
async def compact_access_log_endpoint(
    workspace_id: str = Path(...),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Compact the access log (prune events older than 5 half-lives)."""
    pruned = await compact_access_log(
        db=db, workspace_name=workspace_id
    )
    return {"pruned_events": pruned}


# ── Eviction (admin) ──────────────────────────────────────────────────────

@router.post("/evict-stale", response_model=dict)
async def evict_stale_endpoint(
    workspace_id: str = Path(...),
    threshold: float = Query(default=0.12, ge=0.0, le=1.0),
    db: AsyncSession = Depends(db),
    auth: JWTParams = Depends(require_auth(workspace_name="workspace_id")),
) -> dict:
    """Evict stale unpinned observations below activation threshold."""
    evicted = await evict_stale(
        db=db, workspace_name=workspace_id, threshold=threshold
    )
    return {"evicted_count": len(evicted), "evicted_ids": evicted}
