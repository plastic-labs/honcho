"""CRUD operations for graph memory tables (edges, access_log, contexts, thread bindings)."""

from __future__ import annotations

import datetime
import logging
import math
from collections.abc import Sequence

from sqlalchemy import Select, func, select, text, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.exceptions import ResourceNotFoundException, ValidationException
from src.utils.types import EdgeType, AccessLogEventType


async def _get_document(db: AsyncSession, obs_id: str, workspace_name: str) -> models.Document | None:
    """Get a document by ID (internal helper)."""
    result = await db.execute(
        select(models.Document).where(
            models.Document.id == obs_id,
            models.Document.workspace_name == workspace_name,
            models.Document.deleted_at.is_(None),
        )
    )
    return result.scalar_one_or_none()

logger = logging.getLogger(__name__)

# ── Decay constants (matches spec §3) ─────────────────────────────────────

ACTIVATION_HALF_LIFE_HOURS = 24.0
CONFIDENCE_HALF_LIFE_DAYS = 30.0
CONFIDENCE_THRESHOLD = 0.3
PINNED_FLOOR = 0.85
EVICTION_THRESHOLD = 0.12
REHYDRATE_RESTORE = 0.60
LOG_RETENTION_HALF_LIVES = 5.0

EVENT_WEIGHTS = {
    "access": 0.3,
    "verify": 1.0,
    "recall": 0.5,
    "promote": 1.0,
    "rehydrate": 1.0,
    "evict": 0.0,
}


# ── Helper: compute activation from access log ─────────────────────────────

async def compute_activation(
    db: AsyncSession,
    obs_id: str,
    workspace_name: str,
    now: datetime.datetime | None = None,
) -> float:
    """Derive activation from the access log (spec §3).
    
    activation = Σ(distinct_sources) Σ(events from that source)
                  weight(event) * exp(-Δt / half_life)
    
    Same-source repeats get diminishing returns.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    
    result = await db.execute(
        select(models.AccessLogEntry).where(
            models.AccessLogEntry.obs_id == obs_id,
            models.AccessLogEntry.workspace_name == workspace_name,
        ).order_by(models.AccessLogEntry.created_at)
    )
    events: Sequence[models.AccessLogEntry] = result.scalars().all()
    
    if not events:
        return 0.0
    
    # Group by created_by (source_id)
    source_events: dict[str, list[models.AccessLogEntry]] = {}
    for event in events:
        source_events.setdefault(event.created_by, []).append(event)
    
    total = 0.0
    for source_id, source_evts in source_events.items():
        source_sum = 0.0
        for i, event in enumerate(source_evts):
            weight = EVENT_WEIGHTS.get(event.event_type, 0.0)
            if weight == 0.0:
                continue
            dt = (now - event.created_at).total_seconds()
            dt_hours = dt / 3600.0
            decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)
            # Diminishing returns for same-source repeats
            repeat_factor = 1.0 / (1.0 + math.log(1.0 + i))
            source_sum += weight * decay * repeat_factor
        total += source_sum
    
    return total


async def compute_confidence(
    db: AsyncSession,
    obs_id: str,
    workspace_name: str,
    now: datetime.datetime | None = None,
) -> float:
    """Derive confidence from the access log (spec §3).
    
    confidence = exp(-(now - last_verify) / verify_half_life)
    
    Pure function of last_verify and now — NO compounding.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    
    result = await db.execute(
        select(models.AccessLogEntry).where(
            models.AccessLogEntry.obs_id == obs_id,
            models.AccessLogEntry.workspace_name == workspace_name,
            models.AccessLogEntry.event_type == "verify",
        ).order_by(models.AccessLogEntry.created_at.desc()).limit(1)
    )
    last_verify: models.AccessLogEntry | None = result.scalar_one_or_none()
    
    if last_verify is None:
        return 0.0  # Never verified = no confidence
    
    dt = (now - last_verify.created_at).total_seconds()
    dt_hours = dt / 3600.0
    half_life_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0
    return math.exp(-dt_hours / half_life_hours)


async def is_verify_due(
    db: AsyncSession,
    obs_id: str,
    workspace_name: str,
    is_pinned: bool = False,
    verify_cadence_days: float | None = None,
    now: datetime.datetime | None = None,
) -> tuple[bool, str]:
    """Two triggers (spec §7):
    1. Explicit cadence elapsed (pins only, activation-independent)
    2. Confidence < threshold (always active)
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    
    # Trigger 1: explicit cadence
    if is_pinned and verify_cadence_days is not None:
        result = await db.execute(
            select(models.AccessLogEntry).where(
                models.AccessLogEntry.obs_id == obs_id,
                models.AccessLogEntry.workspace_name == workspace_name,
                models.AccessLogEntry.event_type == "verify",
            ).order_by(models.AccessLogEntry.created_at.desc()).limit(1)
        )
        last_verify: models.AccessLogEntry | None = result.scalar_one_or_none()
        if last_verify is not None:
            elapsed_days = (now - last_verify.created_at).total_seconds() / 86400.0
            if elapsed_days >= verify_cadence_days:
                return True, f"cadence ({verify_cadence_days:.0f}d) elapsed"
    
    # Trigger 2: confidence threshold
    conf = await compute_confidence(db, obs_id, workspace_name, now)
    if conf < CONFIDENCE_THRESHOLD:
        return True, f"confidence ({conf:.3f}) < threshold ({CONFIDENCE_THRESHOLD})"
    
    return False, ""


# ── Edge CRUD ─────────────────────────────────────────────────────────────

async def create_edge(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    source_obs_id: str,
    target_obs_id: str,
    edge_type: EdgeType,
    created_by: str,
    metadata: dict | None = None,
) -> models.Edge:
    """Create an edge with convergence-upsert (INSERT ... ON CONFLICT).
    
    If an edge with the same (workspace, collection, source, target, type)
    already exists, the existing edge's metadata is updated (reinforced).
    """
    # Verify both observations exist
    source_doc = await _get_document(db, source_obs_id, workspace_name)
    if not source_doc:
        raise ResourceNotFoundException(f"Source observation {source_obs_id} not found")
    target_doc = await _get_document(db, target_obs_id, workspace_name)
    if not target_doc:
        raise ResourceNotFoundException(f"Target observation {target_obs_id} not found")
    
    if source_obs_id == target_obs_id:
        raise ValidationException("Source and target observations must be different")
    
    # Use raw SQL for ON CONFLICT upsert
    from sqlalchemy import text as sa_text
    
    stmt = sa_text("""
        INSERT INTO edges (workspace_name, collection_name, source_obs_id, target_obs_id, edge_type, created_by, metadata)
        VALUES (:workspace_name, :collection_name, :source_obs_id, :target_obs_id, :edge_type, :created_by, :metadata)
        ON CONFLICT (workspace_name, collection_name, source_obs_id, target_obs_id, edge_type)
        DO UPDATE SET
            metadata = edges.metadata || jsonb_build_object('reinforced_by', 
                COALESCE(edges.metadata->'reinforced_by', '[]'::jsonb) || to_jsonb(:created_by::text)),
            created_at = NOW()
        RETURNING id
    """)
    
    result = await db.execute(stmt, {
        "workspace_name": workspace_name,
        "collection_name": collection_name,
        "source_obs_id": source_obs_id,
        "target_obs_id": target_obs_id,
        "edge_type": edge_type,
        "created_by": created_by,
        "metadata": metadata or {},
    })
    edge_id = result.scalar_one()
    await db.commit()
    
    # Fetch and return the edge
    edge_result = await db.execute(
        select(models.Edge).where(models.Edge.id == edge_id)
    )
    return edge_result.scalar_one()


async def list_edges(
    db: AsyncSession,
    workspace_name: str,
    source_obs_id: str | None = None,
    target_obs_id: str | None = None,
    edge_type: EdgeType | None = None,
    collection_name: str | None = None,
    limit: int = 100,
) -> Sequence[models.Edge]:
    """List edges with optional filters."""
    stmt = select(models.Edge).where(models.Edge.workspace_name == workspace_name)
    
    if source_obs_id:
        stmt = stmt.where(models.Edge.source_obs_id == source_obs_id)
    if target_obs_id:
        stmt = stmt.where(models.Edge.target_obs_id == target_obs_id)
    if edge_type:
        stmt = stmt.where(models.Edge.edge_type == edge_type)
    if collection_name:
        stmt = stmt.where(models.Edge.collection_name == collection_name)
    
    stmt = stmt.order_by(models.Edge.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


async def delete_edge(db: AsyncSession, edge_id: int, workspace_name: str) -> bool:
    """Delete an edge by ID."""
    result = await db.execute(
        sa_delete(models.Edge).where(
            models.Edge.id == edge_id,
            models.Edge.workspace_name == workspace_name,
        )
    )
    await db.commit()
    return result.rowcount > 0


# ── Access log CRUD ───────────────────────────────────────────────────────

async def create_access_log_entry(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    obs_id: str,
    event_type: AccessLogEventType,
    created_by: str,
    session_id: str | None = None,
) -> models.AccessLogEntry:
    """Append an event to the access log."""
    entry = models.AccessLogEntry(
        workspace_name=workspace_name,
        collection_name=collection_name,
        obs_id=obs_id,
        event_type=event_type,
        created_by=created_by,
        session_id=session_id,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def compact_access_log(
    db: AsyncSession,
    workspace_name: str | None = None,
) -> int:
    """Compact the access log: prune events older than 5 half-lives.
    
    Returns number of pruned events.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        hours=LOG_RETENTION_HALF_LIVES * ACTIVATION_HALF_LIFE_HOURS
    )
    
    stmt = sa_delete(models.AccessLogEntry).where(
        models.AccessLogEntry.created_at < cutoff
    )
    if workspace_name:
        stmt = stmt.where(models.AccessLogEntry.workspace_name == workspace_name)
    
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount


# ── Context CRUD ───────────────────────────────────────────────────────────

async def create_context(
    db: AsyncSession,
    workspace_name: str,
    context_name: str,
    added_by: str,
) -> models.ContextIndex:
    """Create a context by adding the first member.
    
    A context exists by virtue of having members — there is no separate
    context metadata table. The first member creation is the context creation.
    """
    # Just return a placeholder — contexts are defined by their members
    return models.ContextIndex(
        workspace_name=workspace_name,
        context_name=context_name,
        obs_id="",  # Will be set when first member is added
        added_by=added_by,
    )


async def add_context_member(
    db: AsyncSession,
    workspace_name: str,
    context_name: str,
    obs_id: str,
    added_by: str,
    thread_id: str | None = None,
) -> models.ContextIndex:
    """Add an observation to a context."""
    # Verify observation exists
    doc = await _get_document(db, obs_id, workspace_name)
    if not doc:
        raise ResourceNotFoundException(f"Observation {obs_id} not found")
    
    member = models.ContextIndex(
        workspace_name=workspace_name,
        context_name=context_name,
        obs_id=obs_id,
        thread_id=thread_id,
        added_by=added_by,
    )
    db.add(member)
    try:
        await db.commit()
        await db.refresh(member)
    except Exception:
        await db.rollback()
        raise ValidationException(
            f"Observation {obs_id} is already a member of context '{context_name}'"
        )
    return member


async def remove_context_member(
    db: AsyncSession,
    workspace_name: str,
    context_name: str,
    obs_id: str,
) -> bool:
    """Remove an observation from a context."""
    result = await db.execute(
        sa_delete(models.ContextIndex).where(
            models.ContextIndex.workspace_name == workspace_name,
            models.ContextIndex.context_name == context_name,
            models.ContextIndex.obs_id == obs_id,
        )
    )
    await db.commit()
    return result.rowcount > 0


async def get_context_members(
    db: AsyncSession,
    workspace_name: str,
    context_name: str,
) -> Sequence[models.ContextIndex]:
    """Get all members of a context."""
    result = await db.execute(
        select(models.ContextIndex).where(
            models.ContextIndex.workspace_name == workspace_name,
            models.ContextIndex.context_name == context_name,
        )
    )
    return result.scalars().all()


async def get_context_member_count(
    db: AsyncSession,
    workspace_name: str,
    context_name: str,
) -> int:
    """Get the number of members in a context."""
    result = await db.execute(
        select(func.count()).select_from(models.ContextIndex).where(
            models.ContextIndex.workspace_name == workspace_name,
            models.ContextIndex.context_name == context_name,
        )
    )
    return result.scalar() or 0


# ── Thread binding CRUD ────────────────────────────────────────────────────

async def bind_thread(
    db: AsyncSession,
    workspace_name: str,
    thread_id: str,
    context_name: str,
    bound_by: str,
) -> models.ThreadBinding:
    """Bind a thread to a context. Rebinding is denied."""
    binding = models.ThreadBinding(
        workspace_name=workspace_name,
        thread_id=thread_id,
        context_name=context_name,
        bound_by=bound_by,
    )
    db.add(binding)
    try:
        await db.commit()
        await db.refresh(binding)
    except Exception:
        await db.rollback()
        raise ValidationException(
            f"Thread {thread_id} is already bound to a context"
        )
    return binding


async def resolve_thread(
    db: AsyncSession,
    workspace_name: str,
    thread_id: str,
) -> models.ThreadBinding | None:
    """Resolve a thread to its bound context."""
    result = await db.execute(
        select(models.ThreadBinding).where(
            models.ThreadBinding.workspace_name == workspace_name,
            models.ThreadBinding.thread_id == thread_id,
        )
    )
    return result.scalar_one_or_none()


# ── Pin / Verify CRUD ────────────────────────────────────────────────────

async def pin_observation(
    db: AsyncSession,
    workspace_name: str,
    obs_id: str,
    created_by: str,
    verify_cadence_days: int | None = None,
) -> bool:
    """Pin an observation by setting metadata."""
    doc = await _get_document(db, obs_id, workspace_name)
    if not doc:
        raise ResourceNotFoundException(f"Observation {obs_id} not found")
    
    metadata = dict(doc.internal_metadata) if doc.internal_metadata else {}
    metadata["is_pinned"] = True
    metadata["pinned_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata["pinned_by"] = created_by
    if verify_cadence_days is not None:
        metadata["verify_cadence_days"] = verify_cadence_days
    else:
        metadata.pop("verify_cadence_days", None)
    
    doc.internal_metadata = metadata
    await db.commit()
    return True


async def unpin_observation(
    db: AsyncSession,
    workspace_name: str,
    obs_id: str,
) -> bool:
    """Unpin an observation."""
    doc = await _get_document(db, obs_id, workspace_name)
    if not doc:
        raise ResourceNotFoundException(f"Observation {obs_id} not found")
    
    metadata = dict(doc.internal_metadata) if doc.internal_metadata else {}
    metadata["is_pinned"] = False
    metadata.pop("pinned_at", None)
    metadata.pop("pinned_by", None)
    metadata.pop("verify_cadence_days", None)
    
    doc.internal_metadata = metadata
    await db.commit()
    return True


async def verify_observation(
    db: AsyncSession,
    workspace_name: str,
    obs_id: str,
    created_by: str,
) -> models.AccessLogEntry:
    """Record a verification event for an observation."""
    return await create_access_log_entry(
        db=db,
        workspace_name=workspace_name,
        collection_name="",  # Will be resolved from the document
        obs_id=obs_id,
        event_type="verify",
        created_by=created_by,
    )


async def get_verify_due(
    db: AsyncSession,
    workspace_name: str,
    limit: int = 100,
) -> list[dict]:
    """List observations needing verification."""
    from src.crud.document import get_documents_with_filters
    
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.deleted_at.is_(None),
    ).limit(limit)
    result = await db.execute(stmt)
    docs = result.scalars().all()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    due_list: list[dict] = []
    
    for doc in docs:
        metadata = doc.internal_metadata or {}
        is_pinned = metadata.get("is_pinned", False)
        cadence = metadata.get("verify_cadence_days")
        
        is_due, reason = await is_verify_due(
            db, doc.id, workspace_name, is_pinned, cadence, now
        )
        if is_due:
            conf = await compute_confidence(db, doc.id, workspace_name, now)
            due_list.append({
                "obs_id": doc.id,
                "content": doc.content[:100],
                "reason": reason,
                "is_pinned": is_pinned,
                "confidence": conf,
                "last_verified": None,  # Could be fetched from log
            })
    
    return due_list


# ── Eviction ───────────────────────────────────────────────────────────────

async def evict_stale(
    db: AsyncSession,
    workspace_name: str,
    threshold: float = EVICTION_THRESHOLD,
) -> list[str]:
    """Evict unpinned observations below activation threshold.
    
    Returns list of evicted observation IDs.
    """
    from src.crud.document import get_documents_with_filters
    
    stmt = select(models.Document).where(
        models.Document.workspace_name == workspace_name,
        models.Document.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    docs = result.scalars().all()
    
    now = datetime.datetime.now(datetime.timezone.utc)
    evicted: list[str] = []
    
    for doc in docs:
        metadata = doc.internal_metadata or {}
        if metadata.get("is_pinned", False):
            continue
        
        activation = await compute_activation(db, doc.id, workspace_name, now)
        if activation < threshold:
            # Log evict event
            await create_access_log_entry(
                db, workspace_name, "", doc.id, "evict", "system"
            )
            # Soft-delete the document
            doc.deleted_at = now
            evicted.append(doc.id)
    
    await db.commit()
    return evicted
