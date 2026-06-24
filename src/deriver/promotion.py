"""Promotion worker — runs after observations are created.

The promotion test determines if a fact is non-obvious AND durable.
If promoted, edges are created to related observations and the
observation is assigned to the active context.

This runs as a background task (sibling to the Deriver), not inline.

V1 uses a heuristic-based promotion test (keyword matching).
V2 will upgrade to LLM-based classification.
"""

from __future__ import annotations

import logging
import re
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.graph_memory import (
    add_context_member,
    create_access_log_entry,
    create_edge,
)
from src.dependencies import tracked_db
from src.models import Document
from src.utils.types import EdgeType

logger = logging.getLogger(__name__)

# ── Document level → edge type mapping (spec §7.1) ─────────────────────────

LEVEL_TO_EDGE_TYPE: dict[str, EdgeType] = {
    "explicit": "related",
    "deductive": "refines",
    "inductive": "composes-with",
    "contradiction": "contradicts",
}

# ── Heuristic promotion test (v1) ──────────────────────────────────────────

# Patterns that indicate a fact is obvious-from-code (should NOT promote)
OBVIOUS_PATTERNS = [
    r"\bimport\s+\w+",
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\breturn\s+\w+",
    r"\bprint\s*\(",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"^let me check",
    r"^i'll look",
    r"^one moment",
    r"^hang on",
    r"^not sure",
    r"^i don't know",
    r"^i'm not sure",
    r"^let me think",
    r"^give me a sec",
]

# Patterns that indicate a fact is durable (should promote)
DURABLE_PATTERNS = [
    r"\bdecided\b",
    r"\bagreed\b",
    r"\bconcluded\b",
    r"\bdetermined\b",
    r"\bestablished\b",
    r"\bconfirmed\b",
    r"\bthe system uses\b",
    r"\bthe architecture\b",
    r"\bour approach\b",
    r"\ba key insight\b",
    r"\bwe should\b",
    r"\bwe decided\b",
    r"\bafter testing\b",
    r"\bthe reason\b",
    r"\bbecause\b",
    r"\bis important because\b",
]

# Patterns that indicate a fact is temporary (should NOT promote)
TEMPORARY_PATTERNS = [
    r"^today",
    r"^this week",
    r"^right now",
    r"^currently",
    r"^for now",
    r"^temporary",
    r"^maybe",
    r"^perhaps",
    r"^could be",
    r"^might be",
]


def _heuristic_promotion_test(content: str) -> bool:
    """Heuristic promotion test (v1).
    
    TODO: Upgrade to LLM-based classification (kanban: t_3dec782c).
    The prompt template PROMOTION_TEST_PROMPT is ready for use.
    
    Returns True if the fact should be promoted (non-obvious AND durable).
    
    Rules:
    1. If content matches an OBVIOUS pattern → NOT promoted
    2. If content matches a TEMPORARY pattern → NOT promoted
    3. If content matches a DURABLE pattern → promoted
    4. If content is very short (< 20 chars) → NOT promoted
    5. Otherwise → promoted (conservative default)
    """
    content_lower = content.lower().strip()
    
    # Rule 4: Very short facts are unlikely to be durable
    if len(content_lower) < 20:
        return False
    
    # Rule 1: Obvious-from-code patterns
    for pattern in OBVIOUS_PATTERNS:
        if re.search(pattern, content_lower):
            return False
    
    # Rule 2: Temporary patterns
    for pattern in TEMPORARY_PATTERNS:
        if re.search(pattern, content_lower):
            return False
    
    # Rule 3: Durable patterns → promote
    for pattern in DURABLE_PATTERNS:
        if re.search(pattern, content_lower):
            return True
    
    # Rule 5: Conservative default
    return True


# ── Promotion worker ───────────────────────────────────────────────────────

async def process_promotion(
    workspace_name: str,
    collection_name: str,
    obs_id: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
) -> None:
    """Run the promotion pipeline for a single observation.
    
    1. Run promotion test (heuristic for v1)
    2. If promoted: create edges to related observations
    3. If promoted: assign to active context
    4. Log promote event in access log
    """
    start_time = time.perf_counter()
    logger.info(
        "Processing promotion for observation %s in workspace %s",
        obs_id, workspace_name,
    )
    
    async with tracked_db("promotion.fetch") as db:
        # Fetch the observation
        doc = await _get_document(db, obs_id, workspace_name)
        if doc is None:
            logger.warning("Observation %s not found, skipping promotion", obs_id)
            return
        
        content = doc.content
        level = doc.level or "explicit"
        
        # Fetch related observations in the same collection
        related_docs = await _get_related_documents(
            db, workspace_name, collection_name, obs_id, limit=20
        )
    
    # Step 1: Run promotion test
    is_promoted = _heuristic_promotion_test(content)
    
    if not is_promoted:
        logger.debug("Observation %s did not pass promotion test", obs_id)
        return
    
    logger.info("Observation %s promoted to L2", obs_id)
    
    # Step 2: Create edges to related observations
    async with tracked_db("promotion.edges") as db:
        edge_type = LEVEL_TO_EDGE_TYPE.get(level, "related")
        edges_created = 0
        
        for related in related_docs:
            if related.id == obs_id:
                continue
            
            try:
                await create_edge(
                    db=db,
                    workspace_name=workspace_name,
                    collection_name=collection_name,
                    source_obs_id=obs_id,
                    target_obs_id=related.id,
                    edge_type=edge_type,
                    created_by="promotion-worker",
                )
                edges_created += 1
            except Exception as e:
                logger.debug(
                    "Edge creation skipped for %s -> %s: %s",
                    obs_id, related.id, e,
                )
        
        logger.info("Created %d edges for observation %s", edges_created, obs_id)
    
    # Step 3: Assign to active context (if session has one)
    if session_name:
        async with tracked_db("promotion.context") as db:
            try:
                from src.cache.client import cache as _cache
                key = f"active_context:{workspace_name}:{session_name}"
                context_name = await _cache.get(key)
                
                if context_name:
                    await add_context_member(
                        db=db,
                        workspace_name=workspace_name,
                        context_name=context_name,
                        obs_id=obs_id,
                        added_by="promotion-worker",
                    )
                    logger.info(
                        "Assigned observation %s to context %s",
                        obs_id, context_name,
                    )
            except Exception as e:
                logger.debug("Context assignment skipped: %s", e)
    
    # Step 4: Log promote event
    async with tracked_db("promotion.log") as db:
        await create_access_log_entry(
            db=db,
            workspace_name=workspace_name,
            collection_name=collection_name,
            obs_id=obs_id,
            event_type="promote",
            created_by="promotion-worker",
            session_id=session_name,
        )
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Promotion complete for %s in %.0fms", obs_id, duration_ms,
    )


async def _get_document(
    db: AsyncSession,
    obs_id: str,
    workspace_name: str,
) -> Document | None:
    """Get a document by ID."""
    result = await db.execute(
        select(Document).where(
            Document.id == obs_id,
            Document.workspace_name == workspace_name,
            Document.deleted_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def _get_related_documents(
    db: AsyncSession,
    workspace_name: str,
    collection_name: str,
    obs_id: str,
    limit: int = 20,
) -> list[Document]:
    """Get related documents in the same collection."""
    result = await db.execute(
        select(Document).where(
            Document.workspace_name == workspace_name,
            Document.collection_name == collection_name,
            Document.deleted_at.is_(None),
            Document.id != obs_id,
        ).limit(limit)
    )
    return list(result.scalars().all())
