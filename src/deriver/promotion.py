"""Promotion worker — runs after observations are created.

The promotion test determines if a fact is non-obvious AND durable.
If promoted, edges are created to related observations and the
observation is assigned to the active context.

This runs as a background task (sibling to the Deriver), not inline.

v1 used a heuristic-based promotion test (keyword matching).
v2 (kanban t_3dec782c) upgrades to LLM-based classification: a cheap
model returns a single YES/NO token. The heuristic is retained as a
fallback for when the LLM call fails after all retries — per spec §7.4a,
"on persistent failure, promote conservatively (safe but noisy) rather
than dropping."
"""

from __future__ import annotations

import logging
import re
import time
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import ConfiguredModelSettings, settings
from src.crud.graph_memory import (
    add_context_member,
    create_access_log_entry,
    create_edge,
)
from src.dependencies import tracked_db
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.models import Document
from src.telemetry.events.llm import CallPurpose
from src.utils.types import EdgeType

logger = logging.getLogger(__name__)

# ── Document level → edge type mapping (spec §7.1) ─────────────────────────

LEVEL_TO_EDGE_TYPE: dict[str, EdgeType] = {
    "explicit": "related",
    "deductive": "refines",
    "inductive": "composes-with",
    "contradiction": "contradicts",
}

# ── LLM-based promotion test (v2) ──────────────────────────────────────────

# Single-token YES/NO classification prompt (spec §7.4a). The model is
# expected to return exactly "YES" or "NO" (case-insensitive match below).
# Kept deliberately small so a cheap classifier model is sufficient.
PROMOTION_TEST_PROMPT = """You are a memory-promotion classifier for a long-running AI agent system.

You will be given a single extracted observation (a "fact") that the system derived from a conversation. Your job is to decide whether this fact should be PROMOTED into the agent's durable L2 memory, where it will be available to future sessions and connected to related observations.

A fact should be promoted (answer YES) if it is BOTH:
1. Non-obvious: NOT trivially derivable from reading the codebase, repo, logs, or standard documentation alone. Things like `import os`, function definitions, file paths, or `print(...)` statements are obvious-from-code and should NOT be promoted.
2. Durable: will still be true and useful in a future session. Ephemeral state ("currently we are doing X", "today's plan is Y", "let me check Z", "maybe we should W") is NOT durable.

A fact should NOT be promoted (answer NO) if it is:
- Obvious from code (imports, def/class signatures, return statements, print/TODO/FIXME)
- Temporary or hedged ("today", "right now", "for now", "maybe", "perhaps", "might be")
- A verbatim transcription of a tool output or error message with no insight attached
- Very short / content-free (< 20 characters of substance)

Answer with EXACTLY one token, either YES or NO. Do not add any explanation, punctuation, or other text.

<observation>
{content}
</observation>

Answer:"""


def _promotion_test_prompt(content: str) -> str:
    """Build the promotion-test prompt for a single observation."""
    return PROMOTION_TEST_PROMPT.format(content=content)


def _parse_promotion_response(raw: str | None) -> bool | None:
    """Parse the model's single-token YES/NO response.

    Returns True (promote), False (don't promote), or None if the response
    could not be classified (caller decides the fallback).
    """
    if raw is None:
        return None
    if not raw.strip():
        return None
    # Take the first non-empty line, strip whitespace, normalize to upper case,
    # and strip trailing punctuation that some providers append ("YES.", "No,").
    # Then require an exact match against one of the four accepted tokens —
    # this rejects lookalikes like "nope" / "yep" / "yes, but..." rather than
    # letting `startswith` silently classify them (spec §7.4a: unparseable
    # responses fall back to the heuristic, they are not silently NO).
    token = raw.strip().splitlines()[0].strip().upper().rstrip(".!?,:;-")
    if token in ("YES", "Y"):
        return True
    if token in ("NO", "N"):
        return False
    return None


async def _llm_promotion_test(
    content: str,
    *,
    workspace_name: str | None = None,
    observer: str | None = None,
    observed: str | None = None,
) -> bool:
    """LLM-based promotion test (v2).

    Asks a cheap model to return a single YES/NO token classifying whether
    `content` is non-obvious AND durable. On any failure (LLM error after
    retries exhausted, unparseable response), falls back to the v1
    heuristic test — per spec §7.4a, "promote conservatively (safe but
    noisy) rather than dropping."

    Args:
        content: The observation content to classify.
        workspace_name: Optional, for telemetry attribution.
        observer / observed: Optional peer context for telemetry.

    Returns:
        True if the fact should be promoted, False otherwise.
    """
    model_config = _get_promotion_model_config()
    max_tokens = settings.PROMOTION.MAX_TOKENS
    max_input_tokens = settings.PROMOTION.MAX_INPUT_TOKENS
    retry_attempts = settings.PROMOTION.MAX_OUTER_RETRIES

    prompt = _promotion_test_prompt(content)

    try:
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            model_config=model_config,
            prompt=prompt,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            enable_retry=True,
            retry_attempts=retry_attempts,
            temperature=0.0,
            telemetry=LLMTelemetryContext(
                workspace_name=workspace_name,
                call_purpose=CallPurpose.PROMOTION_TEST.value,
                parent_category="promotion",
                observer=observer,
                observed=observed,
                track_name="Promotion Test",
            ),
        )
    except Exception as exc:
        # Spec §7.4a: persistent failure → fall back to the heuristic rather
        # than dropping the observation. Logged at WARNING so operators see
        # the LLM degradation but the pipeline keeps moving.
        logger.warning(
            "Promotion-test LLM call failed after %d retries; falling back "
            "to heuristic test. Error: %s",
            retry_attempts,
            exc,
        )
        return _heuristic_promotion_test(content)

    raw = cast(str | None, response.content)
    parsed = _parse_promotion_response(raw)
    if parsed is None:
        logger.warning(
            "Promotion-test returned an unparseable response (%r); falling "
            "back to heuristic test.",
            raw,
        )
        return _heuristic_promotion_test(content)

    return parsed


def _get_promotion_model_config() -> ConfiguredModelSettings:
    """Return the promotion-worker model config from settings."""
    return settings.PROMOTION.MODEL_CONFIG


# ── Heuristic promotion test (v1, retained as fallback) ────────────────────

# Patterns that indicate a fact is obvious-from-code (should NOT promote)
OBVIOUS_PATTERNS = [
    r"\bimport\s+\w+",
    r"\bdef\s+\w+",
    r"\bclass\s+\w+",
    r"\breturn\s+\w+",
    r"\bprint\s*\(",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bHACK\b",
    r"\bXXX\b",
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
    """Heuristic promotion test (v1, retained as the LLM fallback).

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
    
    # Rule 1: Obvious-from-code patterns (case-insensitive)
    for pattern in OBVIOUS_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return False
    
    # Rule 2: Temporary patterns (case-insensitive)
    for pattern in TEMPORARY_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return False
    
    # Rule 3: Durable patterns (case-insensitive) → promote
    for pattern in DURABLE_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
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

    1. Run promotion test (LLM-based for v2, heuristic fallback on failure)
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

    # Step 1: Run promotion test. Skip the LLM call entirely if the worker
    # is disabled — this makes PROMOTION.ENABLED=False a real off-switch
    # (no model calls, no spend) rather than just a flag that's ignored.
    if settings.PROMOTION.ENABLED:
        is_promoted = await _llm_promotion_test(
            content,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
    else:
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