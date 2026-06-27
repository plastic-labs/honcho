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
import math
import re
import time
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import ConfiguredModelSettings, settings
from src.crud.document import query_documents
from src.crud.graph_memory import (
    add_context_member,
    create_access_log_entry,
    create_edge,
)
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.models import Document
from src.telemetry.events.llm import CallPurpose
from src.utils.types import EdgeType

logger = logging.getLogger(__name__)

# Maximum number of times we'll attempt to promote a single observation before
# marking it as permanently failed. Each attempt is roughly one queue-item
# processing cycle; the count is persisted on the document so it survives
# restarts and re-enqueues.
MAX_PROMOTION_ATTEMPTS = 3

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

    # Clamp the input so a pathologically long observation can't blow the
    # cheap model's context window. Truncation is safe here because the
    # classifier only needs the gist; the original document content is untouched.
    if len(content) > max_input_tokens * 4:  # rough char-per-token upper bound
        content = content[: max_input_tokens * 4]

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


# Fraction of the embedding model's max token budget we allow for a single
# observation embedding. If an observation exceeds this, we chunk and average
# the chunk embeddings. Averaging is not a semantic silver bullet, but it
# avoids silently truncating oversized observations and keeps every chunk under
# the provider's token limit.
MAX_TOKENS_PER_OBSERVATION_EMBEDDING = int(settings.EMBEDDING.MAX_INPUT_TOKENS * 0.9)


def _count_tokens(text: str) -> int:
    """Best-effort token count using the configured embedding tokenizer."""
    try:
        client = embedding_client._get_client()
        return len(client.encoding.encode(text))
    except Exception:
        # If the tokenizer is unavailable, fall back to a conservative word
        # estimate. We never want a token-count failure to break promotion.
        return len(text.split())


def _chunk_intent_aware(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` near intent-aware boundaries while respecting ``max_tokens``.

    Boundaries are considered in order of preference:
      1. Paragraph breaks (blank lines)
      2. Sentence endings (``. ``, ``! ``, ``? ``)
      3. Clause boundaries (``, ``, ``; ``)
      4. Word boundaries (last resort)

    This preserves semantic continuity better than fixed-token chunking.
    """
    text = text.strip()
    if not text:
        return []

    # 1. Paragraph-level split.  Each paragraph is processed independently so a
    # multi-topic observation with clear paragraph breaks gets separate chunks.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # 2. Sentence-level split within each paragraph, merging short sentences
    # into token-bounded chunks.
    chunks: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            if not sentence.strip():
                continue
            tokens = _count_tokens(sentence)
            # A single sentence may already exceed the budget; we'll force a
            # split on clauses/words below, so don't start a fresh chunk for it.
            if current_tokens + tokens > max_tokens and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = tokens
            else:
                current.append(sentence)
                current_tokens += tokens

        if current:
            chunks.append(" ".join(current))

    # 3. Clause-level split for chunks still over budget.
    final: list[str] = []
    for chunk in chunks:
        if _count_tokens(chunk) <= max_tokens:
            final.append(chunk)
            continue
        clauses = re.split(r"(?<=[,;])\s+", chunk)
        current = []
        current_tokens = 0
        for clause in clauses:
            if not clause.strip():
                continue
            tokens = _count_tokens(clause)
            if current_tokens + tokens > max_tokens and current:
                final.append(" ".join(current))
                current = [clause]
                current_tokens = tokens
            else:
                current.append(clause)
                current_tokens += tokens
        if current:
            final.append(" ".join(current))

    # 4. Word-level split as last resort for pathological inputs.
    ultra: list[str] = []
    for chunk in final:
        if _count_tokens(chunk) <= max_tokens:
            ultra.append(chunk)
            continue
        current_words: list[str] = []
        current_tokens = 0
        for word in chunk.split():
            tokens = max(1, _count_tokens(word))
            if current_tokens + tokens > max_tokens and current_words:
                ultra.append(" ".join(current_words))
                current_words = [word]
                current_tokens = tokens
            else:
                current_words.append(word)
                current_tokens += tokens
        if current_words:
            ultra.append(" ".join(current_words))

    return ultra


# Prompt for the single-intent dense-block summary fallback.  Kept minimal so
# a cheap promotion model can serve it.  The full LLMinal L1 mechanical
# compressor lives in Fix 5 and is gated by GRAPH_MEMORY.LLMINAL_COMPRESSION.
_SUMMARY_EMBEDDING_PROMPT = """Summarize the following text in 1-2 concise sentences. Preserve the key intent, entities, and durable facts. Return only the summary, with no explanation or surrounding commentary.

{text}"""


async def _summarize_for_embedding(text: str) -> str:
    """Generate a concise summary of ``text`` suitable for embedding.

    Used as a fallback when an oversized observation cannot be split into
    semantically distinct chunks (single dense block).  The original document
    content is preserved unchanged; only the embedding vector is derived from
    the summary.
    """
    model_config = _get_promotion_model_config()
    max_tokens = settings.PROMOTION.MAX_TOKENS
    max_input_tokens = settings.PROMOTION.MAX_INPUT_TOKENS
    retry_attempts = settings.PROMOTION.MAX_OUTER_RETRIES

    prompt = _SUMMARY_EMBEDDING_PROMPT.format(text=text)
    response: HonchoLLMCallResponse[str] = await honcho_llm_call(
        model_config=model_config,
        prompt=prompt,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        enable_retry=True,
        retry_attempts=retry_attempts,
        temperature=0.0,
        telemetry=LLMTelemetryContext(
            workspace_name=None,
            call_purpose=CallPurpose.SUMMARY_SHORT.value,
            parent_category="promotion",
            observer=None,
            observed=None,
            track_name="Promotion Embedding Summary",
        ),
    )
    raw = cast(str | None, response.content)
    if raw is None or not raw.strip():
        raise RuntimeError("summary response was empty")
    return raw.strip()


async def _embed_observation_chunks(doc: Document) -> list[list[float]]:
    """Return one or more embedding vectors for ``doc``.

    If the document already has a stored embedding, that embedding is reused
    as a single chunk.  Small documents are embedded whole.  Oversized
    documents are split at intent-aware boundaries and each chunk is embedded
    independently so that a multi-intent observation can form edges to several
    semantically distinct observation clusters.

    Returns:
        An ordered list of embedding vectors, one per chunk.
    """
    if doc.embedding is not None:
        return [doc.embedding]

    content = doc.content
    if _count_tokens(content) <= MAX_TOKENS_PER_OBSERVATION_EMBEDDING:
        return [await embedding_client.embed(content)]

    chunks = _chunk_intent_aware(content, MAX_TOKENS_PER_OBSERVATION_EMBEDDING)
    if not chunks:
        raise RuntimeError("no chunks produced for oversized observation")

    # Fallback for a single dense block that survived sentence/clause/word
    # splitting without forming semantically distinct chunks.  Summarize and
    # embed the summary instead of the raw block.
    if len(chunks) == 1:
        try:
            summary = await _summarize_for_embedding(chunks[0])
            chunks = [summary]
        except Exception as exc:
            logger.warning(
                "Summary embedding fallback failed for observation %s: %s",
                doc.id,
                exc,
            )

    return await embedding_client.simple_batch_embed(chunks)


async def _get_related_observation_ids_for_chunks(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    obs_id: str,
    chunk_embeddings: list[list[float]],
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Find related observations using each chunk embedding independently.

    Candidates are merged across chunks so that an observation related to any
    chunk gets a single edge, weighted by the best (closest) cosine distance
    across all chunks.
    """
    candidates: dict[str, float] = {}

    for chunk_embedding in chunk_embeddings:
        rows = await _get_related_observation_ids(
            db,
            workspace_name,
            observer,
            observed,
            obs_id,
            obs_embedding=chunk_embedding,
            limit=limit,
        )
        for related_id, distance in rows:
            if related_id == obs_id or distance is None:
                continue
            best = candidates.get(related_id)
            if best is None or distance < best:
                candidates[related_id] = distance

    # Sort by ascending distance (highest similarity first) and cap.
    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1])
    return sorted_candidates[:limit]


# Compatibility alias: code that expects a single embedding for an observation
# can still call this name, but the promotion worker now uses per-chunk search.
async def _embed_observation(doc: Document) -> list[float]:
    """Return a single representative embedding vector for ``doc``.

    Deprecated for the promotion worker path: new code should call
    ``_embed_observation_chunks`` to preserve multi-intent signals.  This alias
    averages chunk embeddings to retain backward compatibility with callers
    that only need one vector.
    """
    chunk_embeddings = await _embed_observation_chunks(doc)
    if len(chunk_embeddings) == 1:
        return chunk_embeddings[0]

    dim = len(chunk_embeddings[0])
    sums = [0.0] * dim
    for vec in chunk_embeddings:
        for i, value in enumerate(vec):
            sums[i] += value
    mean = [value / len(chunk_embeddings) for value in sums]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0:
        return mean
    return [value / norm for value in mean]


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

    Per-observation failures are isolated: a single bad observation is marked
    as failed (after MAX_PROMOTION_ATTEMPTS) and the queue item is retired.
    Exceptions are swallowed here so a sick observation cannot abort other
    observations in the batch.
    """
    start_time = time.perf_counter()
    logger.info(
        "Processing promotion for observation %s in workspace %s",
        obs_id, workspace_name,
    )

    # Documents have no collection_name column — the collection identity is the
    # (observer, observed) peer pair. Synthesize a stable name for the edges /
    # access_log rows (which require a non-null collection_name) when the caller
    # didn't supply a real one.
    if not collection_name:
        collection_name = f"{observer}/{observed}"

    doc: Document | None = None
    try:
        async with tracked_db("promotion.fetch") as db:
            # Fetch the observation
            doc = await _get_document(db, obs_id, workspace_name)
            if doc is None:
                logger.warning("Observation %s not found, skipping promotion", obs_id)
                return

            # Increment attempt count at the start of every processing cycle so a
            # persistently sick observation eventually hits MAX_PROMOTION_ATTEMPTS
            # and is permanently skipped.
            doc.promotion_attempts += 1
            await db.commit()

            content = doc.content
            level = doc.level or "explicit"

            # Ensure we have an embedding for vector similarity search.  Oversized
            # observations are chunked at intent-aware boundaries and each chunk
            # is embedded independently, so a multi-intent observation can connect
            # to several semantically distinct observation clusters.
            chunk_embeddings = await _embed_observation_chunks(doc)
            related = await _get_related_observation_ids_for_chunks(
                db,
                workspace_name,
                observer,
                observed,
                obs_id,
                chunk_embeddings=chunk_embeddings,
                limit=20,
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

            for related_id, distance in related:
                if related_id == obs_id:
                    continue

                edge_metadata: dict[str, Any] = {}
                if distance is not None:
                    # cosine_distance = 1 - cosine_similarity, so 1 - distance
                    # is the cosine similarity between the two observations.
                    edge_metadata["weight"] = round(1.0 - distance, 4)

                try:
                    await create_edge(
                        db=db,
                        workspace_name=workspace_name,
                        collection_name=collection_name,
                        source_obs_id=obs_id,
                        target_obs_id=related_id,
                        edge_type=edge_type,
                        created_by="promotion-worker",
                        edge_metadata=edge_metadata,
                    )
                    edges_created += 1
                except Exception as e:
                    logger.debug(
                        "Edge creation skipped for %s -> %s: %s",
                        obs_id, related_id, e,
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

        async with tracked_db("promotion.mark") as db:
            doc = await _get_document(db, obs_id, workspace_name)
            if doc is not None:
                doc.promoted_at = func.now()
                await db.commit()

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Promotion complete for %s in %.0fms", obs_id, duration_ms,
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        attempt_count = doc.promotion_attempts if doc is not None else 0
        logger.error(
            "Promotion failed for observation %s (attempt %d/%d): %s",
            obs_id,
            attempt_count,
            MAX_PROMOTION_ATTEMPTS,
            error_msg,
            exc_info=True,
        )
        if doc is not None and attempt_count >= MAX_PROMOTION_ATTEMPTS:
            async with tracked_db("promotion.fail") as db:
                refreshed = await _get_document(db, obs_id, workspace_name)
                if refreshed is not None:
                    refreshed.promotion_failed = True
                    refreshed.promotion_error = error_msg[:65535]
                    await db.commit()
                    logger.warning(
                        "Observation %s marked as promotion_failed after %d attempts",
                        obs_id,
                        refreshed.promotion_attempts,
                    )
        # Swallow the exception: one sick observation must not crash the batch.
        return


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


# Cosine-distance threshold for creating promotion edges.  Two observations
# must be closer than this distance (i.e. cosine similarity > 0.7) to be
# considered related.  Keeps the graph from wiring unrelated observations.
MAX_PROMOTION_EDGE_COSINE_DISTANCE: float = 0.3


async def _get_related_observation_ids(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    obs_id: str,
    *,
    obs_embedding: list[float] | None = None,
    limit: int = 20,
) -> list[tuple[str, float | None]]:
    """Get related observation ids in the same (observer, observed) collection.

    Documents have no collection_name column; a collection is identified by the
    (observer, observed) peer pair. When ``obs_embedding`` is supplied, results
    are ranked by pgvector cosine similarity to that embedding and filtered by
    ``MAX_PROMOTION_EDGE_COSINE_DISTANCE``.

    Returns a list of ``(id, cosine_distance)`` tuples. ``distance`` is ``None``
    when no embedding is provided. Callers can use the ids after the DB session
    closes without DetachedInstanceError.
    """
    distance_expr = Document.embedding.cosine_distance(obs_embedding) if obs_embedding is not None else None

    stmt = (
        select(Document.id, distance_expr if distance_expr is not None else Document.id)
        .where(Document.workspace_name == workspace_name)
        .where(Document.observer == observer)
        .where(Document.observed == observed)
        .where(Document.embedding.isnot(None))
        .where(Document.deleted_at.is_(None))
        .where(Document.id != obs_id)
    )

    if obs_embedding is not None:
        stmt = stmt.where(
            Document.embedding.cosine_distance(obs_embedding)
            <= MAX_PROMOTION_EDGE_COSINE_DISTANCE
        ).order_by(Document.embedding.cosine_distance(obs_embedding))

    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    rows = result.all()
    if obs_embedding is not None:
        return [(row[0], float(row[1])) for row in rows]
    return [(row[0], None) for row in rows]
