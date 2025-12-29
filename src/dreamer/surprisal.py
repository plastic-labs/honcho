"""
Surprisal-based observation sampling for dream processing.

Computes geometric surprisal scores for observations using tree-based
data structures, enabling targeted deductive reasoning on anomalous
or novel observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud.document import get_all_documents
from src.dreamer.trees import SurprisalTree, create_tree

logger = logging.getLogger(__name__)


@dataclass
class SurprisalScore:
    """Container for observation with surprisal score."""

    observation: models.Document
    surprisal: float
    embedding: np.ndarray


async def sample_observations_with_surprisal(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> list[SurprisalScore]:
    """
    Sample observations and compute surprisal scores.

    Workflow:
    1. Fetch observations based on SAMPLING_STRATEGY
    2. Extract embeddings from DB (already stored)
    3. Build tree structure using trees.create_tree()
    4. Compute surprisal for each observation
    5. Rank by surprisal (highest first)
    6. Filter by threshold and take top N

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier

    Returns:
        List of SurprisalScore objects, ranked by surprisal (highest first)
    """
    try:
        # 1. Fetch observations
        observations = await _fetch_observations(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        # Edge case: No observations
        if not observations:
            logger.warning(
                f"No observations found for {workspace_name}/{observer}/{observed}"
            )
            return []

        # Edge case: Too few observations for tree
        min_observations = settings.DREAM.SURPRISAL.TREE_K * 2
        if len(observations) < min_observations:
            logger.warning(
                f"Too few observations ({len(observations)} < {min_observations}), skipping surprisal computation"
            )
            return []

        # 2. Extract embeddings
        embeddings = _extract_embeddings(observations)
        if embeddings.size == 0:
            logger.error("Failed to extract embeddings")
            return []

        # 3. Build tree
        tree = _build_tree(embeddings)

        # 4. Compute surprisal
        scores = _compute_surprisal_scores(observations, embeddings, tree)

        # Edge case: Invalid surprisal values
        valid_scores = [
            s for s in scores if not np.isinf(s.surprisal) and not np.isnan(s.surprisal)
        ]
        if len(valid_scores) < len(scores):
            logger.warning(
                f"Filtered {len(scores) - len(valid_scores)} invalid surprisal scores"
            )

        # 5. Normalize surprisal scores to [0, 1] range
        normalized_scores = _normalize_scores(valid_scores)

        # 6. Rank by normalized surprisal
        normalized_scores.sort(key=lambda x: x.surprisal, reverse=True)

        # Log top 5 scores BEFORE filtering
        top_n = min(5, len(normalized_scores))
        percent = settings.DREAM.SURPRISAL.TOP_PERCENT_SURPRISAL * 100
        logger.info(f"🎯 Surprisal computation complete. Taking top {percent:.0f}%")
        logger.info(f"Top {top_n} observations by normalized surprisal score:")
        for i, score in enumerate(normalized_scores[:top_n], 1):
            content = score.observation.content
            if len(content) > 80:
                content = content[:77] + "..."
            logger.info(
                f"  #{i} [surprisal={score.surprisal:.3f}] [level={score.observation.level}] {content}"
            )

        filtered = _filter_by_percent(normalized_scores)

        logger.info(
            f"Selected: {len(filtered)}/{len(observations)} observations (top {percent:.0f}%)"
        )

        # Log summary statistics for filtered results
        if filtered:
            logger.info(
                "📊 Filtered statistics: "
                + f"min={filtered[-1].surprisal:.3f}, "
                + f"max={filtered[0].surprisal:.3f}, "
                + f"mean={sum(s.surprisal for s in filtered) / len(filtered):.3f}"
            )
        else:
            logger.info("No observations exceeded the surprisal threshold")

        return filtered

    except Exception as e:
        logger.error(f"Surprisal sampling failed: {e}", exc_info=True)
        # Return empty to allow dream to continue
        return []


async def _fetch_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> list[models.Document]:
    """
    Fetch observations based on configured sampling strategy.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name

    Returns:
        List of Document objects
    """
    strategy = settings.DREAM.SURPRISAL.SAMPLING_STRATEGY
    sample_size = settings.DREAM.SURPRISAL.SAMPLE_SIZE
    levels = settings.DREAM.SURPRISAL.INCLUDE_LEVELS

    if strategy == "recent":
        return await _fetch_recent_observations(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            limit=sample_size,
            levels=levels,
        )
    elif strategy == "random":
        return await _fetch_random_observations(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            limit=sample_size,
            levels=levels,
        )
    elif strategy == "all":
        return await _fetch_all_observations(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            limit=sample_size,
            levels=levels,
        )
    else:
        logger.warning(f"Unknown sampling strategy: {strategy}, using 'recent'")
        return await _fetch_recent_observations(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            limit=sample_size,
            levels=levels,
        )


async def _fetch_recent_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    limit: int,
    levels: list[str],
) -> list[models.Document]:
    """
    Fetch most recent observations.

    Uses existing get_all_documents() query with level filtering.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        limit: Maximum number of observations to fetch
        levels: Document levels to include

    Returns:
        List of Document objects ordered by created_at DESC
    """
    stmt = get_all_documents(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        filters={"level": levels} if levels else None,
        limit=limit,
    )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _fetch_random_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    limit: int,
    levels: list[str],
) -> list[models.Document]:
    """
    Fetch random sample of observations.

    Uses PostgreSQL's random() function for efficient random sampling.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        limit: Maximum number of observations to fetch
        levels: Document levels to include

    Returns:
        List of Document objects in random order
    """
    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(func.random())
        .limit(limit)
    )

    if levels:
        stmt = stmt.where(models.Document.level.in_(levels))

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _fetch_all_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    limit: int,
    levels: list[str],
) -> list[models.Document]:
    """
    Fetch all observations up to limit.

    Orders by created_at DESC for consistency.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        limit: Maximum number of observations to fetch
        levels: Document levels to include

    Returns:
        List of Document objects ordered by created_at DESC
    """
    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(models.Document.created_at.desc())
        .limit(limit)
    )

    if levels:
        stmt = stmt.where(models.Document.level.in_(levels))

    result = await db.execute(stmt)
    return list(result.scalars().all())


def _extract_embeddings(observations: list[models.Document]) -> np.ndarray:
    """
    Extract embeddings from observations as numpy array.

    Args:
        observations: List of Document objects with embeddings

    Returns:
        np.ndarray of shape (N, 1536) containing embeddings
    """
    if not observations:
        return np.array([])

    embeddings_list = [obs.embedding for obs in observations]
    embeddings_array = np.array(embeddings_list, dtype=np.float32)

    return embeddings_array


def _build_tree(embeddings: np.ndarray) -> SurprisalTree:
    """
    Build tree structure from embeddings.

    Args:
        embeddings: np.ndarray of shape (N, embedding_dim)

    Returns:
        SurprisalTree configured per settings
    """
    if embeddings.size == 0:
        # Return empty tree (will handle gracefully in caller)
        return create_tree(settings.DREAM.SURPRISAL.TREE_TYPE)

    tree = create_tree(
        tree_type=settings.DREAM.SURPRISAL.TREE_TYPE,
        k=settings.DREAM.SURPRISAL.TREE_K,
    )

    tree.batch_insert(embeddings)

    return tree


def _compute_surprisal_scores(
    observations: list[models.Document],
    embeddings: np.ndarray,
    tree: SurprisalTree,
) -> list[SurprisalScore]:
    """
    Compute surprisal score for each observation.

    Args:
        observations: List of Document objects
        embeddings: np.ndarray of embeddings matching observations
        tree: Built SurprisalTree

    Returns:
        List of SurprisalScore objects (unfiltered, unsorted)
    """
    scores: list[SurprisalScore] = []

    for obs, embedding in zip(observations, embeddings, strict=False):
        surprisal = tree.surprisal(embedding)

        scores.append(
            SurprisalScore(
                observation=obs,
                surprisal=surprisal,
                embedding=embedding,
            )
        )

    return scores


def _normalize_scores(scores: list[SurprisalScore]) -> list[SurprisalScore]:
    """
    Normalize surprisal scores to [0, 1] range using min-max normalization.

    Args:
        scores: List of SurprisalScore objects with raw surprisal values

    Returns:
        List of SurprisalScore objects with normalized surprisal values
    """
    if not scores:
        return []

    # Handle edge case: all scores are identical
    surprisal_values = [s.surprisal for s in scores]
    min_surprisal = min(surprisal_values)
    max_surprisal = max(surprisal_values)

    if max_surprisal == min_surprisal:
        # All scores identical - set all to 0.5 (middle of range)
        return [
            SurprisalScore(
                observation=s.observation, surprisal=0.5, embedding=s.embedding
            )
            for s in scores
        ]

    # Min-max normalization: (x - min) / (max - min)
    normalized: list[SurprisalScore] = []
    for score in scores:
        normalized_value = (score.surprisal - min_surprisal) / (
            max_surprisal - min_surprisal
        )
        normalized.append(
            SurprisalScore(
                observation=score.observation,
                surprisal=normalized_value,
                embedding=score.embedding,
            )
        )

    return normalized


def _filter_by_percent(scores: list[SurprisalScore]) -> list[SurprisalScore]:
    """
    Filter observations by top percentage.

    Assumes scores are already sorted by surprisal (highest first).

    Args:
        scores: List of SurprisalScore objects, sorted by surprisal DESC

    Returns:
        Filtered list of SurprisalScore objects (top N% by surprisal)
    """
    if not scores:
        return []

    # Take top percentage
    top_percent = settings.DREAM.SURPRISAL.TOP_PERCENT_SURPRISAL
    count = max(1, int(len(scores) * top_percent))  # At least 1 observation

    return scores[:count]
