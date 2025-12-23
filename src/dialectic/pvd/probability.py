"""
Probability computation using tree-based surprisal estimation.

Computes p(v|e,a) and p(v|a) for PVD scoring using on-demand tree building.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.dreamer.trees import SurprisalTree, create_tree

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityResult:
    """Probability computation result for a single observation."""

    document_id: str
    p_entity: float  # p(v|e,a) - probability in session context (entity)
    p_anchor: float  # p(v|a) - probability in global context (anchor)
    surprisal_entity: float  # Raw surprisal from session tree
    surprisal_anchor: float  # Raw surprisal from global tree


async def compute_probabilities(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str | None,
    candidate_docs: list[models.Document],
    tree_type: str = "kdtree",
    k: int = 5,
) -> dict[str, ProbabilityResult]:
    """
    Compute p(v|e,a) and p(v|a) for candidate observations.

    Entity (e) = session, Anchor (a) = peer

    Strategy:
    1. If session_name provided:
       - Build session tree from observations in this session
       - Build global tree from all observations for this observer/observed pair
    2. If no session_name:
       - Only build global tree
       - Set p(v|e,a) = p(v|a) (no session distinction)

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier (entity)
        candidate_docs: Documents to compute probabilities for
        tree_type: Type of tree to build (kdtree, rptree, etc.)
        k: k parameter for kNN-based trees

    Returns:
        Dict mapping document_id -> ProbabilityResult
    """
    if not candidate_docs:
        return {}

    try:
        results = {}
        candidate_embeddings = {doc.id: np.array(doc.embedding) for doc in candidate_docs}

        # Build global tree (anchor = peer)
        logger.debug(
            f"Building global tree for {workspace_name}/{observer}/{observed}"
        )
        global_tree, global_embeddings, global_doc_ids = await _build_global_tree(
            db=db,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            tree_type=tree_type,
            k=k,
        )

        # Build session tree if session_name provided (entity = session)
        session_tree = None
        if session_name:
            logger.debug(
                f"Building session tree for {workspace_name}/{observer}/{observed}/{session_name}"
            )
            session_tree, session_embeddings, session_doc_ids = await _build_session_tree(
                db=db,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                session_name=session_name,
                tree_type=tree_type,
                k=k,
            )

        # Compute probabilities for each candidate
        for doc_id, embedding in candidate_embeddings.items():
            # Compute p(v|a) from global tree
            surprisal_anchor = global_tree.surprisal(embedding)
            p_anchor = _surprisal_to_probability(surprisal_anchor)

            # Compute p(v|e,a) from session tree if available
            if session_tree is not None:
                surprisal_entity = session_tree.surprisal(embedding)
                p_entity = _surprisal_to_probability(surprisal_entity)
            else:
                # No session distinction, use global probability
                surprisal_entity = surprisal_anchor
                p_entity = p_anchor

            results[doc_id] = ProbabilityResult(
                document_id=doc_id,
                p_entity=p_entity,
                p_anchor=p_anchor,
                surprisal_entity=surprisal_entity,
                surprisal_anchor=surprisal_anchor,
            )

        logger.debug(
            f"Computed probabilities for {len(results)} documents "
            f"(session_tree: {session_tree is not None}, global_tree: True)"
        )

        return results

    except Exception as e:
        logger.error(f"Failed to compute probabilities: {e}", exc_info=True)
        # Return uniform probabilities as fallback
        return {
            doc.id: ProbabilityResult(
                document_id=doc.id,
                p_entity=0.5,
                p_anchor=0.5,
                surprisal_entity=0.0,
                surprisal_anchor=0.0,
            )
            for doc in candidate_docs
        }


async def _build_session_tree(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    session_name: str,
    tree_type: str,
    k: int,
) -> tuple[SurprisalTree, np.ndarray, list[str]]:
    """
    Build tree from session-specific observations.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        session_name: Session identifier
        tree_type: Type of tree to build
        k: k parameter for kNN-based trees

    Returns:
        Tuple of (tree, embeddings_array, document_ids)
    """
    # Fetch session observations
    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
            models.Document.session_name == session_name,
        )
        .order_by(models.Document.created_at.desc())
    )

    result = await db.execute(stmt)
    documents = list(result.scalars().all())

    if not documents:
        logger.warning(
            f"No session observations found for {workspace_name}/{observer}/{observed}/{session_name}"
        )
        # Return empty tree
        tree = create_tree(tree_type=tree_type, k=k)
        return tree, np.array([]), []

    # Extract embeddings
    embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
    doc_ids = [doc.id for doc in documents]

    # Build tree
    tree = create_tree(tree_type=tree_type, k=k)
    tree.batch_insert(embeddings)

    logger.debug(
        f"Built session tree with {len(documents)} observations "
        f"(type: {tree_type}, k: {k})"
    )

    return tree, embeddings, doc_ids


async def _build_global_tree(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    tree_type: str,
    k: int,
    max_observations: int = 10000,
) -> tuple[SurprisalTree, np.ndarray, list[str]]:
    """
    Build tree from all observations for this observer/observed pair.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        tree_type: Type of tree to build
        k: k parameter for kNN-based trees
        max_observations: Maximum number of observations to include (sample if exceeded)

    Returns:
        Tuple of (tree, embeddings_array, document_ids)
    """
    # Fetch all observations
    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.observer == observer,
            models.Document.observed == observed,
        )
        .order_by(models.Document.created_at.desc())
        .limit(max_observations)
    )

    result = await db.execute(stmt)
    documents = list(result.scalars().all())

    if not documents:
        logger.warning(
            f"No observations found for {workspace_name}/{observer}/{observed}"
        )
        # Return empty tree
        tree = create_tree(tree_type=tree_type, k=k)
        return tree, np.array([]), []

    # Extract embeddings
    embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
    doc_ids = [doc.id for doc in documents]

    # Build tree
    tree = create_tree(tree_type=tree_type, k=k)
    tree.batch_insert(embeddings)

    logger.debug(
        f"Built global tree with {len(documents)} observations "
        f"(type: {tree_type}, k: {k})"
    )

    return tree, embeddings, doc_ids


def _surprisal_to_probability(surprisal: float) -> float:
    """
    Convert tree surprisal to probability.

    Surprisal = -log P(v), so P(v) = exp(-surprisal)

    We clamp the result to a reasonable range to avoid numerical issues.

    Args:
        surprisal: Surprisal value from tree

    Returns:
        Probability in range [0, 1]
    """
    # Handle edge cases
    if np.isnan(surprisal) or np.isinf(surprisal):
        return 0.5  # Neutral probability

    # Convert: P(v) = exp(-surprisal)
    # Clamp surprisal to reasonable range to avoid overflow/underflow
    surprisal = np.clip(surprisal, -10, 10)
    probability = np.exp(-surprisal)

    # Clamp to [0, 1] range
    probability = np.clip(probability, 1e-10, 1.0)

    return float(probability)
