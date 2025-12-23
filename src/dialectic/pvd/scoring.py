"""
PVD scoring formula implementation.

Implements the scoring function:
score(v; q) = α*sim(q,v) + β*log p(v|e,a) - γ*log p(v|a)
"""

import logging
from dataclasses import dataclass

import numpy as np

from src import models
from src.dialectic.pvd.parameters import PVDParameters
from src.dialectic.pvd.probability import ProbabilityResult

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    """Document with PVD score breakdown."""

    document: models.Document
    pvd_score: float
    semantic_score: float
    entity_log_prob: float
    anchor_log_prob: float
    p_entity: float
    p_anchor: float


def compute_pvd_scores(
    candidate_docs: list[models.Document],
    query_embedding: np.ndarray,
    probabilities: dict[str, ProbabilityResult],
    parameters: PVDParameters,
) -> list[ScoredDocument]:
    """
    Compute PVD scores for all candidates.

    Formula: score(v; q) = α*sim(q,v) + β*log p(v|e,a) - γ*log p(v|a)

    Where:
    - sim(q,v) = 1 - cosine_distance(query_embedding, doc_embedding)
    - p(v|e,a) = entity-conditioned probability (from session tree)
    - p(v|a) = anchor-conditioned probability (from global tree)
    - α, β, γ = weights from PVDParameters

    Args:
        candidate_docs: Documents to score
        query_embedding: Query vector
        probabilities: Probability results for each document
        parameters: PVD parameters (alpha, beta, gamma)

    Returns:
        List of ScoredDocument objects with score breakdown
    """
    scored_docs = []

    for doc in candidate_docs:
        # Get probability result for this document
        prob_result = probabilities.get(
            doc.id,
            ProbabilityResult(
                document_id=doc.id,
                p_entity=0.5,
                p_anchor=0.5,
                surprisal_entity=0.0,
                surprisal_anchor=0.0,
            ),
        )

        # Compute semantic similarity score (normalized to [0, 1])
        semantic_score = _compute_semantic_similarity(query_embedding, doc.embedding)

        # Compute log probabilities (with numerical stability)
        entity_log_prob = np.log(max(prob_result.p_entity, 1e-10))
        anchor_log_prob = np.log(max(prob_result.p_anchor, 1e-10))

        # Compute combined PVD score
        pvd_score = (
            parameters.alpha * semantic_score
            + parameters.beta * entity_log_prob
            - parameters.gamma * anchor_log_prob
        )

        scored_docs.append(
            ScoredDocument(
                document=doc,
                pvd_score=pvd_score,
                semantic_score=semantic_score,
                entity_log_prob=entity_log_prob,
                anchor_log_prob=anchor_log_prob,
                p_entity=prob_result.p_entity,
                p_anchor=prob_result.p_anchor,
            )
        )

    logger.debug(
        f"Computed PVD scores for {len(scored_docs)} documents "
        f"(α={parameters.alpha:.2f}, β={parameters.beta:.2f}, γ={parameters.gamma:.2f})"
    )

    return scored_docs


def rerank_by_pvd(
    scored_docs: list[ScoredDocument], top_k: int
) -> list[models.Document]:
    """
    Sort by PVD score (descending) and return top-k documents.

    Args:
        scored_docs: Documents with PVD scores
        top_k: Number of documents to return

    Returns:
        Top-k documents sorted by PVD score
    """
    # Sort by PVD score (highest first)
    scored_docs.sort(key=lambda x: x.pvd_score, reverse=True)

    # Extract top-k documents
    top_docs = [scored.document for scored in scored_docs[:top_k]]

    if scored_docs:
        logger.debug(
            f"Reranked {len(scored_docs)} documents, returning top {top_k}. "
            f"Score range: [{scored_docs[-1].pvd_score:.3f}, {scored_docs[0].pvd_score:.3f}]"
        )

    return top_docs


def _compute_semantic_similarity(
    query_embedding: np.ndarray, doc_embedding: np.ndarray | list[float]
) -> float:
    """
    Compute semantic similarity between query and document.

    Uses cosine similarity, normalized to [0, 1] range:
    similarity = 1 - cosine_distance

    Args:
        query_embedding: Query vector
        doc_embedding: Document vector

    Returns:
        Similarity score in range [0, 1]
    """
    # Convert doc_embedding to numpy array if it's a list
    if isinstance(doc_embedding, list):
        doc_embedding = np.array(doc_embedding)

    # Compute cosine distance
    cosine_distance = _cosine_distance(query_embedding, doc_embedding)

    # Convert to similarity (0 = orthogonal, 1 = identical)
    similarity = 1.0 - cosine_distance

    # Clamp to [0, 1] range for numerical stability
    similarity = np.clip(similarity, 0.0, 1.0)

    return float(similarity)


def _cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    cosine_distance = 1 - cosine_similarity
                    = 1 - (dot(a, b) / (norm(a) * norm(b)))

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine distance in range [0, 2]
    """
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Orthogonal

    # Compute cosine similarity
    cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)

    # Convert to distance
    cosine_distance = 1.0 - cosine_similarity

    # Clamp to [0, 2] range for numerical stability
    cosine_distance = np.clip(cosine_distance, 0.0, 2.0)

    return float(cosine_distance)
