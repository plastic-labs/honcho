"""
Exhaustive pre-scan infrastructure for efficient dreaming.

This module fetches ALL observations and pre-computes ALL analysis needed
for dreaming. No limits, no sampling - everything gets processed.

By moving exploration to code, specialist subagents receive complete context
and only need to reason, not search.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUpdateCandidate:
    """A potential knowledge update detected by pre-scan."""

    old_observation: models.Document
    new_observation: models.Document
    topic: str  # Extracted topic/entity that changed
    similarity: float  # Semantic similarity between observations


@dataclass
class DuplicateCandidate:
    """A potential duplicate pair detected by pre-scan."""

    doc_a: models.Document
    doc_b: models.Document
    similarity: float  # Cosine similarity between embeddings


@dataclass
class PatternCluster:
    """A cluster of related observations for induction."""

    observations: list[models.Document]
    theme: str  # Detected theme/topic
    count: int  # Number of observations in cluster


@dataclass
class DreamContext:
    """
    Complete pre-computed context for exhaustive dreaming.

    Contains ALL observations and ALL pre-computed analysis.
    Specialists process this exhaustively - no limits, no sampling.
    """

    # ALL observations by level - no limits
    explicit_observations: list[models.Document] = field(default_factory=list)
    deductive_observations: list[models.Document] = field(default_factory=list)
    inductive_observations: list[models.Document] = field(default_factory=list)

    # Peer card
    peer_card: list[str] = field(default_factory=list)

    # ALL pre-computed analysis - no limits
    knowledge_update_candidates: list[KnowledgeUpdateCandidate] = field(
        default_factory=list
    )
    duplicate_candidates: list[DuplicateCandidate] = field(default_factory=list)
    pattern_clusters: list[PatternCluster] = field(default_factory=list)

    @property
    def explicit_count(self) -> int:
        return len(self.explicit_observations)

    @property
    def deductive_count(self) -> int:
        return len(self.deductive_observations)

    @property
    def inductive_count(self) -> int:
        return len(self.inductive_observations)

    @property
    def total_observations(self) -> int:
        return self.explicit_count + self.deductive_count + self.inductive_count

    @property
    def has_potential_updates(self) -> bool:
        return len(self.knowledge_update_candidates) > 0

    @property
    def has_duplicates(self) -> bool:
        return len(self.duplicate_candidates) > 0

    @property
    def cluster_count(self) -> int:
        return len(self.pattern_clusters)

    def summary(self) -> str:
        """Generate a summary string for logging/debugging."""
        return (
            f"DreamContext: "
            f"{self.explicit_count} explicit, "
            f"{self.deductive_count} deductive, "
            f"{self.inductive_count} inductive "
            f"({self.total_observations} total), "
            f"{len(self.knowledge_update_candidates)} update candidates, "
            f"{len(self.duplicate_candidates)} duplicate candidates, "
            f"{self.cluster_count} pattern clusters"
        )


async def fetch_all_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
) -> tuple[
    list[models.Document],
    list[models.Document],
    list[models.Document],
]:
    """
    Fetch ALL observations, grouped by level. No limits.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name

    Returns:
        Tuple of (explicit, deductive, inductive) observation lists
    """
    # Fetch ALL observations - no limit
    stmt = crud.document.get_all_documents(
        workspace_name,
        observer=observer,
        observed=observed,
        limit=None,  # No limit - get everything
    )
    result = await db.execute(stmt)
    all_docs: Sequence[models.Document] = result.scalars().all()

    # Group by level
    explicit: list[models.Document] = []
    deductive: list[models.Document] = []
    inductive: list[models.Document] = []

    for doc in all_docs:
        if doc.level == "explicit":
            explicit.append(doc)
        elif doc.level == "deductive":
            deductive.append(doc)
        elif doc.level == "inductive":
            inductive.append(doc)

    logger.info(
        f"Fetched ALL observations: {len(explicit)} explicit, "
        f"{len(deductive)} deductive, {len(inductive)} inductive"
    )

    return explicit, deductive, inductive


async def prescan_for_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    *,
    duplicate_threshold: float = 0.05,  # cosine distance threshold
    cluster_min_samples: int = 3,
) -> DreamContext:
    """
    Exhaustively pre-compute everything needed for dreaming.

    Fetches ALL observations and computes ALL analysis.
    No limits, no sampling - specialists receive complete context.

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observer: Observer peer name
        observed: Observed peer name
        duplicate_threshold: Cosine distance threshold for duplicate detection
        cluster_min_samples: Minimum samples for a valid cluster

    Returns:
        DreamContext with ALL pre-computed data
    """
    # Import here to avoid circular imports
    from src.dreamer.clustering import cluster_observations
    from src.dreamer.knowledge_updates import detect_knowledge_updates

    logger.info(
        f"Exhaustive pre-scan starting: {workspace_name}/{observer}/{observed}"
    )

    # 1. Fetch ALL observations
    explicit, deductive, inductive = await fetch_all_observations(
        db, workspace_name, observer, observed
    )

    # Combine for analysis that needs all observations
    all_obs = explicit + deductive + inductive

    # 2. Fetch peer card
    peer_card = await crud.peer_card.get_peer_card(
        db, workspace_name, observer=observer, observed=observed
    )

    # 3. Detect ALL knowledge update candidates
    knowledge_update_candidates = await detect_knowledge_updates(
        all_obs, duplicate_threshold=duplicate_threshold
    )

    # 4. Detect ALL duplicate candidates
    duplicate_candidates = find_duplicate_candidates(
        all_obs, threshold=duplicate_threshold
    )

    # 5. Cluster ALL explicit observations for induction
    pattern_clusters = cluster_observations(
        explicit, min_samples=cluster_min_samples
    )

    context = DreamContext(
        explicit_observations=explicit,
        deductive_observations=deductive,
        inductive_observations=inductive,
        peer_card=peer_card or [],
        knowledge_update_candidates=knowledge_update_candidates,
        duplicate_candidates=duplicate_candidates,
        pattern_clusters=pattern_clusters,
    )

    logger.info(f"Exhaustive pre-scan complete: {context.summary()}")
    return context


def find_duplicate_candidates(
    observations: list[models.Document],
    threshold: float = 0.05,
) -> list[DuplicateCandidate]:
    """
    Find ALL potential duplicate observation pairs using embedding similarity.

    Uses O(n^2) comparison. For very large sets (1000+), this could be
    optimized with approximate nearest neighbors, but correctness > speed.

    Args:
        observations: ALL observations to check
        threshold: Maximum cosine distance to consider a duplicate

    Returns:
        ALL duplicate candidate pairs
    """
    import numpy as np

    if len(observations) < 2:
        return []

    duplicates: list[DuplicateCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Pre-compute all embeddings and norms
    embeddings = [np.array(doc.embedding) for doc in observations]
    norms = [np.linalg.norm(emb) for emb in embeddings]

    for i, doc_a in enumerate(observations):
        if norms[i] == 0:
            continue

        for j in range(i + 1, len(observations)):
            doc_b = observations[j]

            # Only compare same-level observations
            if doc_a.level != doc_b.level:
                continue

            if norms[j] == 0:
                continue

            # Skip already seen pairs
            id_a, id_b = (doc_a.id, doc_b.id) if doc_a.id < doc_b.id else (doc_b.id, doc_a.id)
            pair_key = (id_a, id_b)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Compute cosine distance
            cosine_sim = np.dot(embeddings[i], embeddings[j]) / (norms[i] * norms[j])
            cosine_dist = 1 - cosine_sim

            if cosine_dist < threshold:
                duplicates.append(
                    DuplicateCandidate(
                        doc_a=doc_a,
                        doc_b=doc_b,
                        similarity=float(cosine_sim),
                    )
                )

    # Sort by similarity (highest first = most likely duplicates)
    duplicates.sort(key=lambda x: x.similarity, reverse=True)

    logger.info(f"Found {len(duplicates)} duplicate candidates from {len(observations)} observations")
    return duplicates
