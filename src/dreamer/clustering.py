"""
Pattern clustering for inductive reasoning.

This module clusters observations by semantic similarity to identify
patterns that might be worth generalizing into inductive observations.

Uses DBSCAN for density-based clustering which:
- Doesn't require specifying number of clusters
- Can identify noise points (observations that don't fit patterns)
- Works well with high-dimensional embeddings
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src import models

from src.dreamer.prescan import PatternCluster

logger = logging.getLogger(__name__)


def extract_theme_keywords(observations: list[models.Document], top_k: int = 3) -> str:
    """
    Extract theme keywords from a cluster of observations.

    Uses simple term frequency to find the most common meaningful words.

    Args:
        observations: List of observations in the cluster
        top_k: Number of keywords to extract

    Returns:
        Space-separated keywords representing the theme
    """
    # Common stop words to filter out
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "any",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "as",
        "because",
        "while",
        "although",
        "if",
        "unless",
        "until",
    }

    # Collect all words from observations
    word_counter: Counter[str] = Counter()
    for obs in observations:
        # Simple tokenization: lowercase and split on non-alphanumeric
        import re

        words = re.findall(r"\b[a-z]+\b", obs.content.lower())
        # Filter stop words and short words
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        word_counter.update(meaningful)

    # Get top keywords
    top_words = [word for word, _ in word_counter.most_common(top_k)]

    return " ".join(top_words) if top_words else "general"


def cluster_observations(
    observations: list[models.Document],
    *,
    eps: float = 0.3,
    min_samples: int = 3,
) -> list[PatternCluster]:
    """
    Cluster observations by semantic similarity for inductive reasoning.

    Uses DBSCAN to find dense clusters of related observations
    that might represent patterns worth generalizing.

    Args:
        observations: List of observations to cluster
        eps: Maximum distance between samples for neighborhood (cosine distance)
        min_samples: Minimum samples to form a cluster

    Returns:
        List of pattern clusters, sorted by size (largest first)
    """
    if len(observations) < min_samples:
        logger.debug(
            f"Not enough observations for clustering: {len(observations)} < {min_samples}"
        )
        return []

    # Get embeddings as numpy array
    try:
        embeddings = np.array([obs.embedding for obs in observations])
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to extract embeddings: {e}")
        return []

    if embeddings.shape[0] < min_samples:
        return []

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Use sklearn DBSCAN if available, otherwise fall back to simple implementation
    try:
        from sklearn.cluster import DBSCAN

        # DBSCAN with precomputed distance matrix for cosine
        # Convert to cosine distance: 1 - similarity
        # For normalized vectors: distance = 1 - dot product
        distance_matrix = 1 - np.dot(normalized, normalized.T)
        # Ensure diagonal is 0 and values are non-negative
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distance_matrix)

    except ImportError:
        logger.warning("sklearn not available, using simple clustering fallback")
        labels = _simple_cluster(normalized, eps=eps, min_samples=min_samples)

    # Group observations by cluster label
    clusters: dict[int, list[models.Document]] = {}
    for obs, label_val in zip(observations, labels):
        label = int(label_val)
        if label == -1:  # Noise point
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(obs)

    # Convert to PatternCluster objects
    pattern_clusters: list[PatternCluster] = []
    for label, cluster_obs in clusters.items():
        if len(cluster_obs) >= min_samples:
            theme = extract_theme_keywords(cluster_obs)
            pattern_clusters.append(
                PatternCluster(
                    observations=cluster_obs,
                    theme=theme,
                    count=len(cluster_obs),
                )
            )

    # Sort by size (largest clusters first)
    pattern_clusters.sort(key=lambda x: x.count, reverse=True)

    logger.info(
        f"Clustered {len(observations)} observations into {len(pattern_clusters)} pattern clusters"
    )
    return pattern_clusters


def _simple_cluster(
    normalized: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Simple fallback clustering when sklearn is not available.

    Uses a greedy approach to form clusters based on cosine similarity.

    Args:
        normalized: Normalized embedding vectors
        eps: Maximum cosine distance for neighbors
        min_samples: Minimum samples for a cluster

    Returns:
        Array of cluster labels (-1 for noise)
    """
    n = normalized.shape[0]
    labels = np.full(n, -1)
    current_label = 0

    # Compute similarity matrix
    similarity = np.dot(normalized, normalized.T)
    distance = 1 - similarity

    visited: set[int] = set()

    for i in range(n):
        if i in visited:
            continue

        # Find neighbors within eps distance
        neighbors = set(np.where(distance[i] <= eps)[0])
        neighbors.discard(i)

        if len(neighbors) + 1 < min_samples:
            continue

        # Start a new cluster
        cluster = {i}
        visited.add(i)
        labels[i] = current_label

        # Expand cluster
        to_process = list(neighbors - visited)
        while to_process:
            j = to_process.pop(0)
            if j in visited:
                continue

            visited.add(j)
            labels[j] = current_label
            cluster.add(j)

            # Find neighbors of j
            j_neighbors = set(np.where(distance[j] <= eps)[0])
            j_neighbors.discard(j)

            if len(j_neighbors) + 1 >= min_samples:
                new_neighbors = j_neighbors - visited
                to_process.extend(new_neighbors)

        current_label += 1

    return labels
