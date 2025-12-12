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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src import models

from sklearn.cluster import DBSCAN

from src.dreamer.prescan import PatternCluster

logger = logging.getLogger(__name__)


def extract_cluster_theme(observations: list[models.Document]) -> str:
    """
    Generate a theme description for a cluster of observations.

    Uses the observation closest to the cluster centroid as representative,
    then creates a short summary. This is language-agnostic and doesn't
    rely on keyword detection.

    Args:
        observations: List of observations in the cluster

    Returns:
        A short theme description based on the most central observation
    """
    if not observations:
        return "general"

    if len(observations) == 1:
        # Single observation - use truncated content as theme
        content = observations[0].content
        return content[:50] + "..." if len(content) > 50 else content

    # Get embeddings and find centroid
    try:
        embeddings = np.array([obs.embedding for obs in observations])
        centroid = np.mean(embeddings, axis=0)

        # Find observation closest to centroid (most representative)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = int(np.argmin(distances))
        central_obs = observations[closest_idx]

        # Use truncated content of the most central observation as theme
        content = central_obs.content
        return content[:60] + "..." if len(content) > 60 else content

    except (ValueError, AttributeError) as e:
        logger.debug(f"Could not compute centroid for theme: {e}")
        # Fall back to first observation's content
        content = observations[0].content
        return content[:50] + "..." if len(content) > 50 else content


def cluster_observations(
    observations: list[models.Document],
    *,
    eps: float = 0.4,
    min_samples: int = 10,
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
        # DBSCAN with precomputed distance matrix for cosine
        # Convert to cosine distance: 1 - similarity
        # For normalized vectors: distance = 1 - dot product
        distance_matrix = 1 - np.dot(normalized, normalized.T)
        # Ensure diagonal is 0 and values are non-negative
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distance_matrix)  # pyright: ignore

    except ImportError:
        logger.warning("sklearn not available, using simple clustering fallback")
        labels = _simple_cluster(normalized, eps=eps, min_samples=min_samples)

    # Group observations by cluster label
    clusters: dict[int, list[models.Document]] = {}
    for obs, label_val in zip(observations, labels, strict=True):
        label = int(label_val)
        if label == -1:  # Noise point
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(obs)

    # Convert to PatternCluster objects
    pattern_clusters: list[PatternCluster] = []
    for _label, cluster_obs in clusters.items():
        if len(cluster_obs) >= min_samples:
            theme = extract_cluster_theme(cluster_obs)
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
    min_samples: int = 10,
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
