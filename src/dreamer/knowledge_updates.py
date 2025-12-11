"""
Knowledge update detection for the dreamer system.

This module detects when facts have changed over time by analyzing
observations for temporal conflicts - same topic, different values,
different timestamps.

Uses a hybrid approach:
1. Embedding similarity to group related observations
2. Keyword extraction to identify changeable entities
3. Returns candidates for LLM verification
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src import models
    from src.dreamer.prescan import KnowledgeUpdateCandidate

logger = logging.getLogger(__name__)

# Patterns for extractable entities that commonly change
DATE_PATTERNS = [
    # Month Day, Year or Month Day
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    # MM/DD/YYYY or MM-DD-YYYY
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    # Day of week
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
]

NUMBER_PATTERNS = [
    # Numbers with units
    r"\b\d+(?:\.\d+)?(?:\s*(?:dollars?|\$|%|percent|years?|months?|weeks?|days?|hours?|minutes?|people|items?|times?))\b",
    # Plain numbers in context
    r"\b(?:costs?|price|amount|total|count|number)\s+(?:is|was|of)?\s*\$?\d+(?:\.\d+)?\b",
]

# Keywords indicating temporal/changeable information
DEADLINE_KEYWORDS = [
    "deadline",
    "due",
    "by",
    "until",
    "before",
    "scheduled",
    "meeting",
    "appointment",
    "event",
    "plan",
    "plans",
]

UPDATE_KEYWORDS = [
    "changed",
    "updated",
    "rescheduled",
    "moved",
    "now",
    "actually",
    "instead",
    "new",
    "different",
    "revised",
    "postponed",
    "cancelled",
    "canceled",
]

# Topics that commonly have updates
CHANGEABLE_TOPICS = [
    "deadline",
    "date",
    "time",
    "price",
    "cost",
    "location",
    "address",
    "phone",
    "email",
    "status",
    "plan",
    "schedule",
    "meeting",
    "appointment",
]


def extract_entities(text: str) -> dict[str, Any]:
    """
    Extract dates, numbers, and key entities from observation text.

    Args:
        text: The observation content

    Returns:
        Dictionary with extracted entity types
    """
    text_lower = text.lower()

    # Extract dates
    dates: list[str] = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))

    # Extract numbers with context
    numbers: list[str] = []
    for pattern in NUMBER_PATTERNS:
        numbers.extend(re.findall(pattern, text, re.IGNORECASE))

    return {
        "dates": dates,
        "numbers": numbers,
        "has_deadline": any(kw in text_lower for kw in DEADLINE_KEYWORDS),
        "has_update_language": any(kw in text_lower for kw in UPDATE_KEYWORDS),
        "changeable_topics": [
            topic for topic in CHANGEABLE_TOPICS if topic in text_lower
        ],
    }


def compute_topic_signature(text: str) -> str:
    """
    Extract a topic signature for grouping related observations.

    This creates a simplified representation of what the observation
    is about, stripping out specific values to find topic matches.

    Args:
        text: The observation content

    Returns:
        A normalized topic signature string
    """
    text_lower = text.lower()

    # Remove dates and numbers
    for pattern in DATE_PATTERNS + NUMBER_PATTERNS:
        text_lower = re.sub(pattern, "<VALUE>", text_lower, flags=re.IGNORECASE)

    # Remove common filler words
    filler = [
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "to",
        "for",
        "of",
        "on",
        "at",
        "in",
        "with",
        "by",
    ]
    words = text_lower.split()
    significant_words = [w for w in words if w not in filler and len(w) > 2]

    # Return first few significant words as topic signature
    return " ".join(significant_words[:5])


async def detect_knowledge_updates(
    observations: list[models.Document],
    *,
    duplicate_threshold: float = 0.05,
    topic_similarity_threshold: float = 0.3,
) -> list[KnowledgeUpdateCandidate]:
    """
    Detect potential knowledge updates using hybrid approach.

    Strategy:
    1. Filter to observations with changeable entities (dates, numbers)
    2. Group by embedding similarity to find topically related pairs
    3. Within groups, look for value conflicts (different dates/numbers)
    4. Return candidates ordered by confidence

    Args:
        observations: List of observations to analyze
        duplicate_threshold: Cosine distance for exact duplicates (skip these)
        topic_similarity_threshold: Cosine distance for topic relatedness

    Returns:
        List of knowledge update candidates for LLM verification
    """
    from src.dreamer.prescan import KnowledgeUpdateCandidate

    if len(observations) < 2:
        return []

    # Step 1: Filter to observations with changeable entities
    changeable_obs: list[tuple[models.Document, dict[str, Any]]] = []
    for obs in observations:
        entities = extract_entities(obs.content)
        has_changeable = (
            len(entities["dates"]) > 0
            or len(entities["numbers"]) > 0
            or entities["has_deadline"]
            or len(entities["changeable_topics"]) > 0
        )
        if has_changeable:
            changeable_obs.append((obs, entities))

    if len(changeable_obs) < 2:
        logger.debug("Not enough observations with changeable entities")
        return []

    logger.debug(f"Found {len(changeable_obs)} observations with changeable entities")

    # Step 2: Compute pairwise similarities to find related observations
    embeddings = [np.array(obs.embedding) for obs, _ in changeable_obs]
    candidates: list[KnowledgeUpdateCandidate] = []

    for i, (obs_a, entities_a) in enumerate(changeable_obs):
        emb_a = embeddings[i]
        norm_a = np.linalg.norm(emb_a)
        if norm_a == 0:
            continue

        for j, (obs_b, entities_b) in enumerate(changeable_obs[i + 1 :], start=i + 1):
            emb_b = embeddings[j]
            norm_b = np.linalg.norm(emb_b)
            if norm_b == 0:
                continue

            # Compute cosine distance
            cosine_sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
            cosine_dist = 1 - cosine_sim

            # Skip exact duplicates
            if cosine_dist < duplicate_threshold:
                continue

            # Check if topically related (not too similar, not too different)
            if cosine_dist > topic_similarity_threshold:
                continue

            # Step 3: Check for value conflicts
            # Different dates about same topic?
            dates_a = set(entities_a["dates"])
            dates_b = set(entities_b["dates"])
            has_date_conflict = (
                len(dates_a) > 0 and len(dates_b) > 0 and dates_a != dates_b
            )

            # Different numbers about same topic?
            numbers_a = set(entities_a["numbers"])
            numbers_b = set(entities_b["numbers"])
            has_number_conflict = (
                len(numbers_a) > 0 and len(numbers_b) > 0 and numbers_a != numbers_b
            )

            # If we found a conflict, create a candidate
            if has_date_conflict or has_number_conflict:
                # Determine which is older/newer
                if obs_a.created_at < obs_b.created_at:
                    old_obs, new_obs = obs_a, obs_b
                else:
                    old_obs, new_obs = obs_b, obs_a

                # Extract topic
                topic = _extract_shared_topic(obs_a.content, obs_b.content)

                candidates.append(
                    KnowledgeUpdateCandidate(
                        old_observation=old_obs,
                        new_observation=new_obs,
                        topic=topic,
                        similarity=cosine_sim,
                    )
                )

    # Sort by similarity (higher = more related = more likely true update)
    candidates.sort(key=lambda x: x.similarity, reverse=True)

    logger.info(f"Detected {len(candidates)} knowledge update candidates")
    return candidates


def _extract_shared_topic(text_a: str, text_b: str) -> str:
    """
    Extract the shared topic between two related observations.

    Args:
        text_a: First observation content
        text_b: Second observation content

    Returns:
        A description of the shared topic
    """
    # Get topic signatures
    sig_a = set(compute_topic_signature(text_a).split())
    sig_b = set(compute_topic_signature(text_b).split())

    # Find common words
    common = sig_a & sig_b

    if common:
        return " ".join(sorted(common)[:3])

    # Fall back to first few words of first observation
    return compute_topic_signature(text_a)[:50]
