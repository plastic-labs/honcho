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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models

logger = logging.getLogger(__name__)


@dataclass
class ObservationWithContext:
    """An observation with its source message context for temporal reasoning."""

    observation: models.Document
    source_message: (
        models.Message | None
    )  # The message this observation was extracted from
    preceding_message: (
        models.Message | None
    )  # The message immediately before (for conversation context)


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

    # Explicit observations with their source message context (for temporal reasoning)
    explicit_with_context: list[ObservationWithContext] = field(default_factory=list)

    # Peer card
    peer_card: list[str] = field(default_factory=list)

    # ALL pre-computed analysis - no limits
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
    def cluster_count(self) -> int:
        return len(self.pattern_clusters)

    def summary(self) -> str:
        """Generate a summary string for logging/debugging."""
        return (
            f"DreamContext: "
            f"{self.explicit_count} explicit ({len(self.explicit_with_context)} with context), "
            f"{self.deductive_count} deductive, "
            f"{self.inductive_count} inductive "
            f"({self.total_observations} total), "
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
        + f"{len(deductive)} deductive, {len(inductive)} inductive"
    )

    return explicit, deductive, inductive


async def fetch_message_context_for_observations(
    db: AsyncSession,
    workspace_name: str,
    observations: list[models.Document],
) -> list[ObservationWithContext]:
    """
    Fetch source message context for a list of observations.

    For each observation, retrieves:
    - The source message (from observation's message_ids metadata)
    - The immediately preceding message (for conversation context)

    Args:
        db: Database session
        workspace_name: Workspace identifier
        observations: List of observations to fetch context for

    Returns:
        List of ObservationWithContext with message context attached
    """
    if not observations:
        return []

    # Collect all message IDs we need to fetch
    message_ids_needed: set[int] = set()
    obs_to_message_ids: dict[str, list[int]] = {}

    for obs in observations:
        # Get message_ids from internal_metadata
        metadata = obs.internal_metadata or {}
        message_ids = metadata.get("message_ids", [])
        if message_ids:
            obs_to_message_ids[obs.id] = message_ids
            message_ids_needed.update(message_ids)

    if not message_ids_needed:
        # No message context available, return observations without context
        return [
            ObservationWithContext(
                observation=obs, source_message=None, preceding_message=None
            )
            for obs in observations
        ]

    # Fetch all needed messages in one query
    stmt = select(models.Message).where(
        models.Message.workspace_name == workspace_name,
        models.Message.id.in_(message_ids_needed),
    )
    result = await db.execute(stmt)
    messages_by_id: dict[int, models.Message] = {
        msg.id: msg for msg in result.scalars().all()
    }

    # For each message, we also need the preceding message (seq_in_session - 1)
    # Collect the session/seq pairs we need
    preceding_keys: list[tuple[str, int]] = []
    for msg in messages_by_id.values():
        if msg.seq_in_session > 0:
            preceding_keys.append((msg.session_name, msg.seq_in_session - 1))

    # Fetch preceding messages
    preceding_messages: dict[tuple[str, int], models.Message] = {}
    if preceding_keys:
        # Build a query for all preceding messages
        # This is a bit tricky - we need to match (session_name, seq_in_session) pairs
        for session_name, seq in preceding_keys:
            stmt = select(models.Message).where(
                models.Message.workspace_name == workspace_name,
                models.Message.session_name == session_name,
                models.Message.seq_in_session == seq,
            )
            result = await db.execute(stmt)
            msg = result.scalar_one_or_none()
            if msg:
                preceding_messages[(session_name, seq)] = msg

    # Build ObservationWithContext objects
    result_list: list[ObservationWithContext] = []
    for obs in observations:
        source_message: models.Message | None = None
        preceding_message: models.Message | None = None

        message_ids = obs_to_message_ids.get(obs.id, [])
        if message_ids:
            # Use the last message ID as the source (most recent)
            source_id = message_ids[-1]
            source_message = messages_by_id.get(source_id)

            # Get the preceding message if we have a source
            if source_message and source_message.seq_in_session > 0:
                key = (source_message.session_name, source_message.seq_in_session - 1)
                preceding_message = preceding_messages.get(key)

        result_list.append(
            ObservationWithContext(
                observation=obs,
                source_message=source_message,
                preceding_message=preceding_message,
            )
        )

    return result_list


async def prescan_for_dream(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    *,
    cluster_min_samples: int = 10,
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
    from src.dreamer.clustering import cluster_observations

    logger.info(f"Exhaustive pre-scan starting: {workspace_name}/{observer}/{observed}")

    # 1. Fetch ALL observations
    explicit, deductive, inductive = await fetch_all_observations(
        db, workspace_name, observer, observed
    )

    # 2. Fetch peer card
    peer_card = await crud.peer_card.get_peer_card(
        db, workspace_name, observer=observer, observed=observed
    )

    # 3. Fetch message context for explicit observations (for temporal reasoning)
    explicit_with_context = await fetch_message_context_for_observations(
        db, workspace_name, explicit
    )

    # 5. Cluster ALL explicit observations for induction
    pattern_clusters = cluster_observations(explicit, min_samples=cluster_min_samples)

    context = DreamContext(
        explicit_observations=explicit,
        deductive_observations=deductive,
        inductive_observations=inductive,
        explicit_with_context=explicit_with_context,
        peer_card=peer_card or [],
        pattern_clusters=pattern_clusters,
    )

    logger.info(f"Exhaustive pre-scan complete: {context.summary()}")
    return context
