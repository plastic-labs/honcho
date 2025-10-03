from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from typing import Any, Final, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.config import settings
from src.crud.peer import get_peer
from src.utils.embedding_store import EmbeddingStore
from src.utils.formatting import parse_datetime_iso
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)

logger = getLogger(__name__)

# The collection name for documents that make up a peer's global representation
GLOBAL_REPRESENTATION_COLLECTION_NAME: Final[str] = "global_representation"


async def get_peer_card(
    db: AsyncSession,
    workspace_name: str,
    observed_name: str,
    observer_name: str,
) -> list[str] | None:
    """
    Get peer card from internal_metadata.

    The peer card is returned for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observed_name: Peer name of the peer described in the peer card
        observer_name: Peer name of the observer

    Returns:
        The peer's card text if present, otherwise None (also None if peer not found).
    """
    try:
        peer = await get_peer(
            db, workspace_name, schemas.PeerCreate(name=observer_name)
        )
        return cast(
            list[str] | None,
            peer.internal_metadata.get(
                construct_peer_card_label(
                    observer=observer_name, observed=observed_name
                )
            ),
        )
    except exceptions.ResourceNotFoundException:
        return None


async def set_peer_card(
    db: AsyncSession,
    workspace_name: str,
    observed_name: str,
    observer_name: str,
    peer_card: list[str] | None,
) -> None:
    """
    Set peer card for a peer.

    If observer_name is provided, the peer card is set for the observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observed_name: Peer name of the peer described in the peer card
        observer_name: Peer name of the observer
        peer_card: List of strings to set as the peer card

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == observer_name)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                {
                    construct_peer_card_label(
                        observer=observer_name, observed=observed_name
                    ): peer_card
                }
            )
        )
    )
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Peer {observer_name} not found in workspace {workspace_name}"
        )
    await db.commit()


async def get_working_representation(
    db: AsyncSession,
    workspace_name: str,
    observer_name: str,
    observed_name: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    include_most_derived: bool = False,
    max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
) -> Representation:
    """
    Get raw working representation data from the relevant document collection.
    """
    # Determine metadata key based on observer/observed relationship
    collection_name = construct_collection_name(
        observer=observer_name, observed=observed_name
    )

    total = max_observations

    # 0 if no semantic query, otherwise min of total // 3 and total
    semantic_observations = (
        min(
            max(
                0,
                semantic_search_top_k
                if semantic_search_top_k is not None
                else total // 3,
            ),
            total,
        )
        if include_semantic_query
        else 0
    )

    if include_semantic_query and include_most_derived:
        # three-way blend: both semantic and derived requested
        top_observations = min(max(0, total // 3), total - semantic_observations)
    elif include_most_derived:
        # two-way blend: only derived requested
        top_observations = min(max(0, total // 2), total - semantic_observations)
    else:
        # no derived observations requested
        top_observations = 0

    # remaining observations are recent
    recent_observations = total - semantic_observations - top_observations

    if include_semantic_query:
        semantically_relevant_representation = await EmbeddingStore(
            workspace_name=workspace_name,
            peer_name=observer_name,
            collection_name=collection_name,
            db=db,
        ).get_relevant_observations(
            query=include_semantic_query,
            top_k=semantic_observations,
            max_distance=semantic_search_max_distance
            if semantic_search_max_distance is not None
            else 0.3,
        )
        representation = semantically_relevant_representation
    else:
        representation = Representation()

    if include_most_derived:
        stmt = (
            select(models.Document)
            .limit(top_observations)
            .where(
                models.Document.workspace_name == workspace_name,
                models.Document.collection_name == collection_name,
            )
            .order_by(models.Document.internal_metadata["times_derived"].desc())
        )

        result = await db.execute(stmt)
        documents = result.scalars().all()

        representation.merge_representation(representation_from_documents(documents))

    stmt = (
        select(models.Document)
        .limit(recent_observations)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.collection_name == collection_name,
            *(
                [models.Document.session_name == session_name]
                if session_name is not None
                else []
            ),
        )
        .order_by(models.Document.created_at.desc())
    )

    result = await db.execute(stmt)
    documents = result.scalars().all()

    if not documents:
        logger.warning(
            f"No observations for {observed_name} (observer: {observer_name}) found. Normal if brand-new peer."
        )

    representation.merge_representation(representation_from_documents(documents))

    return representation


def _safe_datetime_from_metadata(
    internal_metadata: dict[str, Any], fallback_datetime: datetime
) -> datetime:
    message_created_at = internal_metadata.get("message_created_at")
    if message_created_at is None:
        return fallback_datetime.replace(microsecond=0)

    if isinstance(message_created_at, str):
        try:
            return parse_datetime_iso(message_created_at)
        except ValueError:
            return fallback_datetime.replace(microsecond=0)

    if isinstance(message_created_at, datetime):
        return message_created_at.replace(microsecond=0)
    return fallback_datetime.replace(microsecond=0)


def representation_from_documents(
    documents: Sequence[models.Document],
) -> Representation:
    return Representation(
        explicit=[
            ExplicitObservation(
                created_at=_safe_datetime_from_metadata(
                    doc.internal_metadata, doc.created_at
                ),
                content=doc.content,
                message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                session_name=doc.session_name,
            )
            for doc in documents
            if doc.internal_metadata.get("level") == "explicit"
        ],
        deductive=[
            DeductiveObservation(
                created_at=_safe_datetime_from_metadata(
                    doc.internal_metadata, doc.created_at
                ),
                conclusion=doc.content,
                message_ids=doc.internal_metadata.get("message_ids", [(0, 0)]),
                session_name=doc.session_name,
                premises=doc.internal_metadata.get("premises", []),
            )
            for doc in documents
            if doc.internal_metadata.get("level") == "deductive"
        ],
    )


def construct_collection_name(*, observer: str, observed: str) -> str:
    if observer == observed:
        return GLOBAL_REPRESENTATION_COLLECTION_NAME
    return f"{observer}_{observed}"


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"
