from collections.abc import Sequence
from logging import getLogger
from typing import Final, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.config import settings
from src.crud.peer import get_peer
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)

logger = getLogger(__name__)

# The collection name for documents that make up a peer's global representation
GLOBAL_REPRESENTATION_COLLECTION_NAME: Final[str] = "global_representation"

# The key for the working representation in the session peer's internal_metadata
WORKING_REPRESENTATION_METADATA_KEY = "working_representation"

# Old working representation key--remove in 2.3.0?
WORKING_REPRESENTATION_LEGACY_METADATA_KEY = "global_representation"


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
) -> Representation | None:
    """
    Get raw working representation data from internal_metadata.
    """
    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        collection_name = WORKING_REPRESENTATION_METADATA_KEY
    else:
        collection_name = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    stmt = (
        select(models.Document)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.collection_name == collection_name,
        )
        .order_by(models.Document.created_at.desc())
        .limit(settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS)
    )

    result = await db.execute(stmt)
    documents = result.scalars().all()

    if not documents:
        logger.warning(f"No peer {observer_name} observations found")
        return None

    return representation_from_documents(documents)


def representation_from_documents(
    documents: Sequence[models.Document],
) -> Representation:
    return Representation(
        explicit=[
            ExplicitObservation(
                created_at=doc.created_at,
                content=doc.content,
                message_id=doc.internal_metadata["message_id"],
                session_name=doc.internal_metadata["session_name"],
            )
            for doc in documents
            if doc.internal_metadata["level"] == "explicit"
        ],
        deductive=[
            DeductiveObservation(
                created_at=doc.created_at,
                conclusion=doc.content,
                message_id=doc.internal_metadata["message_id"],
                session_name=doc.internal_metadata["session_name"],
                premises=doc.internal_metadata["premises"],
            )
            for doc in documents
            if doc.internal_metadata["level"] == "deductive"
        ],
    )


def construct_collection_name(*, observer: str, observed: str) -> str:
    return f"{observer}_{observed}"


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"
