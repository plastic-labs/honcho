from logging import getLogger
from typing import Final, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.crud.peer import get_peer
from src.deriver.queue_payload import RepresentationPayload
from src.utils.representation import Representation, StoredRepresentation

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
    session_name: str,
) -> StoredRepresentation | None:
    """
    Get raw working representation data from internal_metadata.

    Returns either structured data (new format) or string (legacy format).
    """
    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = WORKING_REPRESENTATION_METADATA_KEY
    else:
        metadata_key = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    stmt = select(models.SessionPeer.internal_metadata).where(
        models.SessionPeer.peer_name == observer_name,
        models.SessionPeer.workspace_name == workspace_name,
        models.SessionPeer.session_name == session_name,
    )

    result = await db.execute(stmt)
    peer_internal_metadata = result.scalar_one_or_none()

    if not peer_internal_metadata:
        logger.warning(
            f"No peer {observer_name} internal metadata found for session {session_name}"
        )
        return None

    working_rep_data = peer_internal_metadata.get(metadata_key)
    if working_rep_data:
        return StoredRepresentation(**working_rep_data)

    logger.warning(
        f"No working representation found for observer: {observer_name}, observed: {observed_name}"
    )
    return None


async def set_working_representation(
    db: AsyncSession,
    representation: Representation,
    payload: RepresentationPayload,
) -> None:
    """
    Set working representation for observer/observed relationship.

    If the provided representation is structured (dict with `final_observations`),
    append new observations to the existing ones for both `explicit` and `deductive`
    kinds, update `message_id` and `created_at`, and cap each observations list to
    the most recent `WORKING_REPRESENTATION_MAX_OBSERVATIONS` items (FIFO trimming
    of oldest entries).

    Args:
        db: Database session
        representation: Working representation data
        workspace_name: Name of the workspace
        observer_name: Name of the peer doing the observing
        observed_name: Name of the peer being observed (required for explicit global/local)
        session_name: Name of the session
    """

    observer_name = payload.target_name
    observed_name = payload.sender_name

    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = WORKING_REPRESENTATION_METADATA_KEY
    else:
        metadata_key = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    new_representation = await get_working_representation(
        db=db,
        workspace_name=payload.workspace_name,
        observer_name=observer_name,
        observed_name=observed_name,
        session_name=payload.session_name,
    )
    if new_representation:
        new_representation.merge_representation(representation)
    else:
        new_representation = StoredRepresentation(
            created_at=payload.created_at,
            message_id=str(payload.message_id),
            **representation.model_dump(),
        )

    stmt = (
        update(models.SessionPeer)
        .where(models.SessionPeer.workspace_name == payload.workspace_name)
        .where(models.SessionPeer.peer_name == observer_name)
        .where(models.SessionPeer.session_name == payload.session_name)
        .values(
            internal_metadata=models.SessionPeer.internal_metadata.op("||")(
                {metadata_key: new_representation.model_dump(mode="json")}
            )
        )
    )

    await db.execute(stmt)
    await db.commit()

    logger.info(
        "Saved working representation to session peer %s - %s with key %s",
        payload.session_name,
        observer_name,
        metadata_key,
    )


def construct_collection_name(*, observer: str, observed: str) -> str:
    return f"{observer}_{observed}"


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"
