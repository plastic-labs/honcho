from logging import getLogger
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.config import settings
from src.crud.peer import get_peer
from src.utils.shared_models import ObservationDict

logger = getLogger(__name__)

# The collection name for documents that make up a peer's global representation
GLOBAL_REPRESENTATION_COLLECTION_NAME = "global_representation"

# The key for the working representation in the session peer's internal_metadata
WORKING_REPRESENTATION_METADATA_KEY = "working_representation"

# Old working representation key--remove in 2.3.0?
WORKING_REPRESENTATION_LEGACY_METADATA_KEY = "global_representation"


Observation = str | ObservationDict


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
) -> str:
    """
    Get working representation for observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer_name: Name of the peer doing the observing
        observed_name: Name of the peer being observed
        session_name: Name of the session

    Returns:
        Formatted working representation string
    """
    working_rep_data = await get_working_representation_data(
        db, workspace_name, observer_name, observed_name, session_name
    )

    if not working_rep_data:
        logger.warning(
            f"No working representation found for observer: {observer_name}, observed: {observed_name}"
        )
        return ""

    # Handle both old format (string) and new format (structured data)
    if isinstance(working_rep_data, str):
        return working_rep_data

    # New structured format - extract and format final_observations
    try:
        final_observations = working_rep_data.get("final_observations", {})
        if not final_observations:
            logger.warning("No final_observations found in working representation data")
            return ""

        return _format_observations_by_level(final_observations)
    except Exception:
        logger.exception("Error processing working representation")
        return ""


async def get_working_representation_data(
    db: AsyncSession,
    workspace_name: str,
    observer_name: str,
    observed_name: str,
    session_name: str,
) -> dict[str, Any] | str | None:
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
    peer_metadata = result.scalar_one_or_none()

    if not peer_metadata:
        return None

    working_rep_data = peer_metadata.get(metadata_key)
    if working_rep_data:
        return cast(dict[str, Any] | str, working_rep_data)

    # Try legacy key--remove in 2.3.0?
    if observer_name == observed_name:
        working_rep_data = peer_metadata.get(WORKING_REPRESENTATION_LEGACY_METADATA_KEY)
        if working_rep_data:
            return cast(dict[str, Any] | str, working_rep_data)

    return None


def _format_observations_by_level(final_observations: dict[str, Any]) -> str:
    """Format final observations into structured text by level."""
    formatted_sections: list[str] = []

    for level in ["explicit", "deductive"]:
        observations_raw: Any = final_observations.get(level, [])
        observations: list[Any] = cast(list[Any], observations_raw or [])
        if observations:
            formatted_sections.append(f"{level.upper()} OBSERVATIONS:")
            formatted_sections.extend(_format_observation_list(observations))
            formatted_sections.append("")

    return "\n".join(formatted_sections) if formatted_sections else ""


def _format_observation_list(observations: list[Observation]) -> list[str]:
    """Format a list of observations into consistent string format."""
    formatted: list[str] = []
    for obs in observations:
        if isinstance(obs, dict):
            # Determine core content and premises
            if "conclusion" in obs:
                conclusion_text: str = obs["conclusion"]
                premises: list[str] = obs.get("premises", [])
                if premises:
                    premises_text = "; ".join(premises)
                    formatted_obs = f"{conclusion_text} (based on: {premises_text})"
                else:
                    formatted_obs = conclusion_text
            else:
                content_text: str = obs.get("content", str(obs))
                formatted_obs = content_text

            formatted.append(f"- {formatted_obs}")
        else:
            # Handle string fallback
            formatted.append(f"- {str(obs)}")
    return formatted


def _merge_working_representation(
    existing: dict[str, Any] | str | None, new: dict[str, Any]
) -> dict[str, Any]:
    """Merge a new working representation into an existing one.

    - Appends `explicit` and `deductive` observations in that order
    - Trims each list to the most recent `WORKING_REPRESENTATION_MAX_OBSERVATIONS` entries (FIFO)
    - Uses the latest `thinking`, `message_id`, and `created_at`
    """
    max_observations = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS

    new_final_raw: Any = new.get("final_observations") or {}
    new_explicit: list[Observation] = cast(
        list[Observation], (new_final_raw.get("explicit") or [])
    )
    new_deductive: list[ObservationDict] = cast(
        list[ObservationDict], (new_final_raw.get("deductive") or [])
    )

    existing_explicit: list[Observation] = []
    existing_deductive: list[ObservationDict] = []
    if isinstance(existing, dict):
        existing_final_raw: Any = existing.get("final_observations") or {}
        existing_explicit = cast(
            list[Observation], (existing_final_raw.get("explicit") or [])
        )
        existing_deductive = cast(
            list[ObservationDict], (existing_final_raw.get("deductive") or [])
        )

    merged_explicit: list[Observation] = existing_explicit + new_explicit
    merged_deductive: list[ObservationDict] = existing_deductive + new_deductive

    if len(merged_explicit) > max_observations:
        merged_explicit = merged_explicit[-max_observations:]
    if len(merged_deductive) > max_observations:
        merged_deductive = merged_deductive[-max_observations:]

    return {
        "final_observations": {
            "explicit": merged_explicit,
            "thinking": cast(str | None, new_final_raw.get("thinking")),
            "deductive": merged_deductive,
        },
        "message_id": cast(str | None, new.get("message_id")),
        "created_at": cast(str | None, new.get("created_at")),
    }


async def set_working_representation(
    db: AsyncSession,
    representation: str | dict[str, Any],
    workspace_name: str,
    observer_name: str,
    observed_name: str,
    session_name: str,
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
        representation: Working representation data (string or structured dict)
        workspace_name: Name of the workspace
        observer_name: Name of the peer doing the observing
        observed_name: Name of the peer being observed (required for explicit global/local)
        session_name: Name of the session
    """
    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = WORKING_REPRESENTATION_METADATA_KEY
    else:
        metadata_key = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    merged_value: str | dict[str, Any] = representation
    if isinstance(representation, dict):
        try:
            existing = await get_working_representation_data(
                db=db,
                workspace_name=workspace_name,
                observer_name=observer_name,
                observed_name=observed_name,
                session_name=session_name,
            )
            merged_value = _merge_working_representation(
                existing,
                representation,
            )
        except Exception:
            logger.exception(
                "Failed to merge working representation; storing as provided"
            )
            merged_value = representation

    stmt = (
        update(models.SessionPeer)
        .where(models.SessionPeer.workspace_name == workspace_name)
        .where(models.SessionPeer.peer_name == observer_name)
        .where(models.SessionPeer.session_name == session_name)
        .values(
            internal_metadata=models.SessionPeer.internal_metadata.op("||")(
                {metadata_key: merged_value}
            )
        )
    )

    await db.execute(stmt)
    await db.commit()

    logger.info(
        "Saved working representation to session peer %s - %s with key %s",
        session_name,
        observer_name,
        metadata_key,
    )


def construct_collection_name(*, observer: str, observed: str) -> str:
    return f"{observer}_{observed}"


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"
