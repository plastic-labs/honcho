from logging import getLogger
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import exceptions, models, schemas
from src.crud.peer import get_peer

logger = getLogger(__name__)


async def get_peer_card(
    db: AsyncSession, workspace_name: str, peer_name: str
) -> str | None:
    """
    Get peer card from internal_metadata.

    Args:
        workspace_name: Name of the workspace
        peer_name: Name of the peer

    Returns:
        Peer card for the peer
        (None if the peer does not exist or if the peer card has not been created yet)
    """
    try:
        peer = await get_peer(db, workspace_name, schemas.PeerCreate(name=peer_name))
        return peer.internal_metadata.get("peer_card", None)
    except exceptions.ResourceNotFoundException:
        return None


async def set_peer_card(
    db: AsyncSession, workspace_name: str, peer_name: str, peer_card: str | None
) -> None:
    """
    Set peer card for a peer.

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    stmt = (
        update(models.Peer)
        .where(models.Peer.workspace_name == workspace_name)
        .where(models.Peer.name == peer_name)
        .values(
            internal_metadata=models.Peer.internal_metadata.op("||")(
                {"peer_card": peer_card}
            )
        )
    )
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise exceptions.ResourceNotFoundException(
            f"Peer {peer_name} not found in workspace {workspace_name}"
        )
    await db.commit()


async def get_working_representation(
    db: AsyncSession,
    workspace_name: str,
    observer_name: str,
    observed_name: str,
    session_name: str | None = None,
) -> str:
    """
    Get working representation for observer/observed relationship.

    Args:
        db: Database session
        workspace_name: Name of the workspace
        observer_name: Name of the peer doing the observing
        observed_name: Name of the peer being observed (required for explicit global/local)
        session_name: Optional session name (None for peer-level metadata)

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
    observed_name: str,  # now required
    session_name: str | None = None,
) -> dict[str, Any] | str | None:
    """
    Get raw working representation data from internal_metadata.

    Returns either structured data (new format) or string (legacy format).
    """
    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = "global_representation"
    else:
        metadata_key = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    if session_name:
        stmt = select(models.SessionPeer.internal_metadata).where(
            models.SessionPeer.peer_name == observer_name,
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.session_name == session_name,
        )
    else:
        stmt = select(models.Peer.internal_metadata).where(
            models.Peer.name == observer_name,
            models.Peer.workspace_name == workspace_name,
        )

    result = await db.execute(stmt)
    peer_metadata = result.scalar_one_or_none()

    if not peer_metadata:
        return None

    # Try new prefixed key first, then fallback to legacy keys
    working_rep_data = peer_metadata.get(metadata_key)
    if working_rep_data:
        return working_rep_data

    # Fallback logic for migration period
    legacy_data = peer_metadata.get("latest_working_representation")
    if legacy_data:
        logger.debug(
            "Using legacy key 'latest_working_representation' for %s->%s",
            observer_name,
            observed_name,
        )
        return legacy_data

    # Final fallback to old user_representation key
    USER_REPRESENTATION_METADATA_KEY = "user_representation"
    user_rep_data = peer_metadata.get(USER_REPRESENTATION_METADATA_KEY)
    if user_rep_data:
        logger.debug(
            "Using legacy key '%s' for %s->%s",
            USER_REPRESENTATION_METADATA_KEY,
            observer_name,
            observed_name,
        )
        return user_rep_data

    return None


def _format_observations_by_level(final_observations: dict[str, Any]) -> str:
    """Format final observations into structured text by level."""
    formatted_sections: list[str] = []

    for level in ["explicit", "deductive"]:
        observations: list[Any] = final_observations.get(level, [])
        if observations:
            formatted_sections.append(f"{level.upper()} OBSERVATIONS:")
            formatted_sections.extend(_format_observation_list(observations))
            formatted_sections.append("")

    return "\n".join(formatted_sections) if formatted_sections else ""


def _format_observation_list(observations: list[dict[str, Any] | str]) -> list[str]:
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


async def set_working_representation(
    db: AsyncSession,
    representation: str | dict[str, Any],
    workspace_name: str,
    observer_name: str,  # renamed from peer_name
    observed_name: str,  # now required - no default
    session_name: str | None = None,
) -> None:
    """
    Set working representation for observer/observed relationship.

    Args:
        db: Database session
        representation: Working representation data (string or structured dict)
        workspace_name: Name of the workspace
        observer_name: Name of the peer doing the observing
        observed_name: Name of the peer being observed (required for explicit global/local)
        session_name: Optional session name (None for peer-level metadata)
    """
    # Determine metadata key based on observer/observed relationship
    if observer_name == observed_name:
        metadata_key = "global_representation"
    else:
        metadata_key = construct_collection_name(
            observer=observer_name, observed=observed_name
        )

    if session_name:
        # Session-level: save all types (global and local)
        stmt = (
            update(models.SessionPeer)
            .where(models.SessionPeer.workspace_name == workspace_name)
            .where(models.SessionPeer.peer_name == observer_name)
            .where(models.SessionPeer.session_name == session_name)
            .values(
                internal_metadata=models.SessionPeer.internal_metadata.op("||")(
                    {metadata_key: representation}
                )
            )
        )
    else:
        # Peer-level: only save global representations
        if observer_name == observed_name:
            stmt = (
                update(models.Peer)
                .where(models.Peer.workspace_name == workspace_name)
                .where(models.Peer.name == observer_name)
                .values(
                    internal_metadata=models.Peer.internal_metadata.op("||")(
                        {metadata_key: representation}
                    )
                )
            )
        else:
            logger.error(
                "Skipping peer-level local representation save (this should never happen!): observer=%s, observed=%s",
                observer_name,
                observed_name,
            )
            return

    await db.execute(stmt)
    await db.commit()


def construct_collection_name(*, observer: str, observed: str) -> str:
    return f"{observer}_{observed}"
