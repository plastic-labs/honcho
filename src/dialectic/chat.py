"""
Chat functionality for the Dialectic API.

Provides the agentic_chat function for answering queries about peers
using the DialecticAgent.
"""

import logging

from src import crud
from src.dependencies import tracked_db
from src.dialectic.core import DialecticAgent

logger = logging.getLogger(__name__)


async def agentic_chat(
    workspace_name: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
) -> str:
    """
    Answer a query about a peer using the agentic dialectic.

    Args:
        workspace_name: Workspace identifier
        session_name: Session identifier (may be None for global queries)
        query: The question to answer about the peer
        observer: The peer making the query
        observed: The peer being queried about

    Returns:
        The synthesized answer string
    """
    async with tracked_db("dialectic.agentic_chat") as db:
        # Get peer cards for context
        observer_peer_card = await crud.get_peer_card(
            db, workspace_name, observer=observer, observed=observer
        )
        observed_peer_card = None
        if observer != observed:
            observed_peer_card = await crud.get_peer_card(
                db, workspace_name, observer=observer, observed=observed
            )

        # Create and run the dialectic agent
        agent = DialecticAgent(
            db=db,
            workspace_name=workspace_name,
            session_name=session_name,
            observer=observer,
            observed=observed,
            observer_peer_card=observer_peer_card,
            observed_peer_card=observed_peer_card,
        )

        response = await agent.answer(query)

    return response
