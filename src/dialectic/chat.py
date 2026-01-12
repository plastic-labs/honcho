"""
Chat functionality for the Dialectic API.

Provides the agentic_chat function for answering queries about peers
using the DialecticAgent.
"""

import logging
from collections.abc import AsyncIterator

from src import crud
from src.config import ReasoningLevel
from src.dependencies import tracked_db
from src.dialectic.core import DialecticAgent
from src.utils.config_helpers import get_configuration

logger = logging.getLogger(__name__)


async def agentic_chat(
    workspace_name: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
) -> str:
    """
    Answer a query about a peer using the agentic dialectic.

    Args:
        workspace_name: Workspace identifier
        session_name: Session identifier (may be None for global queries)
        query: The question to answer about the peer
        observer: The peer making the query
        observed: The peer being queried about
        reasoning_level: Level of reasoning to apply

    Returns:
        The synthesized answer string
    """
    async with tracked_db("dialectic.agentic_chat") as db:
        # Resolve configuration to check if peer cards should be used
        session = None
        if session_name:
            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
        workspace = await crud.get_workspace(db, workspace_name=workspace_name)
        configuration = get_configuration(None, session, workspace)

        # Get peer cards for context (if enabled)
        observer_peer_card = None
        observed_peer_card = None
        if configuration.peer_card.use:
            observer_peer_card = await crud.get_peer_card(
                db, workspace_name, observer=observer, observed=observer
            )
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
            reasoning_level=reasoning_level,
        )

        response = await agent.answer(query)

    return response


async def agentic_chat_stream(
    workspace_name: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
) -> AsyncIterator[str]:
    """
    Stream an answer to a query about a peer using the agentic dialectic.

    Args:
        workspace_name: Workspace identifier
        session_name: Session identifier (may be None for global queries)
        query: The question to answer about the peer
        observer: The peer making the query
        observed: The peer being queried about
        reasoning_level: Level of reasoning to apply

    Yields:
        Chunks of the response text as they are generated
    """
    async with tracked_db("dialectic.agentic_chat_stream") as db:
        # Resolve configuration to check if peer cards should be used
        session = None
        if session_name:
            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
        workspace = await crud.get_workspace(db, workspace_name=workspace_name)
        configuration = get_configuration(None, session, workspace)

        # Get peer cards for context (if enabled)
        observer_peer_card = None
        observed_peer_card = None
        if configuration.peer_card.use:
            observer_peer_card = await crud.get_peer_card(
                db, workspace_name, observer=observer, observed=observer
            )
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
            reasoning_level=reasoning_level,
        )

        async for chunk in agent.answer_stream(query):
            yield chunk
