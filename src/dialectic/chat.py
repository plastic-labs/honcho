"""
Chat functionality for the Dialectic API.

Provides the agentic_chat function for answering queries about peers
using the DialecticAgent.
"""

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

from src import crud, models
from src.config import ReasoningLevel
from src.dependencies import tracked_db
from src.dialectic.core import DialecticAgent
from src.exceptions import ValidationException
from src.utils.config_helpers import get_configuration
from src.utils.scopes import is_scope_peer

logger = logging.getLogger(__name__)


def _reject_scope_participants(*peers: models.Peer) -> None:
    """Refuse a dialectic run whose observer or observed is a scope peer.

    A scope is a silent observer with ``observe_me=false``: no representation of
    one exists to query. Querying *from* a scope's perspective is a read-side
    surface that does not exist yet, and not something the raw peer routes expose.

    Raises:
        ValidationException: If any participant is a scope.
    """
    offenders = sorted(
        {p.name for p in peers if is_scope_peer(p.name, p.internal_metadata)}
    )
    if offenders:
        raise ValidationException(
            f"Peer name(s) {offenders} are scopes."
            + " No representation is formed of a scope, so a scope cannot be a"
            + " dialectic observer or target."
        )


async def agentic_chat(
    workspace_name: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    response_model: type[BaseModel] | None = None,
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
        session_allowlist: Optional session allowlist restricting all recall
        response_model: Optional Pydantic model the answer must conform to.
            When set, the returned string is JSON matching the model's schema.

    Returns:
        The synthesized answer string
    """
    # Short-lived DB session for validation + config
    async with tracked_db("dialectic.preflight", read_only=True) as db:
        observer_peer = await crud.get_peer(db, workspace_name, observer)
        observed_peer = observer_peer
        if observer != observed:
            observed_peer = await crud.get_peer(db, workspace_name, observed)

        # Resolved-row scope check, not a name check. The routes reject scope
        # names up front for a clear error, but that runs before resolution: a
        # scope created in between would otherwise be used here as observer or
        # target. Checking the rows we just resolved closes that window — an
        # absent name already failed above, and an existing unflagged squatter
        # cannot retroactively become a scope.
        _reject_scope_participants(observer_peer, observed_peer)

        session = None
        if session_name:
            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
        # Read the opaque Session.id while the instance is still bound; the ORM
        # object detaches once this read-only session closes below.
        session_id = session.id if session else None
        workspace = await crud.get_workspace(db, workspace_name=workspace_name)
        configuration = get_configuration(None, session, workspace)

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
    # DB session closed — agent runs without holding a connection

    agent = DialecticAgent(
        workspace_name=workspace_name,
        session_name=session_name,
        session_id=session_id,
        observer=observer,
        observed=observed,
        observer_peer_card=observer_peer_card,
        observed_peer_card=observed_peer_card,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
    )

    return await agent.answer(query, response_model=response_model)


async def agentic_chat_stream(
    workspace_name: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    response_model: type[BaseModel] | None = None,
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
        session_allowlist: Optional session allowlist restricting all recall
        response_model: Optional Pydantic model the answer must conform to.
            When set, the streamed text accumulates to JSON matching the
            model's schema.

    Yields:
        Chunks of the response text as they are generated
    """
    # Short-lived DB session for validation + config
    async with tracked_db("dialectic.preflight", read_only=True) as db:
        observer_peer = await crud.get_peer(db, workspace_name, observer)
        observed_peer = observer_peer
        if observer != observed:
            observed_peer = await crud.get_peer(db, workspace_name, observed)

        # Resolved-row scope check, not a name check. The routes reject scope
        # names up front for a clear error, but that runs before resolution: a
        # scope created in between would otherwise be used here as observer or
        # target. Checking the rows we just resolved closes that window — an
        # absent name already failed above, and an existing unflagged squatter
        # cannot retroactively become a scope.
        _reject_scope_participants(observer_peer, observed_peer)

        session = None
        if session_name:
            session = await crud.get_session(
                db, workspace_name=workspace_name, session_name=session_name
            )
        # Read the opaque Session.id while the instance is still bound; the ORM
        # object detaches once this read-only session closes below.
        session_id = session.id if session else None
        workspace = await crud.get_workspace(db, workspace_name=workspace_name)
        configuration = get_configuration(None, session, workspace)

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
    # DB session closed — agent streams without holding a connection

    agent = DialecticAgent(
        workspace_name=workspace_name,
        session_name=session_name,
        session_id=session_id,
        observer=observer,
        observed=observed,
        observer_peer_card=observer_peer_card,
        observed_peer_card=observed_peer_card,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
    )

    async for chunk in agent.answer_stream(query, response_model=response_model):
        yield chunk
