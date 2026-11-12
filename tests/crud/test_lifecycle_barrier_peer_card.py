"""A public peer-card write obeys parent liveness.

The route passes no snapshot: ``set_peer_card`` captures and checks in its own
transaction. It still creates the observer on first use, as before.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.peer_card import get_peer_card, set_peer_card
from src.exceptions import WriteParentGoneError
from tests.lifecycle_barrier_graph import Graph, seed_graph, tombstone

CARD = ["IDENTITY: Name: Ann"]


async def _set(db: AsyncSession, graph: Graph, *, observer: str, observed: str) -> None:
    await set_peer_card(
        db, graph.workspace.name, CARD, observer=observer, observed=observed
    )


async def test_a_public_card_write_lands_on_live_peers(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    await _set(
        db_session, graph, observer=graph.observer.name, observed=graph.observed.name
    )
    assert (
        await get_peer_card(
            db_session,
            graph.workspace.name,
            observer=graph.observer.name,
            observed=graph.observed.name,
        )
        == CARD
    )


async def test_a_public_card_write_still_creates_its_observer_on_first_use(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    await _set(db_session, graph, observer="new-observer", observed="new-observer")
    assert (
        await get_peer_card(
            db_session,
            graph.workspace.name,
            observer="new-observer",
            observed="new-observer",
        )
        == CARD
    )


async def test_a_public_card_about_a_peer_not_created_yet_is_still_allowed(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    await _set(db_session, graph, observer=graph.observer.name, observed="not-yet")
    assert (
        await get_peer_card(
            db_session,
            graph.workspace.name,
            observer=graph.observer.name,
            observed="not-yet",
        )
        == CARD
    )


async def test_a_public_card_about_a_deleted_peer_is_refused(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    await tombstone(db_session, graph, "observed")

    with pytest.raises(WriteParentGoneError) as raised:
        await _set(
            db_session,
            graph,
            observer=graph.observer.name,
            observed=graph.observed.name,
        )

    assert raised.value.status_code == 404
    metadata = await db_session.scalar(
        select(models.Peer.internal_metadata).where(
            models.Peer.workspace_name == graph.workspace.name,
            models.Peer.name == graph.observer.name,
        )
    )
    assert metadata == {}
