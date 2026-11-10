"""The request session holds no connection while a scope backfill is enqueued.

enqueue_scope_backfill opens its own tracked_db session from the same pool. A caller
still inside a transaction then needs two connections at once, and a saturated pool
of such requests deadlocks until the checkout timeout.
"""

from typing import Any
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas


async def _scope_and_session_with_messages(
    db_session: AsyncSession, workspace: models.Workspace, peer: models.Peer
) -> tuple[str, str]:
    scope_name = str(generate_nanoid())
    result = await crud.get_or_create_scopes(
        db_session, workspace.name, [schemas.ScopeCreate(name=scope_name)]
    )
    session_name = str(generate_nanoid())
    db_session.add(models.Session(name=session_name, workspace_name=workspace.name))
    await db_session.flush()
    db_session.add(
        models.Message(
            session_name=session_name,
            content="hi",
            token_count=1,
            seq_in_session=1,
            peer_name=peer.name,
            workspace_name=workspace.name,
        )
    )
    await db_session.commit()
    await result.post_commit()
    return scope_name, session_name


def _enqueue_asserting_no_open_transaction(db_session: AsyncSession) -> Any:
    enqueued: list[str] = []

    async def fake_enqueue(_workspace: str, *, session_name: str, **_: Any) -> None:
        assert not db_session.in_transaction(), (
            "the caller's session still holds a connection during the enqueue"
        )
        enqueued.append(session_name)

    return fake_enqueue, enqueued


@pytest.mark.asyncio
async def test_add_sessions_to_scope_releases_its_connection_before_enqueue(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = sample_data
    scope_name, session_name = await _scope_and_session_with_messages(
        db_session, workspace, peer
    )
    fake_enqueue, enqueued = _enqueue_asserting_no_open_transaction(db_session)

    with patch("src.deriver.enqueue.enqueue_scope_backfill", fake_enqueue):
        await crud.add_sessions_to_scope(
            db_session, workspace.name, scope_name, [session_name]
        )

    assert enqueued == [session_name]


@pytest.mark.asyncio
async def test_get_or_create_session_releases_its_connection_before_enqueue(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = sample_data
    scope_name, session_name = await _scope_and_session_with_messages(
        db_session, workspace, peer
    )
    fake_enqueue, enqueued = _enqueue_asserting_no_open_transaction(db_session)

    with patch("src.deriver.enqueue.enqueue_scope_backfill", fake_enqueue):
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate.model_validate(
                {"id": session_name, "scopes": [scope_name]}
            ),
            workspace.name,
        )

    assert enqueued == [session_name]
