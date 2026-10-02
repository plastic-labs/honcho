"""The scope guard inside the dialectic entry points.

The route tests mock `agentic_chat` wholesale (`mock_llm_call_functions` in
tests/conftest.py), so the preflight *inside* it has no coverage there — which is
how a guard that rejected every scoped chat went unnoticed. These call it
directly with the agent stubbed, so no LLM work happens.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src.dialectic.chat import agentic_chat
from src.exceptions import ValidationException
from src.models import Peer, Workspace
from src.utils.scopes import scope_peer_name


async def _create_scope(
    client: TestClient, db_session: AsyncSession, workspace_name: str
) -> str:
    """Create a scope and commit it — the preflight opens its own connection."""
    scope_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{workspace_name}/scopes", json={"id": scope_name}
    )
    assert response.status_code in [200, 201]
    await db_session.commit()
    return scope_name


@pytest.mark.asyncio
async def test_scope_observer_reaches_the_agent(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """A single `scope` swaps the observer to the scope peer, so the preflight must
    let a scope through in the observer position — otherwise every scoped chat 422s."""
    workspace, peer = sample_data
    scope_name = await _create_scope(client, db_session, workspace.name)

    with patch("src.dialectic.chat.DialecticAgent") as agent_cls:
        agent_cls.return_value.answer = AsyncMock(return_value="answered")
        answer = await agentic_chat(
            workspace_name=workspace.name,
            session_name=None,
            query="what do you know?",
            observer=scope_peer_name(scope_name),
            observed=peer.name,
        )

    assert answer == "answered"
    assert agent_cls.call_args.kwargs["observer"] == scope_peer_name(scope_name)


@pytest.mark.asyncio
async def test_scope_observed_still_rejected(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """The invariant the guard exists for: no representation is formed of a scope,
    so it can never be the subject — even if the route's name check was raced."""
    workspace, peer = sample_data
    scope_name = await _create_scope(client, db_session, workspace.name)

    with (
        patch("src.dialectic.chat.DialecticAgent") as agent_cls,
        pytest.raises(ValidationException, match=scope_peer_name(scope_name)),
    ):
        await agentic_chat(
            workspace_name=workspace.name,
            session_name=None,
            query="what do you know?",
            observer=peer.name,
            observed=scope_peer_name(scope_name),
        )
    agent_cls.assert_not_called()
