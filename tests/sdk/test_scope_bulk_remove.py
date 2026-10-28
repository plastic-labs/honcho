"""Tests for removing many sessions from a scope through the SDK."""

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_remove_sessions(client_fixture: tuple[Honcho, str]) -> None:
    """Members are removed in one call; non-members are ignored."""
    honcho_client, client_type = client_fixture
    ids = ["kept", "gone-1", "gone-2", "outsider"]

    if client_type == "async":
        for session_id in ids:
            await honcho_client.aio.session(id=session_id)
        scope = await honcho_client.aio.scope("bulk")
        await scope.aio.add_sessions(ids[:3])
        await scope.aio.remove_sessions(
            ["gone-1", Session("gone-2", honcho_client), "outsider"]
        )
        page = await scope.aio.sessions()
        members = {s.id for s in page.items}
    else:
        for session_id in ids:
            honcho_client.session(id=session_id)
        scope = honcho_client.scope("bulk")
        scope.add_sessions(ids[:3])
        scope.remove_sessions(["gone-1", Session("gone-2", honcho_client), "outsider"])
        members = {s.id for s in scope.sessions()}

    assert members == {"kept"}


def test_remove_sessions_checks_batch_size(
    honcho_sync_test_client: Honcho,
) -> None:
    """An empty or oversized batch fails before any request is sent."""
    scope = honcho_sync_test_client.scope("bulk-limits")
    with pytest.raises(ValueError):
        scope.remove_sessions([])
    with pytest.raises(ValueError):
        scope.remove_sessions([f"s{i}" for i in range(101)])
