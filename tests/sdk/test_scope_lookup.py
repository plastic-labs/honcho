"""`honcho.get_scope()` looks a scope up without the get-or-create side effect."""

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.http.exceptions import NotFoundError


@pytest.mark.asyncio
async def test_get_scope_returns_existing_without_creating_missing(
    client_fixture: tuple[Honcho, str],
):
    honcho_client, client_type = client_fixture

    if client_type == "async":
        created = await honcho_client.aio.scope("lookup-me", metadata={"k": "v"})
        found = await honcho_client.aio.get_scope("lookup-me")
        with pytest.raises(NotFoundError):
            await honcho_client.aio.get_scope("never-created")
        listed = {s.id for s in (await honcho_client.aio.scopes()).items}
    else:
        created = honcho_client.scope("lookup-me", metadata={"k": "v"})
        found = honcho_client.get_scope("lookup-me")
        with pytest.raises(NotFoundError):
            honcho_client.get_scope("never-created")
        listed = {s.id for s in honcho_client.scopes().items}

    assert found.id == created.id
    assert found.metadata == {"k": "v"}
    assert found.created_at == created.created_at
    # The failed lookup must not have provisioned the scope as a side effect.
    assert "never-created" not in listed
