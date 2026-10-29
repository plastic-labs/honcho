"""Tests for listing scopes with session counts and a name filter through the SDK."""

from typing import cast

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.scope import Scope


@pytest.mark.asyncio
async def test_scopes_filters_and_session_counts(
    client_fixture: tuple[Honcho, str],
) -> None:
    """`filters` and `include_session_count` reach the server on every page.

    Pages hold one scope each, and the scopes the filter matches alternate with
    ones it doesn't, so a later page fetched without the filter would differ.
    """
    honcho_client, client_type = client_fixture
    members = {"keep-1": ["s1"], "drop-1": [], "keep-2": ["s1", "s2"], "drop-2": []}
    by_name = {"id": {"icontains": "keep"}}

    if client_type == "async":
        for session_id in ("s1", "s2"):
            await honcho_client.aio.session(id=session_id)
        for scope_id, sessions in members.items():
            scope = await honcho_client.aio.scope(scope_id)
            if sessions:
                await scope.aio.add_sessions(sessions)

        async def collect(**options: object) -> list[Scope]:
            """Walk every page of the scope list and collect its scopes."""
            scopes: list[Scope] = []
            page = await honcho_client.aio.scopes(size=1, **options)  # pyright: ignore[reportArgumentType]
            while page is not None:
                scopes += cast(list[Scope], page.items)
                page = await page.get_next_page()
            return scopes

        counted = await collect(filters=by_name, include_session_count=True)
        plain = await collect()
    else:
        for session_id in ("s1", "s2"):
            honcho_client.session(id=session_id)
        for scope_id, sessions in members.items():
            scope = honcho_client.scope(scope_id)
            if sessions:
                scope.add_sessions(sessions)

        def collect_sync(**options: object) -> list[Scope]:
            """Walk every page of the scope list and collect its scopes."""
            page = honcho_client.scopes(size=1, **options)  # pyright: ignore[reportArgumentType]
            return cast(list[Scope], list(page))

        counted = collect_sync(filters=by_name, include_session_count=True)
        plain = collect_sync()

    assert [(s.id, s.session_count) for s in counted] == [("keep-1", 1), ("keep-2", 2)]
    assert [s.id for s in plain] == list(members)
    assert all(s.session_count is None for s in plain)
