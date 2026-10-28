"""Tests for listing sessions by scope membership through the SDK."""

from typing import cast

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_sessions_scope_options(client_fixture: tuple[Honcho, str]) -> None:
    """`scope`, `in_no_scope` and `include_scopes` reach the server on every page.

    Pages hold one session each, and the members of each filter alternate with
    non-members in creation order, so a later page fetched without the options
    would return a different session.
    """
    honcho_client, client_type = client_fixture
    ids = ["s1", "s2", "s3", "s4"]

    if client_type == "async":
        for session_id in ids:
            await honcho_client.aio.session(id=session_id)
        alpha = await honcho_client.aio.scope("alpha")
        beta = await honcho_client.aio.scope("beta")
        await alpha.aio.add_sessions(["s1", "s3"])
        await beta.aio.add_sessions(["s1"])

        async def collect(**options: object) -> list[Session]:
            sessions: list[Session] = []
            page = await honcho_client.aio.sessions(size=1, **options)  # pyright: ignore[reportArgumentType]
            while page is not None:
                sessions += cast(list[Session], page.items)
                page = await page.get_next_page()
            return sessions

        with_scopes = await collect(include_scopes=True)
        in_alpha = await collect(scope="alpha", include_scopes=True)
        in_none = await collect(in_no_scope=True)
        plain = await collect()
    else:
        for session_id in ids:
            honcho_client.session(id=session_id)
        alpha = honcho_client.scope("alpha")
        beta = honcho_client.scope("beta")
        alpha.add_sessions(["s1", "s3"])
        beta.add_sessions(["s1"])

        def collect_sync(**options: object) -> list[Session]:
            page = honcho_client.sessions(size=1, **options)  # pyright: ignore[reportArgumentType]
            return cast(list[Session], list(page))

        with_scopes = collect_sync(include_scopes=True)
        in_alpha = collect_sync(scope="alpha", include_scopes=True)
        in_none = collect_sync(in_no_scope=True)
        plain = collect_sync()

    assert {s.id: s.scopes for s in with_scopes} == {
        "s1": ["alpha", "beta"],
        "s2": [],
        "s3": ["alpha"],
        "s4": [],
    }
    assert [(s.id, s.scopes) for s in in_alpha] == [
        ("s1", ["alpha", "beta"]),
        ("s3", ["alpha"]),
    ]
    assert [s.id for s in in_none] == ["s2", "s4"]
    assert [s.id for s in plain] == ids
    assert all(s.scopes is None for s in plain)
