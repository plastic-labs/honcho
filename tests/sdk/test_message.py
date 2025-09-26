"""Tests for the Message class functionality."""

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.async_client.session import (
    AsyncSession,
)
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_update_after_getting_message(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-update-message")
        assert isinstance(session, AsyncSession)

        user = await honcho_client.peer(id="user-msg")
        assert isinstance(user, AsyncPeer)

        await session.add_messages(
            [
                user.message("Hello, world!"),
            ]
        )
        messages_page = await session.get_messages()
        messages = messages_page.items
        assert len(messages) == 1
        assert messages[0].metadata == {}

        message = messages[0]
        await message.update({"foo": "bar"})

        messages_page = await session.get_messages()
        messages = messages_page.items
        assert len(messages) == 1
        assert messages[0].metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-meta")
        assert isinstance(session, Session)

        user = honcho_client.peer(id="user-msg")
        assert isinstance(user, Peer)

        session.add_messages([user.message("Hello, world!")])

        messages_page = session.get_messages()
        messages = list(messages_page)
        assert len(messages) == 1
        assert messages[0].metadata == {}
        message = messages[0]

        message.update({"foo": "bar"})

        messages_page = session.get_messages()
        messages = list(messages_page)
        assert len(messages) == 1
        assert messages[0].metadata == {"foo": "bar"}


async def test_update_after_session_search(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    honcho_client, client_type = client_fixture
    search_query = "a unique message for session search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="search-session-s-update")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="search-user-s-update")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message(search_query)])

        search_results = await session.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1

        message = search_results[0]
        assert search_query in message.content
        assert message.metadata == {}

        await message.update({"foo": "bar"})

        new_search_results = await session.search(search_query)
        assert isinstance(new_search_results, list)
        assert len(new_search_results) >= 1

        message = new_search_results[0]
        assert search_query in message.content
        assert message.metadata == {"foo": "bar"}

    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-s-update")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-s-update")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = session.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1
        assert search_query in search_results[0].content

        message = search_results[0]
        assert message.metadata == {}

        message = search_results[0]
        message.update({"foo": "bar"})

        new_search_results = session.search(search_query)
        assert isinstance(new_search_results, list)
        assert len(new_search_results) >= 1

        message = new_search_results[0]
        assert search_query in message.content
        assert message.metadata == {"foo": "bar"}


async def test_update_after_workspace_search(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    honcho_client, client_type = client_fixture
    search_query = "a unique message for workspace search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="search-session-ws-update")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="search-user-ws-update")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message(search_query)])

        search_results = await honcho_client.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1

        message = search_results[0]
        assert search_query in message.content
        assert message.metadata == {}

        await message.update({"foo": "bar"})

        new_search_results = await honcho_client.search(search_query)
        assert isinstance(new_search_results, list)
        assert len(new_search_results) >= 1

        message = new_search_results[0]
        assert search_query in message.content
        assert message.metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-ws-update")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-ws-update")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = honcho_client.search(search_query)
        assert isinstance(search_results, list)
        assert len(search_results) >= 1

        message = search_results[0]
        assert search_query in message.content
        assert message.metadata == {}

        message = search_results[0]
        message.update({"foo": "bar"})

        new_search_results = honcho_client.search(search_query)
        assert isinstance(new_search_results, list)
        assert len(new_search_results) >= 1

        message = new_search_results[0]
        assert search_query in message.content
        assert message.metadata == {"foo": "bar"}
