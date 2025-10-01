from unittest.mock import AsyncMock, Mock, patch

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.types import DialecticStreamResponse


@pytest.mark.asyncio
async def test_peer_metadata(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests creation and metadata operations for peers.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-meta")
        assert isinstance(peer, AsyncPeer)

        metadata = await peer.get_metadata()
        assert metadata == {}

        await peer.set_metadata({"foo": "bar"})
        metadata = await peer.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-meta")
        assert isinstance(peer, Peer)

        metadata = peer.get_metadata()
        assert metadata == {}

        peer.set_metadata({"foo": "bar"})
        metadata = peer.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_peer_get_sessions(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests retrieving the sessions a peer is a member of.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-sessions")
        session1 = await honcho_client.session(id="s1")
        session2 = await honcho_client.session(id="s2")

        await session1.add_peers(peer)
        await session2.add_peers(peer)

        sessions_page = await peer.get_sessions()
        sessions = sessions_page.items
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-sessions")
        session1 = honcho_client.session(id="s1")
        session2 = honcho_client.session(id="s2")

        session1.add_peers(peer)
        session2.add_peers(peer)

        sessions_page = peer.get_sessions()
        sessions = list(sessions_page)
        assert len(sessions) == 2
        session_ids = {s.id for s in sessions}
        assert "s1" in session_ids
        assert "s2" in session_ids


@pytest.mark.asyncio
async def test_peer_card_global(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting a global peer card (no target).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-card-global-peer")
        session = await honcho_client.session(id="test-card-global-session")

        # Add some messages to create context
        await session.add_messages([peer.message("I like pizza")])

        # Get global peer card
        card_response = await peer.card()
        assert isinstance(card_response, str)
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-card-global-peer")
        session = honcho_client.session(id="test-card-global-session")

        # Add some messages to create context
        session.add_messages([peer.message("I like pizza")])

        # Get global peer card
        card_response = peer.card()
        assert isinstance(card_response, str)


@pytest.mark.asyncio
async def test_peer_card_local(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting a local peer card (with target).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-card-local-observer")
        target = await honcho_client.peer(id="test-card-local-target")
        session = await honcho_client.session(id="test-card-local-session")

        # Add messages from both peers
        await session.add_messages(
            [observer.message("Hello"), target.message("Hi there")]
        )

        # Get local peer card with target as Peer object
        card_response = await observer.card(target=target)
        assert isinstance(card_response, str)

        # Get local peer card with target as string
        card_response = await observer.card(target=target.id)
        assert isinstance(card_response, str)
    else:
        assert isinstance(honcho_client, Honcho)
        observer = honcho_client.peer(id="test-card-local-observer")
        target = honcho_client.peer(id="test-card-local-target")
        session = honcho_client.session(id="test-card-local-session")

        # Add messages from both peers
        session.add_messages([observer.message("Hello"), target.message("Hi there")])

        # Get local peer card with target as Peer object
        card_response = observer.card(target=target)
        assert isinstance(card_response, str)

        # Get local peer card with target as string
        card_response = observer.card(target=target.id)
        assert isinstance(card_response, str)


@pytest.mark.asyncio
async def test_peer_card_validation(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests peer card validation.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-card-validation-peer")

        # Test with empty string
        with pytest.raises(ValueError, match="target string cannot be empty"):
            await peer.card(target="")
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-card-validation-peer")

        # Test with empty string
        with pytest.raises(ValueError, match="target string cannot be empty"):
            peer.card(target="")


@pytest.mark.asyncio
async def test_peer_chat_streaming(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests streaming chat with mocked response generator.
    """

    honcho_client, client_type = client_fixture
    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-stream-async-peer")
        session = await honcho_client.session(id="test-stream-async-session")

        # Add some messages to create context
        await session.add_messages([peer.message("I like pizza")])

        # Mock the async streaming response
        async def mock_aiter_lines():
            yield 'data: {"delta": {"content": "Hello"}}'
            yield 'data: {"delta": {"content": " async"}}'
            yield 'data: {"done": true}'

        mock_response = AsyncMock()
        mock_response.iter_lines = mock_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock the with_streaming_response.chat call
        with patch.object(
            honcho_client.core.workspaces.peers.with_streaming_response,
            "chat",
            return_value=mock_response,
        ):
            result = await peer.chat("Tell me something", stream=True)
            assert isinstance(result, DialecticStreamResponse)

            # Collect chunks
            chunks: list[str] = []
            async for chunk in result:
                assert isinstance(chunk, str)
                chunks.append(chunk)
            assert chunks == ["Hello", " async"]
            assert result.get_final_response()["content"] == "Hello async"
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-stream-peer")
        session = honcho_client.session(id="test-stream-session")

        # Add some messages to create context
        session.add_messages([peer.message("I like pizza")])

        # Mock the streaming response
        def mock_iter_lines():
            yield 'data: {"delta": {"content": "Hello"}}'
            yield 'data: {"delta": {"content": " world"}}'
            yield 'data: {"done": true}'

        mock_response = Mock()
        mock_response.iter_lines = mock_iter_lines
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)

        # Mock the with_streaming_response.chat call
        with patch.object(
            honcho_client.core.workspaces.peers.with_streaming_response,
            "chat",
            return_value=mock_response,
        ):
            result = peer.chat("Tell me something", stream=True)
            assert isinstance(result, DialecticStreamResponse)

            # Collect chunks
            chunks = list(result)
            assert chunks == ["Hello", " world"]
            assert result.get_final_response()["content"] == "Hello world"


@pytest.mark.asyncio
async def test_peer_chat_non_streaming(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests non-streaming chat (already using core SDK).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-non-stream-peer")
        session = await honcho_client.session(id="test-non-stream-session")

        # Add some messages
        await session.add_messages([peer.message("I like pizza")])

        # Non-streaming chat
        response = await peer.chat("What do I like?", stream=False)
        # Response can be None or a string
        assert response is None or isinstance(response, str)
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-non-stream-peer")
        session = honcho_client.session(id="test-non-stream-session")

        # Add some messages
        session.add_messages([peer.message("I like pizza")])

        # Non-streaming chat
        response = peer.chat("What do I like?", stream=False)
        # Response can be None or a string
        assert response is None or isinstance(response, str)
