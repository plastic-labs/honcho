from unittest.mock import AsyncMock, Mock

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.peer import Peer


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
async def test_async_peer_create_without_config():
    """
    Tests AsyncPeer.create method without config parameter to ensure line 83 is covered.
    This test specifically covers the peer creation line in the create classmethod.
    """
    from unittest.mock import Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)

    # Call the create method without config
    peer = await AsyncPeer.create("test-peer-id", "test-workspace-id", mock_client)

    # Verify the peer was created correctly
    assert isinstance(peer, AsyncPeer)
    assert peer.id == "test-peer-id"
    assert peer.workspace_id == "test-workspace-id"
    assert peer._client is mock_client


@pytest.mark.asyncio
async def test_async_peer_create_with_config():
    """
    Tests AsyncPeer.create method with config parameter to ensure lines 85-86 are covered.
    This test specifically covers the conditional check and get_or_create call with config.
    """
    from unittest.mock import AsyncMock, Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)
    mock_client.workspaces = Mock()
    mock_client.workspaces.peers = Mock()
    mock_client.workspaces.peers.get_or_create = AsyncMock()

    # Define test config
    test_config = {"feature_flags": {"test_flag": True}, "model": "gpt-4"}

    # Call the create method with config
    peer = await AsyncPeer.create(
        "test-peer-with-config", "test-workspace", mock_client, config=test_config
    )

    # Verify the peer was created correctly
    assert isinstance(peer, AsyncPeer)
    assert peer.id == "test-peer-with-config"
    assert peer.workspace_id == "test-workspace"
    assert peer._client is mock_client

    # Verify get_or_create was called with the correct parameters
    mock_client.workspaces.peers.get_or_create.assert_called_once_with(
        workspace_id="test-workspace",
        id="test-peer-with-config",
        configuration=test_config,
    )


@pytest.mark.asyncio
async def test_async_peer_create_return_path():
    """
    Tests AsyncPeer.create method return statement (line 92) to ensure it's covered.
    This test specifically targets the return peer statement in both code paths.
    """
    from unittest.mock import AsyncMock, Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)
    mock_client.workspaces = Mock()
    mock_client.workspaces.peers = Mock()
    mock_client.workspaces.peers.get_or_create = AsyncMock()

    # Test path without config (should hit line 92)
    peer_no_config = await AsyncPeer.create(
        "test-peer-return-1", "test-workspace", mock_client
    )
    assert isinstance(peer_no_config, AsyncPeer)
    assert peer_no_config.id == "test-peer-return-1"

    # Test path with config (should also hit line 92)
    test_config: dict[str, object] = {"test": "value"}
    peer_with_config = await AsyncPeer.create(
        "test-peer-return-2", "test-workspace", mock_client, config=test_config
    )
    assert isinstance(peer_with_config, AsyncPeer)
    assert peer_with_config.id == "test-peer-return-2"

    # Verify that get_or_create was called for the config case
    mock_client.workspaces.peers.get_or_create.assert_called_once_with(
        workspace_id="test-workspace",
        id="test-peer-return-2",
        configuration=test_config,
    )


@pytest.mark.asyncio
async def test_peer_config_initialization(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests peer initialization with config parameter to ensure get_or_create is called.
    """
    honcho_client, client_type = client_fixture
    test_config: dict[str, object] = {"feature_flags": {"test_flag": True}}

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-config", config=test_config)
        assert isinstance(peer, AsyncPeer)
        assert peer.id == "test-peer-config"

        # Test without config for comparison
        peer_no_config = await honcho_client.peer(id="test-peer-no-config")
        assert isinstance(peer_no_config, AsyncPeer)
        assert peer_no_config.id == "test-peer-no-config"
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-config", config=test_config)
        assert isinstance(peer, Peer)
        assert peer.id == "test-peer-config"

        # Test without config for comparison
        peer_no_config = honcho_client.peer(id="test-peer-no-config")
        assert isinstance(peer_no_config, Peer)
        assert peer_no_config.id == "test-peer-no-config"


@pytest.mark.asyncio
async def test_peer_chat_empty_response(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that the chat method returns None when response content is empty, None, or "None".
    This test covers line 112 in the peer.py file.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-empty-response")
        assert isinstance(peer, AsyncPeer)

        # Test with empty string response
        mock_response = Mock()
        mock_response.content = ""
        peer._client.workspaces.peers.chat = AsyncMock(return_value=mock_response)
        result = await peer.chat("test question")
        assert result is None

        # Test with None response
        mock_response.content = None
        result = await peer.chat("test question")
        assert result is None

        # Test with "None" string response
        mock_response.content = "None"
        result = await peer.chat("test question")
        assert result is None

    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-empty-response")
        assert isinstance(peer, Peer)

        # Test with empty string response
        mock_response = Mock()
        mock_response.content = ""
        peer._client.workspaces.peers.chat = Mock(return_value=mock_response)
        result = peer.chat("test question")
        assert result is None

        # Test with None response
        mock_response.content = None
        result = peer.chat("test question")
        assert result is None

        # Test with "None" string response
        mock_response.content = "None"
        result = peer.chat("test question")
        assert result is None


@pytest.mark.asyncio
async def test_async_peer_repr():
    """
    Tests the __repr__ method of the AsyncPeer class.
    This test covers line 322 in the async_client/peer.py file.
    """
    from unittest.mock import Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)

    # Create an AsyncPeer instance
    peer = await AsyncPeer.create("test-peer-repr", "test-workspace", mock_client)

    # Test the __repr__ method explicitly
    repr_str = repr(peer)
    expected = "AsyncPeer(id='test-peer-repr')"
    assert repr_str == expected

    # Also test that __repr__ is called via string formatting and debugging contexts
    debug_str = f"{peer!r}"
    assert debug_str == expected

    # Test different peer IDs to ensure the method works with various inputs
    peer2 = await AsyncPeer.create(
        "test-peer-with-dashes", "test-workspace", mock_client
    )
    repr_str2 = repr(peer2)
    expected2 = "AsyncPeer(id='test-peer-with-dashes')"
    assert repr_str2 == expected2


@pytest.mark.asyncio
async def test_async_peer_str():
    """
    Tests the __str__ method of the AsyncPeer class.
    This test covers line 331 in the async_client/peer.py file.
    """
    from unittest.mock import Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)

    # Create an AsyncPeer instance
    peer = await AsyncPeer.create("test-peer-str", "test-workspace", mock_client)

    # Test the __str__ method explicitly
    str_result = str(peer)
    assert str_result == "test-peer-str"

    # Test with different peer ID to ensure robust coverage
    peer2 = await AsyncPeer.create("another-peer-id", "test-workspace", mock_client)
    str_result2 = str(peer2)
    assert str_result2 == "another-peer-id"

    # Test that __str__ method is actually called in string formatting contexts
    formatted_str = f"Peer: {peer}"
    assert formatted_str == "Peer: test-peer-str"

    # Test direct method call to ensure line 331 is definitely executed
    direct_str = peer.__str__()
    assert direct_str == "test-peer-str"


@pytest.mark.asyncio
async def test_peer_repr(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests the __repr__ method of the Peer class.
    This test covers line 302 in the peer.py file.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-repr")
        assert isinstance(peer, AsyncPeer)
        # Skip async peer since we're testing the sync Peer __repr__
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-repr")
        assert isinstance(peer, Peer)

        # Test the __repr__ method
        repr_str = repr(peer)
        expected = "Peer(id='test-peer-repr')"
        assert repr_str == expected


@pytest.mark.asyncio
async def test_peer_str(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests the __str__ method of the Peer class.
    This test covers line 311 in the peer.py file.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-str")
        assert isinstance(peer, AsyncPeer)
        # Skip async peer since we're testing the sync Peer __str__
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-str")
        assert isinstance(peer, Peer)

        # Test the __str__ method
        str_result = str(peer)
        assert str_result == "test-peer-str"


@pytest.mark.asyncio
async def test_async_peer_chat_return_none():
    """
    Tests that AsyncPeer.chat returns None when response content is in the empty set.
    This test specifically covers line 132: return None in async_client/peer.py
    """
    from unittest.mock import AsyncMock, Mock

    from honcho_core import AsyncHoncho as AsyncHonchoCore

    # Create a mock client
    mock_client = Mock(spec=AsyncHonchoCore)
    mock_client.workspaces = Mock()
    mock_client.workspaces.peers = Mock()

    # Create an AsyncPeer instance
    peer = await AsyncPeer.create("test-peer-chat-none", "test-workspace", mock_client)

    # Test with empty string response content
    mock_response = Mock()
    mock_response.content = ""
    mock_client.workspaces.peers.chat = AsyncMock(return_value=mock_response)

    result = await peer.chat("test question")
    assert result is None

    # Test with None response content
    mock_response.content = None
    result = await peer.chat("test question")
    assert result is None

    # Test with "None" string response content
    mock_response.content = "None"
    result = await peer.chat("test question")
    assert result is None

    # Test with actual content (should return the content, not None)
    mock_response.content = "Actual response"
    result = await peer.chat("test question")
    assert result == "Actual response"
