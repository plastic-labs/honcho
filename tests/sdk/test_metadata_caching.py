"""Tests for metadata and configuration caching in Honcho SDK."""

from typing import cast

import pytest

from sdks.python.src.honcho.api_types import PeerConfig, SessionConfiguration
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import AsyncPage, SyncPage
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_workspace_metadata_caching(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that workspace metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        # Should initialize with None metadata
        assert honcho_client.metadata is None

        # Get metadata should cache it
        metadata = await honcho_client.aio.get_metadata()
        assert isinstance(metadata, dict)
        assert honcho_client.metadata == metadata

        # Set metadata should update cache
        await honcho_client.aio.set_metadata({"theme": "dark", "version": "1.0"})
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}

        # Get should return cached value
        retrieved = await honcho_client.aio.get_metadata()
        assert retrieved == {"theme": "dark", "version": "1.0"}
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}
    else:
        # Should initialize with None metadata
        assert honcho_client.metadata is None

        # Get metadata should cache it
        metadata = honcho_client.get_metadata()
        assert isinstance(metadata, dict)
        assert honcho_client.metadata == metadata

        # Set metadata should update cache
        honcho_client.set_metadata({"theme": "dark", "version": "1.0"})
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}

        # Get should return cached value
        retrieved = honcho_client.get_metadata()
        assert retrieved == {"theme": "dark", "version": "1.0"}
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}


@pytest.mark.asyncio
async def test_peer_metadata_caching(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that peer metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-peer-meta-cache")
        assert isinstance(peer, Peer)

        # Get metadata should cache it
        metadata = await peer.aio.get_metadata()
        assert isinstance(metadata, dict)
        assert peer.metadata == metadata

        # Set metadata should update cache
        await peer.aio.set_metadata({"name": "Alice", "role": "user"})
        assert peer.metadata == {"name": "Alice", "role": "user"}

        # Get should return cached value
        retrieved = await peer.aio.get_metadata()
        assert retrieved == {"name": "Alice", "role": "user"}
    else:
        peer = honcho_client.peer(id="test-peer-meta-cache")
        assert isinstance(peer, Peer)

        # Get metadata should cache it
        metadata = peer.get_metadata()
        assert isinstance(metadata, dict)
        assert peer.metadata == metadata

        # Set metadata should update cache
        peer.set_metadata({"name": "Alice", "role": "user"})
        assert peer.metadata == {"name": "Alice", "role": "user"}

        # Get should return cached value
        retrieved = peer.get_metadata()
        assert retrieved == {"name": "Alice", "role": "user"}


@pytest.mark.asyncio
async def test_peer_config_caching(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that peer configuration is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-peer-config-cache")
        assert isinstance(peer, Peer)

        # Get config should cache it
        config = await peer.aio.get_configuration()
        assert isinstance(config, PeerConfig)
        assert peer.configuration == config

        # Set config should update cache
        new_config = PeerConfig(observe_me=True)
        await peer.aio.set_configuration(new_config)
        assert peer.configuration == new_config

        # Get should return cached value
        retrieved = await peer.aio.get_configuration()
        assert retrieved == new_config
    else:
        peer = honcho_client.peer(id="test-peer-config-cache")
        assert isinstance(peer, Peer)

        # Get config should cache it
        config = peer.get_configuration()
        assert isinstance(config, PeerConfig)
        assert peer.configuration == config

        # Set config should update cache
        new_config = PeerConfig(observe_me=True)
        peer.set_configuration(new_config)
        assert peer.configuration == new_config

        # Get should return cached value
        retrieved = peer.get_configuration()
        assert retrieved == new_config


@pytest.mark.asyncio
async def test_peer_metadata_and_config_independence(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that peer metadata and config are cached independently.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-peer-independent-cache")
        assert isinstance(peer, Peer)

        # Set both metadata and config
        await peer.aio.set_metadata({"name": "Test"})
        config1 = PeerConfig(observe_me=True)
        await peer.aio.set_configuration(config1)

        assert peer.metadata == {"name": "Test"}
        assert peer.configuration == config1

        # Update metadata only
        await peer.aio.set_metadata({"name": "Updated"})

        assert peer.metadata == {"name": "Updated"}
        assert peer.configuration == config1  # Should remain unchanged

        # Update config only
        config2 = PeerConfig(observe_me=False)
        await peer.aio.set_configuration(config2)

        assert peer.metadata == {"name": "Updated"}  # Should remain unchanged
        assert peer.configuration == config2
    else:
        peer = honcho_client.peer(id="test-peer-independent-cache")
        assert isinstance(peer, Peer)

        # Set both metadata and config
        peer.set_metadata({"name": "Test"})
        config1 = PeerConfig(observe_me=True)
        peer.set_configuration(config1)

        assert peer.metadata == {"name": "Test"}
        assert peer.configuration == config1

        # Update metadata only
        peer.set_metadata({"name": "Updated"})

        assert peer.metadata == {"name": "Updated"}
        assert peer.configuration == config1  # Should remain unchanged

        # Update config only
        config2 = PeerConfig(observe_me=False)
        peer.set_configuration(config2)

        assert peer.metadata == {"name": "Updated"}  # Should remain unchanged
        assert peer.configuration == config2


@pytest.mark.asyncio
async def test_peer_list_with_metadata_and_config(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that listed peers have metadata and config populated from API response.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        # Create peers with metadata and config
        peer1 = await honcho_client.aio.peer(id="test-list-peer1")
        await peer1.aio.set_metadata({"name": "Alice"})
        await peer1.aio.set_configuration(PeerConfig(observe_me=True))

        peer2 = await honcho_client.aio.peer(id="test-list-peer2")
        await peer2.aio.set_metadata({"name": "Bob"})
        await peer2.aio.set_configuration(PeerConfig(observe_me=False))

        # List peers and check cached data
        peers_page = await honcho_client.aio.peers()
        assert isinstance(peers_page, AsyncPage)

        peers = cast(list[Peer], peers_page.items)
        peer_map = {p.id: p for p in peers}

        if "test-list-peer1" in peer_map:
            p1 = peer_map["test-list-peer1"]
            assert p1.metadata == {"name": "Alice"}
            assert p1.configuration is not None
            assert p1.configuration.observe_me is True

        if "test-list-peer2" in peer_map:
            p2 = peer_map["test-list-peer2"]
            assert p2.metadata == {"name": "Bob"}
            assert p2.configuration is not None
            assert p2.configuration.observe_me is False
    else:
        # Create peers with metadata and config
        peer1 = honcho_client.peer(id="test-list-peer1")
        peer1.set_metadata({"name": "Alice"})
        peer1.set_configuration(PeerConfig(observe_me=True))

        peer2 = honcho_client.peer(id="test-list-peer2")
        peer2.set_metadata({"name": "Bob"})
        peer2.set_configuration(PeerConfig(observe_me=False))

        # List peers and check cached data
        peers_page = honcho_client.peers()
        assert isinstance(peers_page, SyncPage)

        peers = cast(list[Peer], list(peers_page))
        peer_map = {p.id: p for p in peers}

        if "test-list-peer1" in peer_map:
            p1 = peer_map["test-list-peer1"]
            assert p1.metadata == {"name": "Alice"}
            assert p1.configuration is not None
            assert p1.configuration.observe_me is True

        if "test-list-peer2" in peer_map:
            p2 = peer_map["test-list-peer2"]
            assert p2.metadata == {"name": "Bob"}
            assert p2.configuration is not None
            assert p2.configuration.observe_me is False


@pytest.mark.asyncio
async def test_session_metadata_caching(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that session metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-meta-cache")
        assert isinstance(session, Session)

        # Get metadata should cache it
        metadata = await session.aio.get_metadata()
        assert isinstance(metadata, dict)
        assert session.metadata == metadata

        # Set metadata should update cache
        await session.aio.set_metadata({"title": "Chat Session", "active": True})
        assert session.metadata == {"title": "Chat Session", "active": True}

        # Get should return cached value
        retrieved = await session.aio.get_metadata()
        assert retrieved == {"title": "Chat Session", "active": True}
    else:
        session = honcho_client.session(id="test-session-meta-cache")
        assert isinstance(session, Session)

        # Get metadata should cache it
        metadata = session.get_metadata()
        assert isinstance(metadata, dict)
        assert session.metadata == metadata

        # Set metadata should update cache
        session.set_metadata({"title": "Chat Session", "active": True})
        assert session.metadata == {"title": "Chat Session", "active": True}

        # Get should return cached value
        retrieved = session.get_metadata()
        assert retrieved == {"title": "Chat Session", "active": True}


@pytest.mark.asyncio
async def test_session_config_caching(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that session configuration is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-config-cache")
        assert isinstance(session, Session)

        # Get config should cache it
        config = await session.aio.get_configuration()
        assert isinstance(config, SessionConfiguration)
        assert session.configuration == config

        # Set config should update cache
        new_config = SessionConfiguration()
        await session.aio.set_configuration(new_config)
        assert session.configuration == new_config

        # Get should return cached value
        retrieved = await session.aio.get_configuration()
        assert retrieved == new_config
    else:
        session = honcho_client.session(id="test-session-config-cache")
        assert isinstance(session, Session)

        # Get config should cache it
        config = session.get_configuration()
        assert isinstance(config, SessionConfiguration)
        assert session.configuration == config

        # Set config should update cache
        new_config = SessionConfiguration()
        session.set_configuration(new_config)
        assert session.configuration == new_config

        # Get should return cached value
        retrieved = session.get_configuration()
        assert retrieved == new_config


@pytest.mark.asyncio
async def test_session_metadata_and_config_independence(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that session metadata and config are cached independently.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(id="test-session-independent-cache")
        assert isinstance(session, Session)

        # Set both metadata and config
        await session.aio.set_metadata({"title": "Test"})
        config1 = SessionConfiguration()
        await session.aio.set_configuration(config1)

        assert session.metadata == {"title": "Test"}
        assert session.configuration == config1

        # Update metadata only
        await session.aio.set_metadata({"title": "Updated"})

        assert session.metadata == {"title": "Updated"}
        assert session.configuration == config1  # Should remain unchanged

        # Update config only
        config2 = SessionConfiguration()
        await session.aio.set_configuration(config2)

        assert session.metadata == {"title": "Updated"}  # Should remain unchanged
        assert session.configuration == config2
    else:
        session = honcho_client.session(id="test-session-independent-cache")
        assert isinstance(session, Session)

        # Set both metadata and config
        session.set_metadata({"title": "Test"})
        config1 = SessionConfiguration()
        session.set_configuration(config1)

        assert session.metadata == {"title": "Test"}
        assert session.configuration == config1

        # Update metadata only
        session.set_metadata({"title": "Updated"})

        assert session.metadata == {"title": "Updated"}
        assert session.configuration == config1  # Should remain unchanged

        # Update config only
        config2 = SessionConfiguration()
        session.set_configuration(config2)

        assert session.metadata == {"title": "Updated"}  # Should remain unchanged
        assert session.configuration == config2


@pytest.mark.asyncio
async def test_session_list_with_metadata_and_config(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that listed sessions have metadata and config populated from API response.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        # Create sessions with metadata and config
        session1 = await honcho_client.aio.session(id="test-list-session1")
        await session1.aio.set_metadata({"title": "Session 1"})
        await session1.aio.set_configuration(SessionConfiguration())

        session2 = await honcho_client.aio.session(id="test-list-session2")
        await session2.aio.set_metadata({"title": "Session 2"})
        await session2.aio.set_configuration(SessionConfiguration())

        # List sessions and check cached data
        sessions_page = await honcho_client.aio.sessions()
        assert isinstance(sessions_page, AsyncPage)

        sessions = cast(list[Session], sessions_page.items)
        session_map = {s.id: s for s in sessions}

        if "test-list-session1" in session_map:
            s1 = session_map["test-list-session1"]
            assert s1.metadata == {"title": "Session 1"}
            assert s1.configuration is not None

        if "test-list-session2" in session_map:
            s2 = session_map["test-list-session2"]
            assert s2.metadata == {"title": "Session 2"}
            assert s2.configuration is not None
    else:
        # Create sessions with metadata and config
        session1 = honcho_client.session(id="test-list-session1")
        session1.set_metadata({"title": "Session 1"})
        session1.set_configuration(SessionConfiguration())

        session2 = honcho_client.session(id="test-list-session2")
        session2.set_metadata({"title": "Session 2"})
        session2.set_configuration(SessionConfiguration())

        # List sessions and check cached data
        sessions_page = honcho_client.sessions()
        assert isinstance(sessions_page, SyncPage)

        sessions = cast(list[Session], list(sessions_page))
        session_map = {s.id: s for s in sessions}

        if "test-list-session1" in session_map:
            s1 = session_map["test-list-session1"]
            assert s1.metadata == {"title": "Session 1"}
            assert s1.configuration is not None

        if "test-list-session2" in session_map:
            s2 = session_map["test-list-session2"]
            assert s2.metadata == {"title": "Session 2"}
            assert s2.configuration is not None


@pytest.mark.asyncio
async def test_peer_initialization_with_metadata_and_config(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that peers can be initialized with metadata and config.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(
            id="test-init-peer",
            metadata={"name": "Test Peer", "role": "assistant"},
            configuration=PeerConfig(observe_me=False),
        )
        assert isinstance(peer, Peer)
        assert peer.metadata == {"name": "Test Peer", "role": "assistant"}
        assert peer.configuration is not None
        assert peer.configuration.observe_me is False
    else:
        peer = honcho_client.peer(
            id="test-init-peer",
            metadata={"name": "Test Peer", "role": "assistant"},
            configuration=PeerConfig(observe_me=False),
        )
        assert isinstance(peer, Peer)
        assert peer.metadata == {"name": "Test Peer", "role": "assistant"}
        assert peer.configuration is not None
        assert peer.configuration.observe_me is False


@pytest.mark.asyncio
async def test_session_initialization_with_metadata_and_config(
    client_fixture: tuple[Honcho, str],
) -> None:
    """
    Tests that sessions can be initialized with metadata and config.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        session = await honcho_client.aio.session(
            id="test-init-session",
            metadata={"title": "Test Session", "tags": ["important"]},
            configuration=SessionConfiguration(),
        )
        assert isinstance(session, Session)
        assert session.metadata == {"title": "Test Session", "tags": ["important"]}
        assert session.configuration is not None
    else:
        session = honcho_client.session(
            id="test-init-session",
            metadata={"title": "Test Session", "tags": ["important"]},
            configuration=SessionConfiguration(),
        )
        assert isinstance(session, Session)
        assert session.metadata == {"title": "Test Session", "tags": ["important"]}
        assert session.configuration is not None
