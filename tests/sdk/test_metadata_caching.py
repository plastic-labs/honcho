"""Tests for metadata and configuration caching in Honcho SDK."""

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.pagination import AsyncPage
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.async_client.session import AsyncSession
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import SyncPage
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_workspace_metadata_caching(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that workspace metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Should initialize with None metadata
        assert honcho_client.metadata is None

        # Get metadata should cache it
        metadata = await honcho_client.get_metadata()
        assert isinstance(metadata, dict)
        assert honcho_client.metadata == metadata

        # Set metadata should update cache
        await honcho_client.set_metadata({"theme": "dark", "version": "1.0"})
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}

        # Get should return cached value
        retrieved = await honcho_client.get_metadata()
        assert retrieved == {"theme": "dark", "version": "1.0"}
        assert honcho_client.metadata == {"theme": "dark", "version": "1.0"}
    else:
        assert isinstance(honcho_client, Honcho)

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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that peer metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-meta-cache")
        assert isinstance(peer, AsyncPeer)

        # Get metadata should cache it
        metadata = await peer.get_metadata()
        assert isinstance(metadata, dict)
        assert peer.metadata == metadata

        # Set metadata should update cache
        await peer.set_metadata({"name": "Alice", "role": "user"})
        assert peer.metadata == {"name": "Alice", "role": "user"}

        # Get should return cached value
        retrieved = await peer.get_metadata()
        assert retrieved == {"name": "Alice", "role": "user"}
    else:
        assert isinstance(honcho_client, Honcho)
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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that peer configuration is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-config-cache")
        assert isinstance(peer, AsyncPeer)

        # Get config should cache it
        config = await peer.get_config()
        assert isinstance(config, dict)
        assert peer.configuration == config

        # Set config should update cache
        await peer.set_config({"observe_me": True, "observe_others": False})
        assert peer.configuration == {"observe_me": True, "observe_others": False}

        # Get should return cached value
        retrieved = await peer.get_config()
        assert retrieved == {"observe_me": True, "observe_others": False}
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-config-cache")
        assert isinstance(peer, Peer)

        # Get config should cache it
        config = peer.get_config()
        assert isinstance(config, dict)
        assert peer.configuration == config

        # Set config should update cache
        peer.set_config({"observe_me": True, "observe_others": False})
        assert peer.configuration == {"observe_me": True, "observe_others": False}

        # Get should return cached value
        retrieved = peer.get_config()
        assert retrieved == {"observe_me": True, "observe_others": False}


@pytest.mark.asyncio
async def test_peer_deprecated_config_methods(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that deprecated getPeerConfig/setPeerConfig methods work and cache properly.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-deprecated-config")
        assert isinstance(peer, AsyncPeer)

        # Deprecated get method should work and cache
        config = await peer.get_peer_config()
        assert isinstance(config, dict)
        assert peer.configuration == config

        # Deprecated set method should work and update cache
        await peer.set_peer_config({"observe_me": False})
        assert peer.configuration == {"observe_me": False}
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-deprecated-config")
        assert isinstance(peer, Peer)

        # Deprecated get method should work and cache
        config = peer.get_peer_config()
        assert isinstance(config, dict)
        assert peer.configuration == config

        # Deprecated set method should work and update cache
        peer.set_peer_config({"observe_me": False})
        assert peer.configuration == {"observe_me": False}


@pytest.mark.asyncio
async def test_peer_metadata_and_config_independence(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that peer metadata and config are cached independently.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-peer-independent-cache")
        assert isinstance(peer, AsyncPeer)

        # Set both metadata and config
        await peer.set_metadata({"name": "Test"})
        await peer.set_config({"observe_me": True})

        assert peer.metadata == {"name": "Test"}
        assert peer.configuration == {"observe_me": True}

        # Update metadata only
        await peer.set_metadata({"name": "Updated"})

        assert peer.metadata == {"name": "Updated"}
        assert peer.configuration == {"observe_me": True}  # Should remain unchanged

        # Update config only
        await peer.set_config({"observe_me": False})

        assert peer.metadata == {"name": "Updated"}  # Should remain unchanged
        assert peer.configuration == {"observe_me": False}
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-peer-independent-cache")
        assert isinstance(peer, Peer)

        # Set both metadata and config
        peer.set_metadata({"name": "Test"})
        peer.set_config({"observe_me": True})

        assert peer.metadata == {"name": "Test"}
        assert peer.configuration == {"observe_me": True}

        # Update metadata only
        peer.set_metadata({"name": "Updated"})

        assert peer.metadata == {"name": "Updated"}
        assert peer.configuration == {"observe_me": True}  # Should remain unchanged

        # Update config only
        peer.set_config({"observe_me": False})

        assert peer.metadata == {"name": "Updated"}  # Should remain unchanged
        assert peer.configuration == {"observe_me": False}


@pytest.mark.asyncio
async def test_peer_list_with_metadata_and_config(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that listed peers have metadata and config populated from API response.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create peers with metadata and config
        peer1 = await honcho_client.peer(id="test-list-peer1")
        await peer1.set_metadata({"name": "Alice"})
        await peer1.set_config({"observe_me": True})

        peer2 = await honcho_client.peer(id="test-list-peer2")
        await peer2.set_metadata({"name": "Bob"})
        await peer2.set_config({"observe_me": False})

        # List peers and check cached data
        peers_page = await honcho_client.get_peers()
        assert isinstance(peers_page, AsyncPage)

        peers = peers_page.items
        peer_map = {p.id: p for p in peers}

        if "test-list-peer1" in peer_map:
            p1 = peer_map["test-list-peer1"]
            assert p1.metadata == {"name": "Alice"}
            assert p1.configuration == {"observe_me": True}

        if "test-list-peer2" in peer_map:
            p2 = peer_map["test-list-peer2"]
            assert p2.metadata == {"name": "Bob"}
            assert p2.configuration == {"observe_me": False}
    else:
        assert isinstance(honcho_client, Honcho)

        # Create peers with metadata and config
        peer1 = honcho_client.peer(id="test-list-peer1")
        peer1.set_metadata({"name": "Alice"})
        peer1.set_config({"observe_me": True})

        peer2 = honcho_client.peer(id="test-list-peer2")
        peer2.set_metadata({"name": "Bob"})
        peer2.set_config({"observe_me": False})

        # List peers and check cached data
        peers_page = honcho_client.get_peers()
        assert isinstance(peers_page, SyncPage)

        peers = list(peers_page)
        peer_map = {p.id: p for p in peers}

        if "test-list-peer1" in peer_map:
            p1 = peer_map["test-list-peer1"]
            assert p1.metadata == {"name": "Alice"}
            assert p1.configuration == {"observe_me": True}

        if "test-list-peer2" in peer_map:
            p2 = peer_map["test-list-peer2"]
            assert p2.metadata == {"name": "Bob"}
            assert p2.configuration == {"observe_me": False}


@pytest.mark.asyncio
async def test_session_metadata_caching(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that session metadata is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-meta-cache")
        assert isinstance(session, AsyncSession)

        # Get metadata should cache it
        metadata = await session.get_metadata()
        assert isinstance(metadata, dict)
        assert session.metadata == metadata

        # Set metadata should update cache
        await session.set_metadata({"title": "Chat Session", "active": True})
        assert session.metadata == {"title": "Chat Session", "active": True}

        # Get should return cached value
        retrieved = await session.get_metadata()
        assert retrieved == {"title": "Chat Session", "active": True}
    else:
        assert isinstance(honcho_client, Honcho)
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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that session configuration is properly cached after get/set operations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-config-cache")
        assert isinstance(session, AsyncSession)

        # Get config should cache it
        config = await session.get_config()
        assert isinstance(config, dict)
        assert session.configuration == config

        # Set config should update cache
        await session.set_config({"anonymous": True, "summarize": False})
        assert session.configuration == {"anonymous": True, "summarize": False}

        # Get should return cached value
        retrieved = await session.get_config()
        assert retrieved == {"anonymous": True, "summarize": False}
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-config-cache")
        assert isinstance(session, Session)

        # Get config should cache it
        config = session.get_config()
        assert isinstance(config, dict)
        assert session.configuration == config

        # Set config should update cache
        session.set_config({"anonymous": True, "summarize": False})
        assert session.configuration == {"anonymous": True, "summarize": False}

        # Get should return cached value
        retrieved = session.get_config()
        assert retrieved == {"anonymous": True, "summarize": False}


@pytest.mark.asyncio
async def test_session_metadata_and_config_independence(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that session metadata and config are cached independently.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="test-session-independent-cache")
        assert isinstance(session, AsyncSession)

        # Set both metadata and config
        await session.set_metadata({"title": "Test"})
        await session.set_config({"anonymous": True})

        assert session.metadata == {"title": "Test"}
        assert session.configuration == {"anonymous": True}

        # Update metadata only
        await session.set_metadata({"title": "Updated"})

        assert session.metadata == {"title": "Updated"}
        assert session.configuration == {"anonymous": True}  # Should remain unchanged

        # Update config only
        await session.set_config({"anonymous": False})

        assert session.metadata == {"title": "Updated"}  # Should remain unchanged
        assert session.configuration == {"anonymous": False}
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="test-session-independent-cache")
        assert isinstance(session, Session)

        # Set both metadata and config
        session.set_metadata({"title": "Test"})
        session.set_config({"anonymous": True})

        assert session.metadata == {"title": "Test"}
        assert session.configuration == {"anonymous": True}

        # Update metadata only
        session.set_metadata({"title": "Updated"})

        assert session.metadata == {"title": "Updated"}
        assert session.configuration == {"anonymous": True}  # Should remain unchanged

        # Update config only
        session.set_config({"anonymous": False})

        assert session.metadata == {"title": "Updated"}  # Should remain unchanged
        assert session.configuration == {"anonymous": False}


@pytest.mark.asyncio
async def test_session_list_with_metadata_and_config(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that listed sessions have metadata and config populated from API response.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create sessions with metadata and config
        session1 = await honcho_client.session(id="test-list-session1")
        await session1.set_metadata({"title": "Session 1"})
        await session1.set_config({"anonymous": True})

        session2 = await honcho_client.session(id="test-list-session2")
        await session2.set_metadata({"title": "Session 2"})
        await session2.set_config({"anonymous": False})

        # List sessions and check cached data
        sessions_page = await honcho_client.get_sessions()
        assert isinstance(sessions_page, AsyncPage)

        sessions = sessions_page.items
        session_map = {s.id: s for s in sessions}

        if "test-list-session1" in session_map:
            s1 = session_map["test-list-session1"]
            assert s1.metadata == {"title": "Session 1"}
            assert s1.configuration == {"anonymous": True}

        if "test-list-session2" in session_map:
            s2 = session_map["test-list-session2"]
            assert s2.metadata == {"title": "Session 2"}
            assert s2.configuration == {"anonymous": False}
    else:
        assert isinstance(honcho_client, Honcho)

        # Create sessions with metadata and config
        session1 = honcho_client.session(id="test-list-session1")
        session1.set_metadata({"title": "Session 1"})
        session1.set_config({"anonymous": True})

        session2 = honcho_client.session(id="test-list-session2")
        session2.set_metadata({"title": "Session 2"})
        session2.set_config({"anonymous": False})

        # List sessions and check cached data
        sessions_page = honcho_client.get_sessions()
        assert isinstance(sessions_page, SyncPage)

        sessions = list(sessions_page)
        session_map = {s.id: s for s in sessions}

        if "test-list-session1" in session_map:
            s1 = session_map["test-list-session1"]
            assert s1.metadata == {"title": "Session 1"}
            assert s1.configuration == {"anonymous": True}

        if "test-list-session2" in session_map:
            s2 = session_map["test-list-session2"]
            assert s2.metadata == {"title": "Session 2"}
            assert s2.configuration == {"anonymous": False}


@pytest.mark.asyncio
async def test_peer_initialization_with_metadata_and_config(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that peers can be initialized with metadata and config.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        peer = await honcho_client.peer(
            id="test-init-peer",
            metadata={"name": "Test Peer", "role": "assistant"},
            config={"observe_me": False},
        )
        assert isinstance(peer, AsyncPeer)
        assert peer.metadata == {"name": "Test Peer", "role": "assistant"}
        assert peer.configuration == {"observe_me": False}
    else:
        assert isinstance(honcho_client, Honcho)

        peer = honcho_client.peer(
            id="test-init-peer",
            metadata={"name": "Test Peer", "role": "assistant"},
            config={"observe_me": False},
        )
        assert isinstance(peer, Peer)
        assert peer.metadata == {"name": "Test Peer", "role": "assistant"}
        assert peer.configuration == {"observe_me": False}


@pytest.mark.asyncio
async def test_session_initialization_with_metadata_and_config(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
) -> None:
    """
    Tests that sessions can be initialized with metadata and config.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        session = await honcho_client.session(
            id="test-init-session",
            metadata={"title": "Test Session", "tags": ["important"]},
            config={"anonymous": False},
        )
        assert isinstance(session, AsyncSession)
        assert session.metadata == {"title": "Test Session", "tags": ["important"]}
        assert session.configuration == {"anonymous": False}
    else:
        assert isinstance(honcho_client, Honcho)

        session = honcho_client.session(
            id="test-init-session",
            metadata={"title": "Test Session", "tags": ["important"]},
            config={"anonymous": False},
        )
        assert isinstance(session, Session)
        assert session.metadata == {"title": "Test Session", "tags": ["important"]}
        assert session.configuration == {"anonymous": False}
