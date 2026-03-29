"""Tests for search functionality including peer knowledge search."""

import datetime
from typing import cast

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.utils.search import search


@pytest.mark.asyncio
async def test_peer_perspective_search_single_session(
    db_session: AsyncSession,
):
    """Test that peer_perspective filter returns messages from a single session the peer is in."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create session
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Add peer1 to session
    join_time = datetime.datetime.now(datetime.timezone.utc)
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer1.name,
            joined_at=join_time,
            left_at=None,
        )
    )
    await db_session.flush()

    # Create messages in session (sent by peer2)
    msg1 = models.Message(
        content="Message 1",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=1),
    )
    msg2 = models.Message(
        content="Message 2",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=2,
        created_at=join_time + datetime.timedelta(seconds=2),
    )
    db_session.add_all([msg1, msg2])
    await db_session.flush()

    # Search with peer_perspective filter
    result = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    # context_window=0 (default) returns flat list
    results = cast(list[models.Message], result)

    # peer1 should see both messages
    assert len(results) == 2
    assert msg1.public_id in [m.public_id for m in results]
    assert msg2.public_id in [m.public_id for m in results]


@pytest.mark.asyncio
async def test_peer_perspective_search_multiple_sessions(
    db_session: AsyncSession,
):
    """Test that peer_perspective filter returns messages from all sessions the peer is in."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create two sessions
    session1 = models.Session(name="session1", workspace_name=workspace.name)
    session2 = models.Session(name="session2", workspace_name=workspace.name)
    db_session.add_all([session1, session2])
    await db_session.flush()

    # Add peer1 to both sessions
    join_time = datetime.datetime.now(datetime.timezone.utc)
    for session in [session1, session2]:
        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer1.name,
                joined_at=join_time,
                left_at=None,
            )
        )
    await db_session.flush()

    # Create messages in both sessions
    msg1 = models.Message(
        content="Message in session 1",
        session_name=session1.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=1),
    )
    msg2 = models.Message(
        content="Message in session 2",
        session_name=session2.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=2),
    )
    db_session.add_all([msg1, msg2])
    await db_session.flush()

    # Search with peer_perspective filter
    result = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    results = cast(list[models.Message], result)

    # peer1 should see messages from both sessions
    assert len(results) == 2
    assert msg1.public_id in [m.public_id for m in results]
    assert msg2.public_id in [m.public_id for m in results]


@pytest.mark.asyncio
async def test_peer_perspective_search_temporal_constraints(
    db_session: AsyncSession,
):
    """Test that peer_perspective filter respects joined_at and left_at timestamps."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create session
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Define time windows
    base_time = datetime.datetime.now(datetime.timezone.utc)
    join_time = base_time + datetime.timedelta(seconds=10)
    leave_time = base_time + datetime.timedelta(seconds=20)

    # Add peer1 to session with specific join/leave times
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer1.name,
            joined_at=join_time,
            left_at=leave_time,
        )
    )
    await db_session.flush()

    # Create messages: before join, during participation, after leave
    msg_before = models.Message(
        content="Message before join",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time - datetime.timedelta(seconds=1),
    )
    msg_during = models.Message(
        content="Message during participation",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=2,
        created_at=join_time + datetime.timedelta(seconds=5),
    )
    msg_after = models.Message(
        content="Message after leave",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=3,
        created_at=leave_time + datetime.timedelta(seconds=1),
    )
    db_session.add_all([msg_before, msg_during, msg_after])
    await db_session.flush()

    # Search with peer_perspective filter
    result = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    results = cast(list[models.Message], result)

    # peer1 should only see the message during their participation
    assert len(results) == 1
    assert results[0].public_id == msg_during.public_id


@pytest.mark.asyncio
async def test_peer_perspective_search_active_member(
    db_session: AsyncSession,
):
    """Test that peer_perspective filter works for active members (left_at is NULL)."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create session
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Add peer1 to session (still active, left_at is NULL)
    join_time = datetime.datetime.now(datetime.timezone.utc)
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer1.name,
            joined_at=join_time,
            left_at=None,  # Still active
        )
    )
    await db_session.flush()

    # Create messages after join time
    msg1 = models.Message(
        content="Message 1",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=1),
    )
    msg2 = models.Message(
        content="Message 2",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=2,
        created_at=join_time + datetime.timedelta(seconds=100),
    )
    db_session.add_all([msg1, msg2])
    await db_session.flush()

    # Search with peer_perspective filter
    result = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    results = cast(list[models.Message], result)

    # peer1 should see all messages after join time (no left_at limit)
    assert len(results) == 2
    assert msg1.public_id in [m.public_id for m in results]
    assert msg2.public_id in [m.public_id for m in results]


@pytest.mark.asyncio
async def test_peer_perspective_search_no_sessions(
    db_session: AsyncSession,
):
    """Test that peer_perspective filter returns empty list for peer not in any sessions."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create session but don't add peer1
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Add peer2 to session
    join_time = datetime.datetime.now(datetime.timezone.utc)
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer2.name,
            joined_at=join_time,
            left_at=None,
        )
    )
    await db_session.flush()

    # Create message in session
    msg = models.Message(
        content="Message",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=1),
    )
    db_session.add(msg)
    await db_session.flush()

    # Search with peer_perspective filter for peer1 (not in any sessions)
    results = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

    # peer1 should see no messages
    assert len(results) == 0


@pytest.mark.asyncio
async def test_peer_perspective_search_boundary_timestamps(
    db_session: AsyncSession,
):
    """Test that messages at exact joined_at and left_at timestamps are included."""
    # Create workspace
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    # Create peers
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    # Create session
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    # Define exact timestamps
    join_time = datetime.datetime.now(datetime.timezone.utc)
    leave_time = join_time + datetime.timedelta(seconds=10)

    # Add peer1 to session
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer1.name,
            joined_at=join_time,
            left_at=leave_time,
        )
    )
    await db_session.flush()

    # Create messages at exact boundary times
    msg_at_join = models.Message(
        content="Message at join",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time,  # Exact join time
    )
    msg_at_leave = models.Message(
        content="Message at leave",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=2,
        created_at=leave_time,  # Exact leave time
    )
    db_session.add_all([msg_at_join, msg_at_leave])
    await db_session.flush()

    # Search with peer_perspective filter
    result = await search(
        db_session,
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    results = cast(list[models.Message], result)

    # peer1 should see both boundary messages (inclusive bounds)
    assert len(results) == 2
    assert msg_at_join.public_id in [m.public_id for m in results]
    assert msg_at_leave.public_id in [m.public_id for m in results]
