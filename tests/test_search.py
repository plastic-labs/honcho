"""Tests for search functionality including peer knowledge search."""

import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
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
    await db_session.commit()

    # Search with peer_perspective filter
    results = await search(
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

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
    await db_session.commit()

    # Search with peer_perspective filter
    results = await search(
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

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
    await db_session.commit()

    # Search with peer_perspective filter
    results = await search(
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

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
    await db_session.commit()

    # Search with peer_perspective filter
    results = await search(
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

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
    await db_session.commit()

    # Search with peer_perspective filter for peer1 (not in any sessions)
    results = await search(
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
    await db_session.commit()

    # Search with peer_perspective filter
    results = await search(
        "Message",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )

    # peer1 should see both boundary messages (inclusive bounds)
    assert len(results) == 2
    assert msg_at_join.public_id in [m.public_id for m in results]
    assert msg_at_leave.public_id in [m.public_id for m in results]


# =============================================================================
# Tests for observer scoping in CRUD message functions
# =============================================================================


async def _setup_multi_session_workspace(db_session: AsyncSession):
    """Helper: create workspace with 2 sessions, 2 peers. peer1 only in session1."""
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    peer1 = models.Peer(name="observer", workspace_name=workspace.name)
    peer2 = models.Peer(name="other", workspace_name=workspace.name)
    db_session.add_all([peer1, peer2])
    await db_session.flush()

    session1 = models.Session(name="session_visible", workspace_name=workspace.name)
    session2 = models.Session(name="session_hidden", workspace_name=workspace.name)
    db_session.add_all([session1, session2])
    await db_session.flush()

    join_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=10
    )

    # peer1 is only in session1
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session1.name,
            peer_name=peer1.name,
            joined_at=join_time,
            left_at=None,
        )
    )
    # peer2 is in both sessions
    for s in [session1, session2]:
        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace.name,
                session_name=s.name,
                peer_name=peer2.name,
                joined_at=join_time,
                left_at=None,
            )
        )
    await db_session.flush()

    msg_visible = models.Message(
        content="visible message with keyword",
        session_name=session1.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=1),
    )
    msg_hidden = models.Message(
        content="hidden message with keyword",
        session_name=session2.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(seconds=2),
    )
    db_session.add_all([msg_visible, msg_hidden])
    await db_session.commit()

    return workspace, peer1, peer2, session1, session2, msg_visible, msg_hidden


@pytest.mark.asyncio
async def test_grep_messages_observer_scoping_excludes_non_member_sessions(
    db_session: AsyncSession,
):
    """grep_messages with observer excludes messages from sessions the observer isn't in."""
    (
        workspace,
        peer1,
        _,
        _,
        _,
        msg_visible,
        msg_hidden,
    ) = await _setup_multi_session_workspace(db_session)

    # Without scoping: both messages found
    results_unscoped = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=None,
        text="keyword",
    )
    all_matched_ids = [m.public_id for matches, _ in results_unscoped for m in matches]
    assert msg_visible.public_id in all_matched_ids
    assert msg_hidden.public_id in all_matched_ids

    # With observer scoping: only visible message found
    results_scoped = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=None,
        text="keyword",
        observer=peer1.name,
    )
    scoped_ids = [m.public_id for matches, _ in results_scoped for m in matches]
    assert msg_visible.public_id in scoped_ids
    assert msg_hidden.public_id not in scoped_ids


@pytest.mark.asyncio
async def test_get_messages_by_date_range_observer_scoping(
    db_session: AsyncSession,
):
    """get_messages_by_date_range with observer excludes non-member sessions."""
    (
        workspace,
        peer1,
        _,
        _,
        _,
        msg_visible,
        msg_hidden,
    ) = await _setup_multi_session_workspace(db_session)

    # Without scoping
    results_unscoped = await crud.get_messages_by_date_range(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
    )
    unscoped_ids = [m.public_id for m in results_unscoped]
    assert msg_visible.public_id in unscoped_ids
    assert msg_hidden.public_id in unscoped_ids

    # With observer scoping
    results_scoped = await crud.get_messages_by_date_range(
        db_session,
        workspace_name=workspace.name,
        session_name=None,
        observer=peer1.name,
    )
    scoped_ids = [m.public_id for m in results_scoped]
    assert msg_visible.public_id in scoped_ids
    assert msg_hidden.public_id not in scoped_ids


@pytest.mark.asyncio
async def test_grep_messages_observer_scoping_noop_when_session_provided(
    db_session: AsyncSession,
):
    """When session_name is provided, observer is ignored."""
    (
        workspace,
        peer1,
        _,
        session1,
        _,
        msg_visible,
        _,
    ) = await _setup_multi_session_workspace(db_session)

    results = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=session1.name,
        text="keyword",
        observer=peer1.name,
    )
    matched_ids = [m.public_id for matches, _ in results for m in matches]
    assert msg_visible.public_id in matched_ids


@pytest.mark.asyncio
async def test_grep_messages_observer_scoping_empty_when_no_sessions(
    db_session: AsyncSession,
):
    """Observer not in any sessions returns empty results."""
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    loner = models.Peer(name="loner", workspace_name=workspace.name)
    other = models.Peer(name="other", workspace_name=workspace.name)
    db_session.add_all([loner, other])
    await db_session.flush()

    session = models.Session(name="s1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=other.name,
            joined_at=datetime.datetime.now(datetime.timezone.utc),
            left_at=None,
        )
    )
    await db_session.flush()

    msg = models.Message(
        content="some keyword content",
        session_name=session.name,
        peer_name=other.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db_session.add(msg)
    await db_session.commit()

    results = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=None,
        text="keyword",
        observer=loner.name,
    )
    assert results == []


@pytest.mark.asyncio
async def test_grep_messages_observer_scoping_left_session_still_visible(
    db_session: AsyncSession,
):
    """Observer who left a session still sees all messages in that session.

    Any membership record (regardless of left_at) grants full session visibility.
    """
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()

    observer = models.Peer(name="obs", workspace_name=workspace.name)
    other = models.Peer(name="other", workspace_name=workspace.name)
    db_session.add_all([observer, other])
    await db_session.flush()

    session = models.Session(name="s1", workspace_name=workspace.name)
    db_session.add(session)
    await db_session.flush()

    base_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=10
    )
    join_time = base_time
    leave_time = base_time + datetime.timedelta(minutes=5)

    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=observer.name,
            joined_at=join_time,
            left_at=leave_time,
        )
    )
    await db_session.flush()

    # Message during membership
    msg_during = models.Message(
        content="keyword during",
        session_name=session.name,
        peer_name=other.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=join_time + datetime.timedelta(minutes=2),
    )
    # Message after observer left — still visible because any membership grants full access
    msg_after = models.Message(
        content="keyword after",
        session_name=session.name,
        peer_name=other.name,
        workspace_name=workspace.name,
        seq_in_session=2,
        created_at=leave_time + datetime.timedelta(minutes=1),
    )
    db_session.add_all([msg_during, msg_after])
    await db_session.commit()

    results = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=None,
        text="keyword",
        observer=observer.name,
    )
    matched_ids = [m.public_id for matches, _ in results for m in matches]
    assert msg_during.public_id in matched_ids
    assert msg_after.public_id in matched_ids
