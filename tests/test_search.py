"""Tests for search functionality including peer knowledge search."""

import datetime
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
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
    join_time = datetime.datetime.now(datetime.UTC)
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
    join_time = datetime.datetime.now(datetime.UTC)
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
    base_time = datetime.datetime.now(datetime.UTC)
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
    join_time = datetime.datetime.now(datetime.UTC)
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
    join_time = datetime.datetime.now(datetime.UTC)
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
    join_time = datetime.datetime.now(datetime.UTC)
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

    join_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=10)

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
        _,
        session_hidden,
        _,
        msg_hidden,
    ) = await _setup_multi_session_workspace(db_session)

    results = await crud.grep_messages(
        workspace_name=workspace.name,
        session_name=session_hidden.name,
        text="keyword",
        observer=peer1.name,
    )
    matched_ids = [m.public_id for matches, _ in results for m in matches]
    assert msg_hidden.public_id in matched_ids


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
            joined_at=datetime.datetime.now(datetime.UTC),
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
        created_at=datetime.datetime.now(datetime.UTC),
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

    base_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=10)
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


@pytest.mark.asyncio
async def test_peer_perspective_search_after_active_readd(
    db_session: AsyncSession,
):
    """Active re-add keeps existing messages visible; a genuine rejoin starts a
    new window."""
    workspace = models.Workspace(name=generate_nanoid())
    peer1 = models.Peer(name="peer1", workspace_name=workspace.name)
    peer2 = models.Peer(name="peer2", workspace_name=workspace.name)
    session = models.Session(name="session1", workspace_name=workspace.name)
    db_session.add_all([workspace, peer1, peer2, session])
    await db_session.flush()

    past_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
    await db_session.execute(
        models.session_peers_table.insert().values(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer1.name,
            joined_at=past_time,
            left_at=None,
        )
    )
    msg_old = models.Message(
        content="old persistent message",
        session_name=session.name,
        peer_name=peer2.name,
        workspace_name=workspace.name,
        seq_in_session=1,
        created_at=past_time + datetime.timedelta(minutes=1),
    )
    db_session.add(msg_old)
    await db_session.commit()

    session_create = schemas.SessionCreate(
        name=session.name,
        peers={peer1.name: schemas.SessionPeerConfig()},
    )
    await crud.get_or_create_session(db_session, session_create, workspace.name)
    results = await search(
        "persistent",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    assert msg_old.public_id in [m.public_id for m in results]

    await db_session.execute(
        update(models.SessionPeer)
        .where(
            models.SessionPeer.session_name == session.name,
            models.SessionPeer.peer_name == peer1.name,
            models.SessionPeer.workspace_name == workspace.name,
        )
        .values(left_at=datetime.datetime.now(datetime.UTC))
    )
    await db_session.commit()

    await crud.get_or_create_session(db_session, session_create, workspace.name)
    results = await search(
        "persistent",
        filters={"peer_perspective": peer1.name, "workspace_id": workspace.name},
        limit=10,
    )
    assert msg_old.public_id not in [m.public_id for m in results]


def _axis_vector(axis: int, sign: float = 1.0) -> list[float]:
    from src.config import settings

    vector = [0.0] * settings.EMBEDDING.VECTOR_DIMENSIONS
    vector[axis] = sign
    return vector


async def _messages_with_embeddings(
    db_session: AsyncSession, contents_and_vectors: list[tuple[str, list[float]]]
) -> tuple[str, list[models.Message]]:
    workspace = models.Workspace(name=generate_nanoid())
    db_session.add(workspace)
    await db_session.flush()
    peer = models.Peer(name="author", workspace_name=workspace.name)
    session = models.Session(name="s1", workspace_name=workspace.name)
    db_session.add_all([peer, session])
    await db_session.flush()

    messages = [
        models.Message(
            content=content,
            session_name=session.name,
            peer_name=peer.name,
            workspace_name=workspace.name,
            seq_in_session=seq,
        )
        for seq, (content, _) in enumerate(contents_and_vectors, 1)
    ]
    db_session.add_all(messages)
    await db_session.flush()
    db_session.add_all(
        [
            models.MessageEmbedding(
                content=message.content,
                message_id=message.public_id,
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=peer.name,
                sync_state="synced",
                embedding=vector,
            )
            for message, (_, vector) in zip(messages, contents_and_vectors, strict=True)
        ]
    )
    await db_session.commit()
    return workspace.name, messages


@pytest.mark.asyncio
async def test_max_distance_drops_semantic_hits_beyond_the_floor(
    db_session: AsyncSession, mock_openai_embeddings: dict[str, Any]
):
    """A semantic hit farther than max_distance is not returned (#1177)."""
    workspace_name, (near, far) = await _messages_with_embeddings(
        db_session,
        [("lunch plans", _axis_vector(0)), ("tax forms", _axis_vector(0, -1.0))],
    )
    mock_openai_embeddings["embed"].side_effect = lambda _q: _axis_vector(0)  # pyright: ignore[reportUnknownLambdaType]
    filters = {"workspace_id": workspace_name}

    unbounded = await search("zzqx", filters=filters, limit=10)
    bounded = await search("zzqx", filters=filters, limit=10, max_distance=0.5)

    assert {m.public_id for m in unbounded} == {near.public_id, far.public_id}
    assert [m.public_id for m in bounded] == [near.public_id]


@pytest.mark.asyncio
async def test_max_distance_returns_nothing_when_no_hit_clears_it(
    db_session: AsyncSession, mock_openai_embeddings: dict[str, Any]
):
    """An unrelated query returns [] instead of `limit` arbitrary messages (#1177)."""
    workspace_name, _ = await _messages_with_embeddings(
        db_session,
        [("lunch plans", _axis_vector(0)), ("tax forms", _axis_vector(1))],
    )
    mock_openai_embeddings["embed"].side_effect = lambda _q: _axis_vector(2)  # pyright: ignore[reportUnknownLambdaType]

    results = await search(
        "flumbex zorplat quibnar",
        filters={"workspace_id": workspace_name},
        limit=5,
        max_distance=0.5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_max_distance_keeps_full_text_matches(
    db_session: AsyncSession, mock_openai_embeddings: dict[str, Any]
):
    """The floor applies to the semantic leg only; a lexical match is still returned."""
    workspace_name, (_, far) = await _messages_with_embeddings(
        db_session,
        [("lunch plans", _axis_vector(0)), ("tax forms", _axis_vector(0, -1.0))],
    )
    mock_openai_embeddings["embed"].side_effect = lambda _q: _axis_vector(0)  # pyright: ignore[reportUnknownLambdaType]

    results = await search(
        "tax", filters={"workspace_id": workspace_name}, limit=10, max_distance=0.5
    )

    assert far.public_id in {m.public_id for m in results}


@pytest.mark.parametrize("max_distance", [-0.1, 2.1])
def test_max_distance_outside_cosine_range_is_rejected(max_distance: float):
    with pytest.raises(ValueError):
        schemas.MessageSearchOptions(query="q", max_distance=max_distance)


@pytest.mark.asyncio
async def test_max_distance_is_forwarded_to_the_external_vector_store(
    monkeypatch: pytest.MonkeyPatch,
):
    from src.utils import search as search_module
    from src.vector_store import VectorQueryResult

    received: dict[str, Any] = {}

    class _RecordingStore:
        def get_vector_namespace(self, kind: str, workspace_name: str) -> str:
            return f"{kind}:{workspace_name}"

        async def query(
            self, _namespace: str, _embedding: list[float], **kwargs: Any
        ) -> list[VectorQueryResult]:
            received.update(kwargs)
            return [
                VectorQueryResult(
                    id="chunk-1", score=0.1, metadata={"message_id": "m1"}
                )
            ]

    monkeypatch.setattr(
        search_module, "get_external_vector_store", lambda: _RecordingStore()
    )

    ids = await search_module.query_external_vector_message_ids(
        workspace_name="ws", embedding_query=[0.0], limit=5, max_distance=0.4
    )

    assert ids == ["m1"]
    assert received["max_distance"] == 0.4


class _DistanceStore:
    """External vector store stand-in that ranks fixed per-message distances."""

    def __init__(self, distances: dict[str, float]) -> None:
        self.distances: dict[str, float] = distances

    def get_vector_namespace(self, kind: str, workspace_name: str) -> str:
        return f"{kind}:{workspace_name}"

    async def query(
        self, _namespace: str, _embedding: list[float], **kwargs: Any
    ) -> list[Any]:
        from src.vector_store import VectorQueryResult

        max_distance: float | None = kwargs.get("max_distance")
        return [
            VectorQueryResult(
                id=f"chunk-{message_id}",
                score=distance,
                metadata={"message_id": message_id},
            )
            for message_id, distance in sorted(
                self.distances.items(), key=lambda item: item[1]
            )
            if max_distance is None or distance <= max_distance
        ]


@pytest.mark.asyncio
async def test_max_distance_on_the_external_vector_store_path(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    """Omitted, a relevant hit kept, an unrelated query empty: the external-store path (#1177)."""
    from src.utils import search as search_module

    workspace_name, (near, far) = await _messages_with_embeddings(
        db_session,
        [("lunch plans", _axis_vector(0)), ("tax forms", _axis_vector(1))],
    )
    monkeypatch.setattr(search_module, "_uses_pgvector_message_search", lambda: False)
    filters = {"workspace_id": workspace_name}

    store = _DistanceStore({near.public_id: 0.2, far.public_id: 0.9})
    monkeypatch.setattr(search_module, "get_external_vector_store", lambda: store)
    omitted = await search("zzqx", filters=filters, limit=10)
    bounded = await search("zzqx", filters=filters, limit=10, max_distance=0.5)

    store.distances = {near.public_id: 0.95, far.public_id: 0.9}
    unrelated = await search("zzqx", filters=filters, limit=10, max_distance=0.5)

    assert {m.public_id for m in omitted} == {near.public_id, far.public_id}
    assert [m.public_id for m in bounded] == [near.public_id]
    assert unrelated == []
