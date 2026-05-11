import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestSessionCRUD:
    """Test suite for session CRUD operations"""

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_created_at_desc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_sessions returns sessions in descending created_at order"""
        test_workspace, _ = sample_data

        # Create three sessions with explicit monotonic timestamps
        base_time = datetime.datetime.now(datetime.timezone.utc)
        session_ids = []
        for i in range(3):
            sid = str(generate_nanoid())
            session_ids.append(sid)
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time + datetime.timedelta(seconds=i),
            )
            db_session.add(session)
        await db_session.flush()

        # Query with descending sort
        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="created_at",
            sort_order="desc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Our three sessions should appear in reverse creation order
        assert returned_ids[:3] == session_ids[::-1]

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_created_at_asc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_sessions returns sessions in ascending created_at order"""
        test_workspace, _ = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)
        session_ids = []
        for i in range(3):
            sid = str(generate_nanoid())
            session_ids.append(sid)
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time + datetime.timedelta(seconds=i),
            )
            db_session.add(session)
        await db_session.flush()

        # Query with ascending sort
        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="created_at",
            sort_order="asc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        assert returned_ids[:3] == session_ids

    @pytest.mark.asyncio
    async def test_get_sessions_default_sort_is_asc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that get_sessions defaults to ascending created_at order"""
        test_workspace, _ = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)
        session_ids = []
        for i in range(3):
            sid = str(generate_nanoid())
            session_ids.append(sid)
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time + datetime.timedelta(seconds=i),
            )
            db_session.add(session)
        await db_session.flush()

        # Query with no sort params
        stmt = await crud.get_sessions(workspace_name=test_workspace.name)
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        assert returned_ids[:3] == session_ids

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_desc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_sessions sorts by most recent message timestamp descending"""
        test_workspace, test_peer = sample_data

        # Create three sessions all at the same time
        base_time = datetime.datetime.now(datetime.timezone.utc)
        session_ids = []
        for _i in range(3):
            sid = str(generate_nanoid())
            session_ids.append(sid)
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        # Add messages with different timestamps to each session.
        # Session 0 gets the oldest message, session 2 gets the newest.
        for i, sid in enumerate(session_ids):
            msg = models.Message(
                content=f"msg-{i}",
                session_name=sid,
                peer_name=test_peer.name,
                workspace_name=test_workspace.name,
                seq_in_session=1,
                created_at=base_time + datetime.timedelta(seconds=i + 10),
            )
            db_session.add(msg)
        await db_session.flush()

        # Query with descending sort by last_message_at
        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="desc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Session 2 (newest message) should come first
        assert returned_ids[:3] == session_ids[::-1]

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_asc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_sessions sorts by most recent message timestamp ascending"""
        test_workspace, test_peer = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)
        session_ids = []
        for _i in range(3):
            sid = str(generate_nanoid())
            session_ids.append(sid)
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        for i, sid in enumerate(session_ids):
            msg = models.Message(
                content=f"msg-{i}",
                session_name=sid,
                peer_name=test_peer.name,
                workspace_name=test_workspace.name,
                seq_in_session=1,
                created_at=base_time + datetime.timedelta(seconds=i + 10),
            )
            db_session.add(msg)
        await db_session.flush()

        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="asc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Session 0 (oldest message) should come first
        assert returned_ids[:3] == session_ids

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_no_messages(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test last_message_at sort puts sessions with no messages last (desc)"""
        test_workspace, test_peer = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)

        # Session A has a message, session B does not
        sid_a = str(generate_nanoid())
        sid_b = str(generate_nanoid())
        for sid in [sid_a, sid_b]:
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        msg = models.Message(
            content="hello",
            session_name=sid_a,
            peer_name=test_peer.name,
            workspace_name=test_workspace.name,
            seq_in_session=1,
            created_at=base_time + datetime.timedelta(seconds=5),
        )
        db_session.add(msg)
        await db_session.flush()

        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="desc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Session with a message should come before the one without
        assert returned_ids.index(sid_a) < returned_ids.index(sid_b)

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_nulls_last_asc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test last_message_at sort puts sessions with no messages last (asc)"""
        test_workspace, test_peer = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)

        # Session A has a message, session B does not
        sid_a = str(generate_nanoid())
        sid_b = str(generate_nanoid())
        for sid in [sid_a, sid_b]:
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        msg = models.Message(
            content="hello",
            session_name=sid_a,
            peer_name=test_peer.name,
            workspace_name=test_workspace.name,
            seq_in_session=1,
            created_at=base_time + datetime.timedelta(seconds=5),
        )
        db_session.add(msg)
        await db_session.flush()

        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="asc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Session with a message should come before the one without (NULLS LAST)
        assert returned_ids.index(sid_a) < returned_ids.index(sid_b)

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_mixed_nulls_desc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that messageless sessions sort last among sessions with messages (desc)"""
        test_workspace, test_peer = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)

        # Create 3 sessions: 2 with messages, 1 without
        sid_with_msg_1 = str(generate_nanoid())
        sid_with_msg_2 = str(generate_nanoid())
        sid_no_msg = str(generate_nanoid())
        for sid in [sid_with_msg_1, sid_with_msg_2, sid_no_msg]:
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        # Give the two sessions messages with different timestamps
        for i, sid in enumerate([sid_with_msg_1, sid_with_msg_2]):
            msg = models.Message(
                content=f"msg-{i}",
                session_name=sid,
                peer_name=test_peer.name,
                workspace_name=test_workspace.name,
                seq_in_session=1,
                created_at=base_time + datetime.timedelta(seconds=i + 10),
            )
            db_session.add(msg)
        await db_session.flush()

        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="desc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Descending: newest message first, no-message session last
        assert returned_ids[0] == sid_with_msg_2
        assert returned_ids[1] == sid_with_msg_1
        assert returned_ids[2] == sid_no_msg

    @pytest.mark.asyncio
    async def test_get_sessions_sort_by_last_message_at_mixed_nulls_asc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that messageless sessions sort last among sessions with messages (asc)"""
        test_workspace, test_peer = sample_data

        base_time = datetime.datetime.now(datetime.timezone.utc)

        # Create 3 sessions: 2 with messages, 1 without
        sid_with_msg_1 = str(generate_nanoid())
        sid_with_msg_2 = str(generate_nanoid())
        sid_no_msg = str(generate_nanoid())
        for sid in [sid_with_msg_1, sid_with_msg_2, sid_no_msg]:
            session = models.Session(
                name=sid,
                workspace_name=test_workspace.name,
                created_at=base_time,
            )
            db_session.add(session)
        await db_session.flush()

        for i, sid in enumerate([sid_with_msg_1, sid_with_msg_2]):
            msg = models.Message(
                content=f"msg-{i}",
                session_name=sid,
                peer_name=test_peer.name,
                workspace_name=test_workspace.name,
                seq_in_session=1,
                created_at=base_time + datetime.timedelta(seconds=i + 10),
            )
            db_session.add(msg)
        await db_session.flush()

        stmt = await crud.get_sessions(
            workspace_name=test_workspace.name,
            sort_by="last_message_at",
            sort_order="asc",
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        returned_ids = [s.name for s in sessions]

        # Ascending: oldest message first, no-message session last (NULLS LAST)
        assert returned_ids[0] == sid_with_msg_1
        assert returned_ids[1] == sid_with_msg_2
        assert returned_ids[2] == sid_no_msg

    @pytest.mark.asyncio
    async def test_get_session_peer_configuration(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving peer configuration data from session"""
        test_workspace, test_peer = sample_data

        # Create another peer
        peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(peer2)
        await db_session.flush()

        # Create session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        # Add peers to session with different configurations
        peer_configs = {
            test_peer.name: schemas.SessionPeerConfig(
                observe_others=True, observe_me=False
            ),
            peer2.name: schemas.SessionPeerConfig(
                observe_others=False, observe_me=True
            ),
        }

        # Set up peers in session
        await crud.set_peers_for_session(
            db_session,
            workspace_name=test_workspace.name,
            session_name=test_session.name,
            peer_names=peer_configs,
        )

        # Test the get_session_peer_configuration function
        stmt = await crud.get_session_peer_configuration(
            workspace_name=test_workspace.name,
            session_name=test_session.name,
        )
        result = await db_session.execute(stmt)
        configurations = result.all()

        # Should return configurations for all active peers
        assert len(configurations) == 2

        # Verify the structure of returned data
        for peer_name, peer_config, session_peer_config, is_active in configurations:
            assert isinstance(peer_name, str)
            assert isinstance(peer_config, dict) or peer_config is None
            assert isinstance(session_peer_config, dict)
            assert isinstance(is_active, bool)

            # Check that session_peer_config matches what we set
            expected_config = peer_configs[peer_name]
            assert (
                session_peer_config["observe_others"] == expected_config.observe_others
            )
            assert session_peer_config["observe_me"] == expected_config.observe_me

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, db_session: AsyncSession):
        """Test get_session with non-existent session raises ResourceNotFoundException"""
        with pytest.raises(ResourceNotFoundException):
            await crud.get_session(db_session, "nonexistent", "nonexistent_workspace")

    @pytest.mark.asyncio
    async def test_get_peer_config_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_peer_config with non-existent peer raises ResourceNotFoundException"""
        test_workspace, _test_peer = sample_data

        # Create session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        with pytest.raises(ResourceNotFoundException):
            await crud.get_peer_config(
                db_session, test_workspace.name, test_session.name, "nonexistent_peer"
            )

    @pytest.mark.asyncio
    async def test_clone_session_not_found(self, db_session: AsyncSession):
        """Test clone_session with non-existent session raises ResourceNotFoundException"""
        with pytest.raises(ResourceNotFoundException):
            await crud.clone_session(db_session, "workspace", "nonexistent_session")

    @pytest.mark.asyncio
    async def test_clone_session_invalid_cutoff_message(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test clone_session with invalid cutoff message raises ValueError"""
        test_workspace, _test_peer = sample_data

        # Create session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.flush()

        # Try to clone with invalid cutoff message ID
        with pytest.raises(
            ValueError,
            match="Message not found or doesn't belong to the specified session",
        ):
            await crud.clone_session(
                db_session, test_workspace.name, test_session.name, "invalid_message_id"
            )
