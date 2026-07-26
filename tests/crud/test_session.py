import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.crud.session import (
    _get_or_add_peers_to_session,  # pyright: ignore[reportPrivateUsage]
)
from src.exceptions import ObserverException, ResourceNotFoundException


class TestSessionCRUD:
    """Test suite for session CRUD operations"""

    @pytest.mark.asyncio
    async def test_add_peers_counts_active_observer_stored_configuration(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An active observer remains counted when its submitted config disables it."""
        workspace, active_observer = sample_data
        new_observer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
        db_session.add_all([new_observer, session])
        await db_session.flush()
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 1)

        await db_session.execute(
            models.session_peers_table.insert().values(
                workspace_name=workspace.name,
                session_name=session.name,
                peer_name=active_observer.name,
                joined_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
                configuration={"observe_others": True},
            )
        )
        await db_session.flush()

        with pytest.raises(ObserverException):
            await _get_or_add_peers_to_session(
                db_session,
                workspace.name,
                session.name,
                {
                    active_observer.name: schemas.SessionPeerConfig(
                        observe_others=False
                    ),
                    new_observer.name: schemas.SessionPeerConfig(observe_others=True),
                },
            )

    @pytest.mark.asyncio
    async def test_add_peers_ignores_active_non_observer_submitted_configuration(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An active non-observer does not consume capacity from a submitted config."""
        workspace, active_non_observer = sample_data
        existing_observer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
        db_session.add_all([existing_observer, session])
        await db_session.flush()
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 1)

        await db_session.execute(
            models.session_peers_table.insert(),
            [
                {
                    "workspace_name": workspace.name,
                    "session_name": session.name,
                    "peer_name": active_non_observer.name,
                    "joined_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
                    "configuration": {"observe_others": False},
                },
                {
                    "workspace_name": workspace.name,
                    "session_name": session.name,
                    "peer_name": existing_observer.name,
                    "joined_at": datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
                    "configuration": {"observe_others": True},
                },
            ],
        )
        await db_session.flush()

        await _get_or_add_peers_to_session(
            db_session,
            workspace.name,
            session.name,
            {
                active_non_observer.name: schemas.SessionPeerConfig(observe_others=True),
            },
        )
        active_configuration = await db_session.scalar(
            select(models.SessionPeer.configuration).where(
                models.SessionPeer.workspace_name == workspace.name,
                models.SessionPeer.session_name == session.name,
                models.SessionPeer.peer_name == active_non_observer.name,
            )
        )
        assert active_configuration == {"observe_others": False}

    @pytest.mark.asyncio
    async def test_add_peers_preserves_active_members_and_rejoins_departed_members(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Active memberships are idempotent while departed memberships are rejoined."""
        workspace, active_peer = sample_data
        departed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=workspace.name
        )
        session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
        db_session.add_all([departed_peer, session])
        await db_session.flush()

        joined_at = datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC)
        left_at = joined_at + datetime.timedelta(days=1)
        active_config = schemas.SessionPeerConfig(observe_others=False, observe_me=True)
        departed_config = schemas.SessionPeerConfig(observe_others=True, observe_me=False)
        await db_session.execute(
            models.session_peers_table.insert(),
            [
                {
                    "workspace_name": workspace.name,
                    "session_name": session.name,
                    "peer_name": active_peer.name,
                    "joined_at": joined_at,
                    "left_at": None,
                    "configuration": active_config.model_dump(),
                },
                {
                    "workspace_name": workspace.name,
                    "session_name": session.name,
                    "peer_name": departed_peer.name,
                    "joined_at": joined_at,
                    "left_at": left_at,
                    "configuration": active_config.model_dump(),
                },
            ],
        )
        await db_session.flush()

        await _get_or_add_peers_to_session(
            db_session,
            workspace.name,
            session.name,
            {active_peer.name: departed_config, departed_peer.name: departed_config},
        )
        await db_session.flush()

        memberships = (
            await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.workspace_name == workspace.name,
                    models.SessionPeer.session_name == session.name,
                )
            )
        ).scalars()
        memberships_by_peer = {membership.peer_name: membership for membership in memberships}

        active_membership = memberships_by_peer[active_peer.name]
        assert active_membership.joined_at == joined_at
        assert active_membership.left_at is None
        assert active_membership.configuration == active_config.model_dump()

        departed_membership = memberships_by_peer[departed_peer.name]
        assert departed_membership.joined_at > left_at
        assert departed_membership.left_at is None
        assert departed_membership.configuration == departed_config.model_dump()

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
