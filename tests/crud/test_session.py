from datetime import datetime, timezone

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import Boolean, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.exceptions import ObserverException, ResourceNotFoundException


class TestSessionCRUD:
    """Test suite for session CRUD operations"""

    @pytest.mark.asyncio
    async def test_get_or_create_session_preserves_active_joined_at(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Active re-adds keep joined_at and config; a genuine rejoin starts a
        new window."""
        test_workspace, test_peer = sample_data
        session_name = str(generate_nanoid())
        original_config = schemas.SessionPeerConfig(
            observe_others=True, observe_me=False
        )
        updated_config = schemas.SessionPeerConfig(
            observe_others=False, observe_me=True
        )
        session_peer_stmt = select(
            models.SessionPeer.joined_at,
            models.SessionPeer.left_at,
            models.SessionPeer.configuration,
        ).where(
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.peer_name == test_peer.name,
            models.SessionPeer.workspace_name == test_workspace.name,
        )

        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name, peers={test_peer.name: original_config}
            ),
            test_workspace.name,
        )
        first_joined_at, first_left_at, first_config = (
            await db_session.execute(session_peer_stmt)
        ).one()
        assert first_left_at is None
        assert first_config == original_config.model_dump()

        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name, peers={test_peer.name: updated_config}
            ),
            test_workspace.name,
        )
        second_joined_at, second_left_at, second_config = (
            await db_session.execute(session_peer_stmt)
        ).one()
        assert second_joined_at == first_joined_at
        assert second_left_at is None
        assert second_config == original_config.model_dump()

        session_peer = (
            await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.session_name == session_name,
                    models.SessionPeer.peer_name == test_peer.name,
                    models.SessionPeer.workspace_name == test_workspace.name,
                )
            )
        ).scalar_one()
        session_peer.left_at = datetime.now(timezone.utc)
        await db_session.commit()

        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name, peers={test_peer.name: updated_config}
            ),
            test_workspace.name,
        )
        rejoined_joined_at, rejoined_left_at, rejoined_config = (
            await db_session.execute(session_peer_stmt)
        ).one()
        assert rejoined_joined_at > second_joined_at
        assert rejoined_left_at is None
        assert rejoined_config == updated_config.model_dump()

    @pytest.mark.asyncio
    async def test_set_peers_preserves_active_joined_at(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """PUT-peers keeps active membership windows and refreshes real rejoins."""
        test_workspace, test_peer = sample_data
        session_name = str(generate_nanoid())
        original_config = schemas.SessionPeerConfig(
            observe_others=True, observe_me=False
        )
        updated_config = schemas.SessionPeerConfig(
            observe_others=False, observe_me=True
        )
        db_session.add(
            models.Session(name=session_name, workspace_name=test_workspace.name)
        )
        await db_session.flush()

        session_peer_stmt = select(
            models.SessionPeer.joined_at,
            models.SessionPeer.left_at,
            models.SessionPeer.configuration,
        ).where(
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.peer_name == test_peer.name,
            models.SessionPeer.workspace_name == test_workspace.name,
        )

        await crud.set_peers_for_session(
            db_session,
            workspace_name=test_workspace.name,
            session_name=session_name,
            peer_names={test_peer.name: original_config},
        )
        first_left_at, first_config = (
            await db_session.execute(
                select(
                    models.SessionPeer.left_at,
                    models.SessionPeer.configuration,
                ).where(
                    models.SessionPeer.session_name == session_name,
                    models.SessionPeer.peer_name == test_peer.name,
                    models.SessionPeer.workspace_name == test_workspace.name,
                )
            )
        ).one()
        assert first_left_at is None
        assert first_config == original_config.model_dump()

        session_peer = (
            await db_session.execute(
                select(models.SessionPeer).where(
                    models.SessionPeer.session_name == session_name,
                    models.SessionPeer.peer_name == test_peer.name,
                    models.SessionPeer.workspace_name == test_workspace.name,
                )
            )
        ).scalar_one()
        session_peer.joined_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        await db_session.commit()

        await crud.set_peers_for_session(
            db_session,
            workspace_name=test_workspace.name,
            session_name=session_name,
            peer_names={test_peer.name: updated_config},
        )
        active_joined_at, active_left_at, active_config = (
            await db_session.execute(session_peer_stmt)
        ).one()
        assert active_joined_at == datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert active_left_at is None
        # A replace states the desired end state, so the incoming config lands even
        # though the membership window is untouched.
        assert active_config == updated_config.model_dump()

        await crud.set_peers_for_session(
            db_session,
            workspace_name=test_workspace.name,
            session_name=session_name,
            peer_names={},
        )
        await crud.set_peers_for_session(
            db_session,
            workspace_name=test_workspace.name,
            session_name=session_name,
            peer_names={test_peer.name: updated_config},
        )
        rejoined_joined_at, rejoined_left_at, rejoined_config = (
            await db_session.execute(session_peer_stmt)
        ).one()
        assert rejoined_joined_at > active_joined_at
        assert rejoined_left_at is None
        assert rejoined_config == updated_config.model_dump()

    @pytest.mark.asyncio
    async def test_observer_limit_counts_preserved_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """On the add path an already-active observer keeps its stored config, so
        it still counts against the limit when re-sent as a non-observer."""
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 2)
        test_workspace, _ = sample_data
        # Bound to a local: rollback below expires the ORM instance, and reloading
        # it would lazy-load outside the greenlet context.
        workspace_name = test_workspace.name
        session_name = str(generate_nanoid())
        observer = schemas.SessionPeerConfig(observe_others=True, observe_me=False)
        bystander = schemas.SessionPeerConfig(observe_others=False, observe_me=True)
        existing = [str(generate_nanoid()) for _ in range(2)]

        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name, peers=dict.fromkeys(existing, observer)
            ),
            workspace_name,
        )

        # Adding cannot demote an active member, so re-sending the two observers as
        # non-observers leaves them observing and the third peer makes three.
        with pytest.raises(ObserverException):
            await crud.get_or_create_session(
                db_session,
                schemas.SessionCreate(
                    name=session_name,
                    peers={
                        **dict.fromkeys(existing, bystander),
                        str(generate_nanoid()): observer,
                    },
                ),
                workspace_name,
            )

        # The rejected request left nothing behind.
        await db_session.rollback()
        observer_count = await db_session.scalar(
            select(func.count()).where(
                models.SessionPeer.session_name == session_name,
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.left_at.is_(None),
                models.SessionPeer.configuration["observe_others"].astext.cast(Boolean),
            )
        )
        assert observer_count == 2

    @pytest.mark.asyncio
    async def test_set_peers_observer_limit_counts_replaced_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The replace path applies the incoming config, so demoting active
        observers frees room under the limit in the same request."""
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 2)
        test_workspace, _ = sample_data
        workspace_name = test_workspace.name
        session_name = str(generate_nanoid())
        db_session.add(models.Session(name=session_name, workspace_name=workspace_name))
        await db_session.flush()
        observer = schemas.SessionPeerConfig(observe_others=True, observe_me=False)
        bystander = schemas.SessionPeerConfig(observe_others=False, observe_me=True)
        existing = [str(generate_nanoid()) for _ in range(2)]

        await crud.set_peers_for_session(
            db_session,
            workspace_name=workspace_name,
            session_name=session_name,
            peer_names=dict.fromkeys(existing, observer),
        )

        # Demoting both active observers while adding a new one leaves exactly one.
        await crud.set_peers_for_session(
            db_session,
            workspace_name=workspace_name,
            session_name=session_name,
            peer_names={
                **dict.fromkeys(existing, bystander),
                str(generate_nanoid()): observer,
            },
        )
        observer_count = await db_session.scalar(
            select(func.count()).where(
                models.SessionPeer.session_name == session_name,
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.left_at.is_(None),
                models.SessionPeer.configuration["observe_others"].astext.cast(Boolean),
            )
        )
        assert observer_count == 1

    @pytest.mark.asyncio
    async def test_observer_limit_lets_over_limit_session_take_non_observers(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A session already past the limit still accepts non-observers, so
        sessions that grew over it before enforcement do not become unusable."""
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 2)
        test_workspace, _ = sample_data
        workspace_name = test_workspace.name
        session_name = str(generate_nanoid())
        observer = schemas.SessionPeerConfig(observe_others=True, observe_me=False)
        bystander = schemas.SessionPeerConfig(observe_others=False, observe_me=True)

        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name,
                peers=dict.fromkeys(
                    [str(generate_nanoid()) for _ in range(2)], observer
                ),
            ),
            workspace_name,
        )

        # Now the limit is below what the session already holds.
        monkeypatch.setattr(settings, "SESSION_OBSERVERS_LIMIT", 1)
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=session_name, peers={str(generate_nanoid()): bystander}
            ),
            workspace_name,
        )

        active_count = await db_session.scalar(
            select(func.count()).where(
                models.SessionPeer.session_name == session_name,
                models.SessionPeer.workspace_name == workspace_name,
                models.SessionPeer.left_at.is_(None),
            )
        )
        assert active_count == 3

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
