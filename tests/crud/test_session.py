import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestSessionCRUD:
    """Test suite for session CRUD operations"""

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
