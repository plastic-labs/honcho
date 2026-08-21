import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
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

    @pytest.mark.asyncio
    async def test_delete_session_cascades_to_derived_conclusions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """delete_session must also remove Dreamer-produced deductive/inductive
        conclusions that rest on this session's evidence, even though those
        documents carry session_name=None by design (the session-purity
        invariant in crud/document.py:_dedup_key). Reproduces honcho#997's
        real, still-live gap: such conclusions are invisible to a cascade
        filtered only on Document.session_name and previously survived
        session deletion forever.
        """
        test_workspace, test_peer = sample_data

        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)

        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
        )
        db_session.add(collection)
        await db_session.flush()

        explicit_doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
            content="explicit fact from this session",
            level="explicit",
            session_name=test_session.name,
        )
        db_session.add(explicit_doc)
        await db_session.flush()

        # One-hop: a deduction resting directly on the session's explicit doc.
        deductive_doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
            content="deduction resting on this session's evidence",
            level="deductive",
            session_name=None,
            source_ids=[explicit_doc.id],
        )
        db_session.add(deductive_doc)
        await db_session.flush()

        # Two-hop: an induction resting on the deduction above, not on the
        # explicit doc directly - exercises the transitive frontier expansion
        # (an induction whose support left with a deleted session must leave
        # too, mirroring scope_backfill.py's process_scope_removal cascade).
        inductive_doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
            content="induction resting on the deduction above",
            level="inductive",
            session_name=None,
            source_ids=[deductive_doc.id],
        )
        db_session.add(inductive_doc)

        # An unrelated derived doc in the same collection, resting on
        # unrelated evidence - must survive the session deletion.
        unrelated_doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
            content="unrelated deduction",
            level="deductive",
            session_name=None,
            source_ids=["some-other-doc-id-not-in-this-session"],
        )
        db_session.add(unrelated_doc)
        await db_session.commit()

        explicit_id, deductive_id, inductive_id, unrelated_id = (
            explicit_doc.id,
            deductive_doc.id,
            inductive_doc.id,
            unrelated_doc.id,
        )

        await crud.delete_session(db_session, test_workspace.name, test_session.name)

        result = await db_session.execute(
            select(models.Document.id).where(
                models.Document.workspace_name == test_workspace.name,
            )
        )
        remaining_ids = {row[0] for row in result.all()}

        assert explicit_id not in remaining_ids
        assert deductive_id not in remaining_ids
        assert inductive_id not in remaining_ids
        assert unrelated_id in remaining_ids
