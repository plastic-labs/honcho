from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.operators import in_op

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

    @pytest.mark.asyncio
    async def test_delete_session_chunks_large_derived_conclusion_fanout(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The derived-conclusion id list is bound into an `id.in_(...)`
        predicate in chunks (crud/session.py's _ID_CHUNK_SIZE) rather than as
        one unbounded parameter list, so a session with a large derived
        closure can't build a single query with an unbounded number of bind
        parameters. Shrinks the chunk size instead of creating thousands of
        rows to exercise the multi-chunk path cheaply.
        """
        from src.crud import session as session_crud

        monkeypatch.setattr(session_crud, "_ID_CHUNK_SIZE", 2)

        # Spy on _batch_delete_matching (the function that actually issues
        # each chunked `id.in_(...)` delete - see the loop in delete_session)
        # so we can assert the chunking itself happened, not just the end
        # state. Asserting only remaining rows/counts would pass just as well
        # if someone reverted to one unbounded `id.in_(derived_ids)` delete.
        derived_id_chunk_sizes: list[int] = []
        original_batch_delete = (
            session_crud._batch_delete_matching  # pyright: ignore[reportPrivateUsage]
        )

        async def spy_batch_delete_matching(
            db: AsyncSession, model: Any, filter_conditions: list[Any], **kwargs: Any
        ) -> int:
            for condition in filter_conditions:
                if (
                    model is models.Document
                    and getattr(condition, "left", None) is not None
                    and str(condition.left) == "documents.id"
                    and condition.operator is in_op
                ):
                    derived_id_chunk_sizes.append(len(condition.right.value))
            return await original_batch_delete(db, model, filter_conditions, **kwargs)

        monkeypatch.setattr(
            session_crud, "_batch_delete_matching", spy_batch_delete_matching
        )

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

        # Five one-hop deductions resting directly on the session's explicit
        # doc - with _ID_CHUNK_SIZE monkeypatched to 2, deleting these
        # requires 3 chunked `id.in_(...)` deletes (2 + 2 + 1).
        derived_docs = [
            models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer.name,
                content=f"deduction {i} resting on this session's evidence",
                level="deductive",
                session_name=None,
                source_ids=[explicit_doc.id],
            )
            for i in range(5)
        ]
        db_session.add_all(derived_docs)
        await db_session.commit()

        derived_ids = [d.id for d in derived_docs]

        result = await crud.delete_session(
            db_session, test_workspace.name, test_session.name
        )

        remaining = await db_session.execute(
            select(models.Document.id).where(
                models.Document.workspace_name == test_workspace.name,
            )
        )
        remaining_ids = {row[0] for row in remaining.all()}

        assert explicit_doc.id not in remaining_ids
        for derived_id in derived_ids:
            assert derived_id not in remaining_ids
        # 1 explicit + 5 derived, deleted across the explicit call plus 3
        # chunked derived-id calls.
        assert result.conclusions_deleted == 6

        # The actual assertion this test exists for: the derived ids were
        # bound into 3 separate `id.in_(...)` deletes (chunk sizes 2, 2, 1),
        # not one unbounded `id.in_(derived_ids)` call. Without this, a
        # revert to the unbounded form would still pass every assertion
        # above (the end state - which rows are gone - is identical either
        # way).
        assert derived_id_chunk_sizes == [2, 2, 1]
        assert all(size <= 2 for size in derived_id_chunk_sizes)
