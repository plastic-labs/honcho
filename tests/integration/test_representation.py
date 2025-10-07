"""
Integration tests for the Representation class and related workflows.

This test suite covers the full representation workflow including:
- Document creation with embedding and duplicate detection
- Representation building from documents
- Representation merging and diffing operations
- Working representation retrieval with different strategies
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.utils.representation import (
    DeductiveObservation,
    DeductiveObservationBase,
    ExplicitObservation,
    ExplicitObservationBase,
    PromptRepresentation,
    Representation,
)


@pytest.fixture
def fixed_embedding_vector() -> list[float]:
    """
    Fixture providing a deterministic embedding vector for hermetic tests.

    Returns a 1536-dimensional vector with predictable values to avoid
    network calls to external embedding services during testing.
    """
    # Create a deterministic 1536-dimensional embedding vector
    # Using a simple pattern that's easy to verify in tests
    return [0.1 * (i % 10) for i in range(1536)]


@pytest.mark.asyncio
class TestRepresentationWorkflow:
    """Test suite for the complete representation workflow"""

    async def create_test_workspace_and_peer(
        self,
        db_session: AsyncSession,
        workspace_name: str | None = None,
        peer_name: str | None = None,
    ) -> tuple[models.Workspace, models.Peer]:
        """Helper to create test workspace and peer"""
        workspace_name = workspace_name or generate_nanoid()
        peer_name = peer_name or generate_nanoid()

        workspace = models.Workspace(name=workspace_name)
        db_session.add(workspace)
        await db_session.flush()

        peer = models.Peer(name=peer_name, workspace_name=workspace_name)
        db_session.add(peer)
        await db_session.flush()

        return workspace, peer

    async def create_test_session(
        self,
        db_session: AsyncSession,
        workspace: models.Workspace,
        session_name: str | None = None,
    ) -> models.Session:
        """Helper to create test session"""
        session_name = session_name or generate_nanoid()

        session = models.Session(
            name=session_name,
            workspace_name=workspace.name,
        )
        db_session.add(session)
        await db_session.flush()

        return session

    async def test_representation_class_creation_and_operations(self):
        """Test basic Representation class operations"""
        # Create explicit observations
        explicit_obs1 = ExplicitObservation(
            content="User likes dogs",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="test_session",
        )
        explicit_obs2 = ExplicitObservation(
            content="User has a pet named Rover",
            created_at=datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            message_ids=[(2, 2)],
            session_name="test_session",
        )

        # Create deductive observations
        deductive_obs1 = DeductiveObservation(
            conclusion="User probably has a dog named Rover",
            premises=["User likes dogs", "User has a pet named Rover"],
            created_at=datetime(2025, 1, 1, 12, 2, 0, tzinfo=timezone.utc),
            message_ids=[(3, 3)],
            session_name="test_session",
        )

        # Create representation
        representation = Representation(
            explicit=[explicit_obs1, explicit_obs2], deductive=[deductive_obs1]
        )

        # Test basic properties
        assert not representation.is_empty()
        assert len(representation.explicit) == 2
        assert len(representation.deductive) == 1

        # Test string formatting
        str_output = str(representation)
        assert "EXPLICIT:" in str_output
        assert "DEDUCTIVE:" in str_output
        assert "User likes dogs" in str_output
        assert "User probably has a dog named Rover" in str_output

        # Test no-timestamp formatting
        no_timestamp_output = representation.str_no_timestamps()
        assert "User likes dogs" in no_timestamp_output
        assert "[" not in no_timestamp_output

        # Test markdown formatting
        markdown_output = representation.format_as_markdown()
        assert "## Explicit Observations" in markdown_output
        assert "## Deductive Observations" in markdown_output
        assert "**Conclusion**:" in markdown_output
        assert "**Premises**:" in markdown_output

    async def test_representation_merging_and_diffing(self):
        """Test representation merge and diff operations"""
        # Create first representation
        rep1 = Representation(
            explicit=[
                ExplicitObservation(
                    content="User likes cats",
                    created_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="session1",
                )
            ]
        )

        # Create second representation with some overlap
        rep2 = Representation(
            explicit=[
                ExplicitObservation(
                    content="User likes cats",  # Duplicate
                    created_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="session1",
                ),
                ExplicitObservation(
                    content="User likes dogs",  # New
                    created_at=datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(2, 2)],
                    session_name="session1",
                ),
            ]
        )

        # Test diff - should only return new observations
        diff = rep1.diff_representation(rep2)
        assert len(diff.explicit) == 1
        assert diff.explicit[0].content == "User likes dogs"

        # Test merge - should deduplicate and preserve order
        rep1.merge_representation(rep2)
        assert len(rep1.explicit) == 2
        # Should be sorted by created_at
        assert rep1.explicit[0].content == "User likes cats"
        assert rep1.explicit[1].content == "User likes dogs"

        # Test merge with max_observations limit
        rep3 = Representation(
            explicit=[
                ExplicitObservation(
                    content="User likes birds",
                    created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(3, 3)],
                    session_name="session1",
                )
            ]
        )

        rep1.merge_representation(rep3, max_observations=2)
        assert len(rep1.explicit) == 2
        # Should keep the most recent observations (FIFO)
        assert rep1.explicit[0].content == "User likes dogs"
        assert rep1.explicit[1].content == "User likes birds"


@pytest.mark.asyncio
class TestDocumentCreationWorkflow:
    """Test document creation with embedding and duplicate detection"""

    async def test_get_working_representation_with_semantic_query(
        self, db_session: AsyncSession
    ):
        """Test working representation retrieval with semantic query"""
        workspace, observer_peer = await self.create_test_workspace_and_peer(db_session)
        _, observed_peer = await self.create_test_workspace_and_peer(
            db_session, workspace.name
        )

        # Mock semantic search to return specific documents
        from src.models import Document

        mock_document = Document(
            id="test_doc_id",
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="User likes dogs",
            session_name="test_session",
            internal_metadata={
                "level": "explicit",
                "message_ids": [(1, 1)],
                "session_name": "test_session",
            },
            created_at=datetime.now(timezone.utc),
        )

        with patch(
            "src.crud.representation.RepresentationManager._query_documents_semantic"
        ) as mock_semantic:
            mock_semantic.return_value = [mock_document]

            representation = await crud.get_working_representation(
                workspace.name,
                include_semantic_query="pets dogs animals",
                observer=observer_peer.name,
                observed=observed_peer.name,
            )

            # Should have called semantic search
            mock_semantic.assert_called_once()
            assert len(representation.explicit) == 1
            assert representation.explicit[0].content == "User likes dogs"

    async def test_get_working_representation_with_most_derived(
        self, db_session: AsyncSession
    ):
        """Test working representation retrieval prioritizing most derived observations"""
        workspace, observer_peer = await self.create_test_workspace_and_peer(db_session)
        _, observed_peer = await self.create_test_workspace_and_peer(
            db_session, workspace.name
        )

        session = await self.create_test_session(db_session, workspace)

        # Create collection first - need to do it directly since mock doesn't persist to DB
        collection = models.Collection(
            observer=observer_peer.name,
            observed=observed_peer.name,
            workspace_name=workspace.name,
        )
        db_session.add(collection)
        await db_session.flush()

        # Create document with high times_derived count
        highly_derived_doc = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="Highly derived observation",
            session_name=session.name,
            internal_metadata={"level": "explicit", "times_derived": 5},
            embedding=[0.1] * 1536,
        )
        db_session.add(highly_derived_doc)

        # Create document with lower times_derived count
        less_derived_doc = models.Document(
            workspace_name=workspace.name,
            observer=observer_peer.name,
            observed=observed_peer.name,
            content="Less derived observation",
            session_name=session.name,
            internal_metadata={"level": "explicit", "times_derived": 2},
            embedding=[0.2] * 1536,
        )
        db_session.add(less_derived_doc)

        await db_session.commit()

        # Retrieve with most_derived=True
        representation = await crud.get_working_representation(
            workspace.name,
            include_most_derived=True,
            observer=observer_peer.name,
            observed=observed_peer.name,
        )

        # Should prioritize highly derived observation
        assert len(representation.explicit) >= 1
        # The highly derived observation should be included
        contents = [obs.content for obs in representation.explicit]
        assert "Highly derived observation" in contents

    async def test_representation_from_documents(self):
        """Test converting documents to representation"""
        # Create test documents
        explicit_doc = models.Document(
            workspace_name="test_workspace",
            observer="test_peer",
            observed="test_peer",
            content="User said they like programming",
            internal_metadata={
                "level": "explicit",
                "message_ids": [(1, 1)],
            },
            session_name="test_session",
            embedding=[0.1] * 1536,
            created_at=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )

        deductive_doc = models.Document(
            workspace_name="test_workspace",
            observer="test_peer",
            observed="test_peer",
            content="User is likely a software developer",
            internal_metadata={
                "level": "deductive",
                "message_ids": [(1, 1)],
                "premises": ["User said they like programming"],
            },
            session_name="test_session",
            embedding=[0.2] * 1536,
            created_at=datetime(2025, 1, 1, 10, 1, 0, tzinfo=timezone.utc),
        )

        # Convert to representation
        representation = Representation.from_documents([explicit_doc, deductive_doc])

        assert len(representation.explicit) == 1
        assert len(representation.deductive) == 1

        explicit_obs = representation.explicit[0]
        assert explicit_obs.content == "User said they like programming"
        assert explicit_obs.message_ids == [(1, 1)]
        assert explicit_obs.session_name == "test_session"

        deductive_obs = representation.deductive[0]
        assert deductive_obs.conclusion == "User is likely a software developer"
        assert deductive_obs.premises == ["User said they like programming"]
        assert deductive_obs.message_ids == [(1, 1)]
        assert deductive_obs.session_name == "test_session"

    async def create_test_workspace_and_peer(
        self, db_session: AsyncSession, workspace_name: str | None = None
    ) -> tuple[models.Workspace, models.Peer]:
        """Helper to create test workspace and peer"""
        workspace_name = workspace_name or generate_nanoid()
        peer_name = generate_nanoid()

        # Check if workspace already exists to avoid uniqueness constraint
        workspace = (
            await db_session.execute(
                select(models.Workspace).where(models.Workspace.name == workspace_name)
            )
        ).scalar_one_or_none()

        if workspace is None:
            workspace = models.Workspace(name=workspace_name)
            db_session.add(workspace)
            await db_session.flush()

        peer = models.Peer(name=peer_name, workspace_name=workspace_name)
        db_session.add(peer)
        await db_session.flush()

        return workspace, peer

    async def create_test_session(
        self, db_session: AsyncSession, workspace: models.Workspace
    ) -> models.Session:
        """Helper to create test session"""
        session_name = generate_nanoid()

        session = models.Session(
            name=session_name,
            workspace_name=workspace.name,
        )
        db_session.add(session)
        await db_session.flush()

        return session


@pytest.mark.asyncio
class TestPromptRepresentationConversion:
    """Test conversion between PromptRepresentation and Representation"""

    async def test_prompt_representation_to_representation(self):
        """Test converting PromptRepresentation to Representation"""
        prompt_rep = PromptRepresentation(
            explicit=[
                ExplicitObservationBase(content="User likes coffee"),
                ExplicitObservationBase(content="User works remotely"),
            ],
            deductive=[
                DeductiveObservationBase(
                    conclusion="User probably works from a coffee shop sometimes",
                    premises=["User likes coffee", "User works remotely"],
                )
            ],
        )

        timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        representation = Representation.from_prompt_representation(
            prompt_rep,
            message_ids=(123, 123),
            session_name="test_session",
            created_at=timestamp,
        )

        assert len(representation.explicit) == 2
        assert len(representation.deductive) == 1

        # Check explicit observations
        assert representation.explicit[0].content == "User likes coffee"
        assert representation.explicit[0].message_ids == [(123, 123)]
        assert representation.explicit[0].session_name == "test_session"
        assert representation.explicit[1].content == "User works remotely"
        assert representation.explicit[0].created_at == timestamp

        # Check deductive observation
        deductive_obs = representation.deductive[0]
        assert (
            deductive_obs.conclusion
            == "User probably works from a coffee shop sometimes"
        )
        assert deductive_obs.premises == ["User likes coffee", "User works remotely"]
        assert deductive_obs.message_ids == [(123, 123)]
        assert deductive_obs.session_name == "test_session"
        assert deductive_obs.created_at == timestamp

    async def test_empty_prompt_representation_conversion(self):
        """Test converting empty PromptRepresentation"""
        empty_prompt_rep = PromptRepresentation()
        representation = Representation.from_prompt_representation(
            empty_prompt_rep,
            message_ids=(1, 1),
            session_name="test",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert representation.is_empty()
        assert len(representation.explicit) == 0
        assert len(representation.deductive) == 0


@pytest.mark.asyncio
class TestRepresentationHashingAndEquality:
    """Test hashing and equality behavior for observations"""

    async def test_explicit_observation_equality(self):
        """Test ExplicitObservation equality and hashing"""
        obs1 = ExplicitObservation(
            content="Test content",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        obs2 = ExplicitObservation(
            content="Test content",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        obs3 = ExplicitObservation(
            content="Different content",
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        # Test equality
        assert obs1 == obs2
        assert obs1 != obs3
        assert obs1 != "not an observation"

        # Test hashing (should be able to use in sets)
        obs_set = {obs1, obs2, obs3}
        assert len(obs_set) == 2  # obs1 and obs2 are duplicates

    async def test_deductive_observation_equality(self):
        """Test DeductiveObservation equality and hashing"""
        obs1 = DeductiveObservation(
            conclusion="Test conclusion",
            premises=["premise1", "premise2"],
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        obs2 = DeductiveObservation(
            conclusion="Test conclusion",
            premises=["premise1", "premise2"],
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        obs3 = DeductiveObservation(
            conclusion="Different conclusion",
            premises=["premise1", "premise2"],
            created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            message_ids=[(1, 1)],
            session_name="session1",
        )

        # Test equality
        assert obs1 == obs2
        assert obs1 != obs3
        assert obs1 != "not an observation"

        # Test hashing
        obs_set = {obs1, obs2, obs3}
        assert len(obs_set) == 2
