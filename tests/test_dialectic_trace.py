"""Tests for DialecticTrace CRUD operations."""

import datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud.dialectic_trace import _is_abstention
from src.schemas import DialecticTraceCreate


class TestDialecticTraceUnit:
    """Unit tests for dialectic trace CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_dialectic_trace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating a dialectic trace record."""
        workspace, peer = sample_data

        trace_data = DialecticTraceCreate(
            workspace_name=workspace.name,
            session_name=None,
            observer=peer.name,
            observed=peer.name,
            query="What does the user like?",
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            tool_calls=[
                {"name": "search_memory", "input": {"query": "likes"}, "id": "call_1"}
            ],
            response="The user likes coffee and hiking.",
            reasoning_level="low",
            total_duration_ms=1234.56,
            input_tokens=500,
            output_tokens=100,
        )

        trace = await crud.create_dialectic_trace(db_session, trace_data)

        assert trace.id is not None
        assert len(trace.id) == 21  # nanoid default length
        assert trace.workspace_name == workspace.name
        assert trace.session_name is None
        assert trace.observer == peer.name
        assert trace.observed == peer.name
        assert trace.query == "What does the user like?"
        assert trace.retrieved_doc_ids == ["doc1", "doc2", "doc3"]
        assert len(trace.tool_calls) == 1
        assert trace.tool_calls[0]["name"] == "search_memory"
        assert trace.response == "The user likes coffee and hiking."
        assert trace.reasoning_level == "low"
        assert trace.total_duration_ms == 1234.56
        assert trace.input_tokens == 500
        assert trace.output_tokens == 100
        assert trace.created_at is not None

    @pytest.mark.asyncio
    async def test_create_dialectic_trace_with_session(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test creating a trace with a session name."""
        workspace, peer = sample_data

        # Create a session
        session = models.Session(name=generate_nanoid(), workspace_name=workspace.name)
        db_session.add(session)
        await db_session.flush()

        trace_data = DialecticTraceCreate(
            workspace_name=workspace.name,
            session_name=session.name,
            observer=peer.name,
            observed=peer.name,
            query="Test query",
            response="Test response",
            reasoning_level="medium",
            total_duration_ms=500.0,
            input_tokens=200,
            output_tokens=50,
        )

        trace = await crud.create_dialectic_trace(db_session, trace_data)

        assert trace.session_name == session.name

    @pytest.mark.asyncio
    async def test_get_dialectic_traces(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test retrieving dialectic traces for a workspace."""
        workspace, peer = sample_data

        # Create multiple traces
        for i in range(5):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response=f"Response {i}",
                reasoning_level="low",
                total_duration_ms=100.0 * (i + 1),
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        traces = await crud.get_dialectic_traces(db_session, workspace.name)

        assert len(traces) == 5
        # Should be ordered by created_at descending (most recent first)
        # Due to fast creation, we check that all traces are present
        queries = {t.query for t in traces}
        assert queries == {"Query 0", "Query 1", "Query 2", "Query 3", "Query 4"}

    @pytest.mark.asyncio
    async def test_get_dialectic_traces_with_limit_offset(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test pagination of dialectic traces."""
        workspace, peer = sample_data

        # Create 10 traces
        for i in range(10):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response=f"Response {i}",
                reasoning_level="low",
                total_duration_ms=100.0,
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        # Test limit
        traces = await crud.get_dialectic_traces(db_session, workspace.name, limit=3)
        assert len(traces) == 3

        # Test offset
        traces = await crud.get_dialectic_traces(
            db_session, workspace.name, limit=5, offset=5
        )
        assert len(traces) == 5

    @pytest.mark.asyncio
    async def test_get_dialectic_trace_stats(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting aggregate statistics for dialectic traces."""
        workspace, peer = sample_data

        # Create traces with varying durations
        durations = [100.0, 200.0, 300.0, 400.0, 500.0]
        for i, duration in enumerate(durations):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response=f"Response {i}",
                reasoning_level="low",
                total_duration_ms=duration,
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        stats = await crud.get_dialectic_trace_stats(db_session, workspace.name)

        assert stats["total_queries"] == 5
        assert stats["avg_duration_ms"] == 300.0  # (100+200+300+400+500) / 5
        assert stats["abstention_count"] == 0
        assert stats["abstention_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_dialectic_trace_stats_with_abstentions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that abstentions are correctly detected and counted."""
        workspace, peer = sample_data

        responses = [
            "The user likes coffee.",  # Not abstention
            "I don't have enough information to answer that.",  # Abstention
            "Based on the observations, the user prefers tea.",  # Not abstention
            "I cannot find any relevant data about this topic.",  # Abstention
            "No relevant observations found for this query.",  # Abstention
        ]

        for i, response in enumerate(responses):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response=response,
                reasoning_level="low",
                total_duration_ms=100.0,
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        stats = await crud.get_dialectic_trace_stats(db_session, workspace.name)

        assert stats["total_queries"] == 5
        assert stats["abstention_count"] == 3
        assert stats["abstention_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_get_dialectic_trace_stats_with_since_filter(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test filtering stats by time."""
        workspace, peer = sample_data

        # Create traces
        for i in range(3):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response=f"Response {i}",
                reasoning_level="low",
                total_duration_ms=100.0,
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        # Get stats since far future (should return zero)
        future_time = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)
        stats = await crud.get_dialectic_trace_stats(
            db_session, workspace.name, since=future_time
        )

        assert stats["total_queries"] == 0
        assert stats["abstention_count"] == 0
        assert stats["abstention_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_dialectic_trace_stats_empty_workspace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test stats for workspace with no traces."""
        workspace, _ = sample_data

        stats = await crud.get_dialectic_trace_stats(db_session, workspace.name)

        assert stats["total_queries"] == 0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["abstention_count"] == 0
        assert stats["abstention_rate"] == 0.0


class TestAbstentionDetection:
    """Test the abstention detection helper function."""

    def test_abstention_patterns(self):
        """Test that various abstention patterns are detected."""
        abstention_responses = [
            "I don't have information about that.",
            "I don't have enough information to answer.",
            "I cannot answer this question.",
            "I cannot find any relevant data.",
            "There is no relevant information available.",
            "Not enough information to determine.",
            "Not sufficient context available.",
            "Unable to find any observations.",
            "No observations found for this query.",
            "No memory available about this topic.",
            "No data found regarding this question.",
        ]

        for response in abstention_responses:
            assert _is_abstention(response), f"Should detect abstention: {response}"

    def test_non_abstention_responses(self):
        """Test that normal responses are not flagged as abstentions."""
        normal_responses = [
            "The user likes coffee.",
            "Based on observations, they prefer morning meetings.",
            "The data shows a preference for Python.",
            "They have mentioned enjoying hiking.",
            "According to recent conversations, they work remotely.",
        ]

        for response in normal_responses:
            assert not _is_abstention(response), (
                f"Should not detect abstention: {response}"
            )


class TestDocIdExtraction:
    """Test document ID extraction from messages."""

    def test_extract_doc_ids_from_tool_results(self):
        """Test extracting document IDs from formatted tool results."""
        from src.dialectic.core import _extract_doc_ids_from_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Query: What does the user like?"},
            {
                "role": "assistant",
                "content": "Let me search for relevant information.",
            },
            {
                "role": "user",
                "content": """Found 3 observations:

## Explicit Observations
[id:abc123] [2025-01-01] The user likes coffee
[id:def456] [2025-01-02] The user works remotely

## Deductive Observations
[id:ghi789] [2025-01-03] The user likely prefers morning meetings""",
            },
            {"role": "assistant", "content": "Based on the observations..."},
        ]

        doc_ids = _extract_doc_ids_from_messages(messages)

        assert set(doc_ids) == {"abc123", "def456", "ghi789"}

    def test_extract_doc_ids_no_matches(self):
        """Test extraction when no document IDs are present."""
        from src.dialectic.core import _extract_doc_ids_from_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        doc_ids = _extract_doc_ids_from_messages(messages)

        assert doc_ids == []

    def test_extract_doc_ids_duplicates_removed(self):
        """Test that duplicate IDs are deduplicated."""
        from src.dialectic.core import _extract_doc_ids_from_messages

        messages = [
            {
                "role": "user",
                "content": "[id:abc123] First mention\n[id:abc123] Same ID again",
            },
        ]

        doc_ids = _extract_doc_ids_from_messages(messages)

        assert doc_ids == ["abc123"]
