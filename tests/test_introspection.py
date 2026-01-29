"""Tests for the introspection module."""

import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.dreamer.introspection import (
    INTROSPECTION_OBSERVED,
    SYSTEM_OBSERVER,
    build_introspection_prompt,
    gather_introspection_context,
    run_introspection,
    store_introspection_report,
)
from src.schemas import (
    DialecticTraceCreate,
    IntrospectionReport,
    IntrospectionSignals,
    IntrospectionSuggestion,
)


class TestGatherIntrospectionContext:
    """Tests for gathering introspection signals."""

    @pytest.mark.asyncio
    async def test_gather_introspection_context_empty_workspace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test gathering signals from an empty workspace returns zero values."""
        workspace, _ = sample_data

        signals = await gather_introspection_context(db_session, workspace.name)

        assert signals.total_dialectic_queries == 0
        assert signals.avg_dialectic_duration_ms == 0.0
        assert signals.abstention_count == 0
        assert signals.abstention_rate == 0.0
        assert signals.recent_queries == []
        assert signals.total_observations == 0
        assert signals.observations_by_level == {}
        assert signals.contradiction_count == 0
        # Peer count may be 1 (the sample peer), but should not include system peers
        assert signals.total_sessions == 0
        assert signals.current_deriver_rules == ""
        assert signals.current_dialectic_rules == ""

    @pytest.mark.asyncio
    async def test_gather_introspection_context_with_data(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test gathering signals from a workspace with data."""
        workspace, peer = sample_data

        # Create some dialectic traces
        for i in range(5):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"What does the user like? Query {i}",
                response="The user likes coffee."
                if i < 3
                else "I don't have information.",
                reasoning_level="low",
                total_duration_ms=100.0 * (i + 1),
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        # Create a session
        session = models.Session(name=generate_nanoid(), workspace_name=workspace.name)
        db_session.add(session)
        await db_session.flush()

        signals = await gather_introspection_context(db_session, workspace.name)

        assert signals.total_dialectic_queries == 5
        assert signals.avg_dialectic_duration_ms == 300.0  # (100+200+300+400+500)/5
        assert signals.abstention_count == 2  # Last 2 responses are abstentions
        assert signals.abstention_rate == 0.4
        assert len(signals.recent_queries) == 5
        assert "What does the user like?" in signals.recent_queries[0]
        assert signals.total_sessions == 1


class TestBuildIntrospectionPrompt:
    """Tests for building the introspection prompt."""

    def test_build_introspection_prompt_empty_signals(self):
        """Test prompt building with empty signals."""
        signals = IntrospectionSignals()

        prompt = build_introspection_prompt(signals)

        assert "Total queries: 0" in prompt
        assert "Abstention rate: 0.0%" in prompt
        assert "(No recent queries)" in prompt
        assert "(No observations)" in prompt
        assert "(empty - using defaults)" in prompt

    def test_build_introspection_prompt_with_data(self):
        """Test prompt building with populated signals."""
        signals = IntrospectionSignals(
            total_dialectic_queries=100,
            avg_dialectic_duration_ms=250.5,
            abstention_count=30,
            abstention_rate=0.3,
            recent_queries=["What is the user's name?", "What do they prefer?"],
            total_observations=500,
            observations_by_level={"explicit": 300, "deductive": 150, "inductive": 50},
            contradiction_count=5,
            total_peers=10,
            total_sessions=25,
            current_deriver_rules="Focus on preferences",
            current_dialectic_rules="Be concise",
        )

        prompt = build_introspection_prompt(signals)

        assert "Total queries: 100" in prompt
        assert "250.5ms" in prompt
        assert "Abstention rate: 30.0%" in prompt
        assert "What is the user's name?" in prompt
        assert "explicit: 300" in prompt
        assert "deductive: 150" in prompt
        assert "Contradictions detected: 5" in prompt
        assert "Total peers: 10" in prompt
        assert "Total sessions: 25" in prompt
        assert "Focus on preferences" in prompt
        assert "Be concise" in prompt


class TestRunIntrospection:
    """Tests for the main introspection runner."""

    @pytest.mark.asyncio
    async def test_run_introspection_insufficient_data(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test introspection with insufficient data returns appropriate report."""
        workspace, _ = sample_data

        with patch(
            "src.dreamer.introspection.store_introspection_report",
            new_callable=AsyncMock,
        ) as mock_store:
            report = await run_introspection(db_session, workspace.name)

        assert report is not None
        assert report.workspace_name == workspace.name
        assert "Insufficient data" in report.performance_summary
        assert report.suggestions == []
        assert report.identified_issues == []
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_introspection_with_data(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test introspection with data calls LLM and parses response."""
        workspace, peer = sample_data

        # Create some dialectic traces
        for i in range(3):
            trace_data = DialecticTraceCreate(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                query=f"Query {i}",
                response="Response",
                reasoning_level="low",
                total_duration_ms=100.0,
                input_tokens=100,
                output_tokens=50,
            )
            await crud.create_dialectic_trace(db_session, trace_data)

        # Mock the LLM call
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps(
            {
                "performance_summary": "The workspace is performing well.",
                "identified_issues": ["High abstention rate"],
                "suggestions": [
                    {
                        "target": "deriver_rules",
                        "current_value": "",
                        "suggested_value": "Focus on capturing user preferences",
                        "rationale": "Too many queries about preferences are being missed",
                        "confidence": "medium",
                    }
                ],
            }
        )

        with (
            patch(
                "src.dreamer.introspection.honcho_llm_call",
                new_callable=AsyncMock,
                return_value=mock_llm_response,
            ),
            patch(
                "src.dreamer.introspection.store_introspection_report",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            report = await run_introspection(db_session, workspace.name)

        assert report is not None
        assert report.workspace_name == workspace.name
        assert report.performance_summary == "The workspace is performing well."
        assert "High abstention rate" in report.identified_issues
        assert len(report.suggestions) == 1
        assert report.suggestions[0].target == "deriver_rules"
        assert report.suggestions[0].confidence == "medium"
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_introspection_llm_failure(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test introspection handles LLM failures gracefully."""
        workspace, peer = sample_data

        # Create some dialectic traces
        trace_data = DialecticTraceCreate(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            query="Query",
            response="Response",
            reasoning_level="low",
            total_duration_ms=100.0,
            input_tokens=100,
            output_tokens=50,
        )
        await crud.create_dialectic_trace(db_session, trace_data)

        with (
            patch(
                "src.dreamer.introspection.honcho_llm_call",
                new_callable=AsyncMock,
                side_effect=Exception("LLM API error"),
            ),
            patch(
                "src.dreamer.introspection.store_introspection_report",
                new_callable=AsyncMock,
            ) as mock_store,
        ):
            report = await run_introspection(db_session, workspace.name)

        assert report is not None
        assert "Error during analysis" in report.performance_summary
        assert report.suggestions == []
        mock_store.assert_called_once()


class TestStoreIntrospectionReport:
    """Tests for storing introspection reports."""

    @pytest.mark.asyncio
    async def test_store_introspection_report_creates_document(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that storing a report creates a document in the reserved collection."""

        workspace, _ = sample_data
        # Capture workspace name before any potential session issues
        workspace_name = workspace.name

        # Create system peers and collection directly (bypassing mock)
        system_peer = models.Peer(name=SYSTEM_OBSERVER, workspace_name=workspace_name)
        introspection_peer = models.Peer(
            name=INTROSPECTION_OBSERVED, workspace_name=workspace_name
        )
        db_session.add_all([system_peer, introspection_peer])
        await db_session.flush()

        # Create the collection
        collection = models.Collection(
            workspace_name=workspace_name,
            observer=SYSTEM_OBSERVER,
            observed=INTROSPECTION_OBSERVED,
        )
        db_session.add(collection)
        await db_session.flush()

        report = IntrospectionReport(
            workspace_name=workspace_name,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary="Test summary",
            identified_issues=["Issue 1"],
            suggestions=[
                IntrospectionSuggestion(
                    target="deriver_rules",
                    current_value="",
                    suggested_value="New rule",
                    rationale="Test rationale",
                    confidence="high",
                )
            ],
            signals=IntrospectionSignals(total_dialectic_queries=10),
        )

        await store_introspection_report(db_session, workspace_name, report)

        # Verify document was created
        from sqlalchemy import select

        stmt = (
            select(models.Document)
            .where(models.Document.workspace_name == workspace_name)
            .where(models.Document.observer == SYSTEM_OBSERVER)
            .where(models.Document.observed == INTROSPECTION_OBSERVED)
        )
        result = await db_session.execute(stmt)
        doc = result.scalar_one_or_none()

        assert doc is not None
        assert doc.level == "explicit"
        assert doc.embedding is None  # No embedding for reports
        assert doc.sync_state == "synced"

        # Verify content is valid JSON
        content = json.loads(doc.content)
        assert content["workspace_name"] == workspace_name
        assert content["performance_summary"] == "Test summary"
        assert len(content["suggestions"]) == 1

    @pytest.mark.asyncio
    async def test_store_introspection_report_creates_system_peers(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that storing a report creates the _system and _introspection peers."""
        workspace, _ = sample_data
        # Capture workspace name before any potential session issues
        workspace_name = workspace.name

        report = IntrospectionReport(
            workspace_name=workspace_name,
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary="Test",
            signals=IntrospectionSignals(),
        )

        await store_introspection_report(db_session, workspace_name, report)

        # Verify system peers were created
        from sqlalchemy import select

        stmt = (
            select(models.Peer)
            .where(models.Peer.workspace_name == workspace_name)
            .where(models.Peer.name.in_([SYSTEM_OBSERVER, INTROSPECTION_OBSERVED]))
        )
        result = await db_session.execute(stmt)
        peers = list(result.scalars().all())

        assert len(peers) == 2
        peer_names = {p.name for p in peers}
        assert SYSTEM_OBSERVER in peer_names
        assert INTROSPECTION_OBSERVED in peer_names


class TestIntrospectionDreamDispatch:
    """Tests for introspection via the dream dispatch mechanism."""

    @pytest.mark.asyncio
    async def test_introspection_dream_type_dispatch(
        self,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that INTROSPECTION dream type is properly dispatched."""
        from src.dreamer.orchestrator import process_dream
        from src.schemas import DreamType
        from src.utils.queue_payload import DreamPayload

        workspace, peer = sample_data

        payload = DreamPayload(
            dream_type=DreamType.INTROSPECTION,
            observer=peer.name,
            observed=peer.name,
            session_name=None,
        )

        with patch(
            "src.dreamer.introspection.run_introspection",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = IntrospectionReport(
                workspace_name=workspace.name,
                generated_at=datetime.datetime.now(datetime.timezone.utc),
                performance_summary="Test",
                suggestions=[],
                signals=IntrospectionSignals(),
            )

            await process_dream(payload, workspace.name)

            mock_run.assert_called_once()
            # Verify it was called with the workspace name
            call_args = mock_run.call_args
            assert call_args[0][1] == workspace.name
