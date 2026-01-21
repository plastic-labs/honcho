# pyright: reportPrivateUsage=false
"""Integration tests for telemetry with a mock ingestion endpoint.

These tests verify the full telemetry pipeline:
- Event creation → CloudEvent conversion → HTTP transport → Server receipt

Uses httpx's MockTransport to simulate the server without real network calls.
This avoids event loop issues and doesn't need database/cache fixtures.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.telemetry.emitter import TelemetryEmitter
from src.telemetry.events.agent import (
    AgentIterationEvent,
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    AgentToolSummaryCreatedEvent,
)
from src.telemetry.events.deletion import DeletionCompletedEvent
from src.telemetry.events.dialectic import DialecticCompletedEvent
from src.telemetry.events.dream import DreamRunEvent, DreamSpecialistEvent
from src.telemetry.events.reconciliation import (
    CleanupStaleItemsCompletedEvent,
    SyncVectorsCompletedEvent,
)
from src.telemetry.events.representation import RepresentationCompletedEvent

# =============================================================================
# Mock HTTP Transport for capturing requests
# =============================================================================


class MockTransport:
    """A mock transport that captures requests for testing."""

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.received_events: list[dict[str, Any]] = []
        self.response_code: int = 200
        self.fail_next_n: int = 0

    def reset(self):
        """Reset captured data."""
        self.requests.clear()
        self.received_events.clear()
        self.fail_next_n = 0

    async def aclose(self) -> None:
        """Mock close method to satisfy AsyncClient interface."""
        pass

    def _make_response(self, status_code: int, url: str) -> httpx.Response:
        """Create an httpx.Response with a properly attached request."""
        request = httpx.Request("POST", url)
        response = httpx.Response(status_code, request=request)
        return response

    async def post(
        self, url: str, content: bytes = b"", headers: dict[str, str] | None = None
    ) -> httpx.Response:
        """Handle a POST request and return a response.

        This mimics httpx.AsyncClient.post() interface.
        """
        self.requests.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers or {},
                "content": content,
            }
        )

        # Check if we should fail
        if self.fail_next_n > 0:
            self.fail_next_n -= 1
            return self._make_response(500, url)

        # Parse the request body
        try:
            content_str = content.decode("utf-8")
            content_type = (headers or {}).get("Content-Type", "")

            if "batch" in content_type:
                events = json.loads(content_str)
                self.received_events.extend(events)
            else:
                event = json.loads(content_str)
                self.received_events.append(event)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._make_response(400, url)

        return self._make_response(self.response_code, url)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fixed_timestamp() -> datetime:
    """Return a fixed timestamp for deterministic event testing."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def mock_transport() -> MockTransport:
    """Create a mock transport for capturing HTTP requests."""
    return MockTransport()


# =============================================================================
# Sample Event Factories
# =============================================================================


def create_representation_event(
    timestamp: datetime, suffix: str = ""
) -> RepresentationCompletedEvent:
    """Create a RepresentationCompletedEvent."""
    return RepresentationCompletedEvent(
        timestamp=timestamp,
        workspace_name="test_workspace",
        session_name="test_session",
        observed="user_peer",
        queue_items_processed=3,
        earliest_message_id="msg_001",
        latest_message_id=f"msg_010{suffix}",
        message_count=10,
        explicit_conclusion_count=5,
        context_preparation_ms=50.0,
        llm_call_ms=1200.0,
        total_duration_ms=1300.0,
        input_tokens=5000,
        output_tokens=500,
    )


def create_dream_run_event(timestamp: datetime) -> DreamRunEvent:
    """Create a DreamRunEvent."""
    return DreamRunEvent(
        timestamp=timestamp,
        run_id="abc12345",
        workspace_name="test_workspace",
        session_name="test_session",
        observer="assistant",
        observed="user_peer",
        specialists_run=["deduction", "induction"],
        deduction_success=True,
        induction_success=True,
        total_iterations=15,
        total_input_tokens=25000,
        total_output_tokens=3000,
        total_duration_ms=45000.0,
    )


def create_dream_specialist_event(timestamp: datetime) -> DreamSpecialistEvent:
    """Create a DreamSpecialistEvent."""
    return DreamSpecialistEvent(
        timestamp=timestamp,
        run_id="abc12345",
        specialist_type="deduction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        iterations=8,
        tool_calls_count=12,
        input_tokens=15000,
        output_tokens=2000,
        duration_ms=25000.0,
        success=True,
    )


def create_dialectic_event(timestamp: datetime) -> DialecticCompletedEvent:
    """Create a DialecticCompletedEvent."""
    return DialecticCompletedEvent(
        timestamp=timestamp,
        run_id="def67890",
        workspace_name="test_workspace",
        peer_name="user_peer",
        reasoning_level="medium",
        total_duration_ms=3500.0,
        input_tokens=8000,
        output_tokens=1200,
    )


def create_agent_iteration_event(timestamp: datetime) -> AgentIterationEvent:
    """Create an AgentIterationEvent."""
    return AgentIterationEvent(
        timestamp=timestamp,
        run_id="abc12345",
        parent_category="dream",
        agent_type="deduction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        iteration=3,
        tool_calls=["search_memory", "create_observations"],
        input_tokens=3000,
        output_tokens=500,
    )


def create_conclusions_created_event(
    timestamp: datetime,
) -> AgentToolConclusionsCreatedEvent:
    """Create an AgentToolConclusionsCreatedEvent."""
    return AgentToolConclusionsCreatedEvent(
        timestamp=timestamp,
        run_id="abc12345",
        iteration=3,
        parent_category="dream",
        agent_type="deduction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        conclusion_count=5,
        levels=["explicit", "deductive", "deductive", "explicit", "deductive"],
    )


def create_conclusions_deleted_event(
    timestamp: datetime,
) -> AgentToolConclusionsDeletedEvent:
    """Create an AgentToolConclusionsDeletedEvent."""
    return AgentToolConclusionsDeletedEvent(
        timestamp=timestamp,
        run_id="abc12345",
        iteration=5,
        parent_category="dream",
        agent_type="deduction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        conclusion_count=3,
    )


def create_peer_card_updated_event(
    timestamp: datetime,
) -> AgentToolPeerCardUpdatedEvent:
    """Create an AgentToolPeerCardUpdatedEvent."""
    return AgentToolPeerCardUpdatedEvent(
        timestamp=timestamp,
        run_id="abc12345",
        iteration=7,
        parent_category="dream",
        agent_type="induction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        facts_count=12,
    )


def create_summary_created_event(timestamp: datetime) -> AgentToolSummaryCreatedEvent:
    """Create an AgentToolSummaryCreatedEvent."""
    return AgentToolSummaryCreatedEvent(
        timestamp=timestamp,
        run_id="ghi11111",
        iteration=1,
        parent_category="representation",
        agent_type="summarizer",
        workspace_name="test_workspace",
        session_name="test_session",
        message_id="msg_020",
        message_count=20,
        message_seq_in_session=20,
        summary_type="short",
        input_tokens=4000,
        output_tokens=300,
    )


def create_deletion_event(timestamp: datetime) -> DeletionCompletedEvent:
    """Create a DeletionCompletedEvent."""
    return DeletionCompletedEvent(
        timestamp=timestamp,
        workspace_name="test_workspace",
        deletion_type="workspace",
        resource_id="ws_123",
        success=True,
        peers_deleted=5,
        sessions_deleted=10,
        messages_deleted=500,
        conclusions_deleted=200,
    )


def create_sync_vectors_event(timestamp: datetime) -> SyncVectorsCompletedEvent:
    """Create a SyncVectorsCompletedEvent."""
    return SyncVectorsCompletedEvent(
        timestamp=timestamp,
        documents_synced=150,
        documents_failed=2,
        message_embeddings_synced=500,
        message_embeddings_failed=0,
        total_duration_ms=12000.0,
    )


def create_cleanup_event(timestamp: datetime) -> CleanupStaleItemsCompletedEvent:
    """Create a CleanupStaleItemsCompletedEvent."""
    return CleanupStaleItemsCompletedEvent(
        timestamp=timestamp,
        documents_cleaned=25,
        queue_items_cleaned=100,
        total_duration_ms=5000.0,
    )


# =============================================================================
# Integration Tests - Basic Event Sending
# =============================================================================


class TestBasicEventSending:
    """Tests for basic event sending functionality."""

    @pytest.mark.asyncio
    async def test_single_event_received(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Single event is received by the server."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=10,
            flush_interval_seconds=10,  # Don't auto-flush
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "integration_test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "representation.completed"
        assert "id" in received
        assert received["id"].startswith("evt_")

    @pytest.mark.asyncio
    async def test_multiple_events_received(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Multiple events are received in batches."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=10,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "integration_test"
            for i in range(5):
                event = create_representation_event(fixed_timestamp, f"_{i}")
                emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 5

    @pytest.mark.asyncio
    async def test_cloudevent_structure(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """CloudEvent has correct structure and attributes."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=10,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "integration_test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        received = mock_transport.received_events[0]

        # Required CloudEvent attributes
        assert "id" in received
        assert "source" in received
        assert "type" in received
        assert "time" in received
        assert "data" in received

        # Verify source includes namespace
        assert "integration_test" in received["source"]
        assert "/honcho/" in received["source"]

        # Verify data contains event fields
        data = received["data"]
        assert data["workspace_name"] == "test_workspace"
        assert data["message_count"] == 10


# =============================================================================
# Integration Tests - All Event Types
# =============================================================================


class TestAllEventTypes:
    """Tests that all 12 event types can be sent and received."""

    @pytest.mark.asyncio
    async def test_representation_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """RepresentationCompletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        assert mock_transport.received_events[0]["type"] == "representation.completed"

    @pytest.mark.asyncio
    async def test_dream_run_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """DreamRunEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_dream_run_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "dream.run"
        assert received["data"]["run_id"] == "abc12345"

    @pytest.mark.asyncio
    async def test_dream_specialist_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """DreamSpecialistEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_dream_specialist_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "dream.specialist"
        assert received["data"]["specialist_type"] == "deduction"

    @pytest.mark.asyncio
    async def test_dialectic_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """DialecticCompletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_dialectic_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "dialectic.completed"
        assert received["data"]["reasoning_level"] == "medium"

    @pytest.mark.asyncio
    async def test_agent_iteration_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """AgentIterationEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_agent_iteration_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "agent.iteration"
        assert received["data"]["iteration"] == 3

    @pytest.mark.asyncio
    async def test_conclusions_created_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """AgentToolConclusionsCreatedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_conclusions_created_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "agent.tool.conclusions.created"
        assert received["data"]["conclusion_count"] == 5

    @pytest.mark.asyncio
    async def test_conclusions_deleted_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """AgentToolConclusionsDeletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_conclusions_deleted_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "agent.tool.conclusions.deleted"
        assert received["data"]["conclusion_count"] == 3

    @pytest.mark.asyncio
    async def test_peer_card_updated_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """AgentToolPeerCardUpdatedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_peer_card_updated_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "agent.tool.peer_card.updated"
        assert received["data"]["facts_count"] == 12

    @pytest.mark.asyncio
    async def test_summary_created_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """AgentToolSummaryCreatedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_summary_created_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "agent.tool.summary.created"
        assert received["data"]["summary_type"] == "short"

    @pytest.mark.asyncio
    async def test_deletion_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """DeletionCompletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_deletion_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "deletion.completed"
        assert received["data"]["deletion_type"] == "workspace"

    @pytest.mark.asyncio
    async def test_sync_vectors_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """SyncVectorsCompletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_sync_vectors_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "reconciliation.sync_vectors.completed"
        assert received["data"]["documents_synced"] == 150

    @pytest.mark.asyncio
    async def test_cleanup_event(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """CleanupStaleItemsCompletedEvent is sent correctly."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_cleanup_event(fixed_timestamp)
            emitter.emit(event)

        await emitter.flush()

        assert len(mock_transport.received_events) == 1
        received = mock_transport.received_events[0]
        assert received["type"] == "reconciliation.cleanup_stale_items.completed"
        assert received["data"]["documents_cleaned"] == 25


# =============================================================================
# Integration Tests - Error Scenarios
# =============================================================================


class TestErrorScenarios:
    """Tests for error handling with HTTP transport."""

    @pytest.mark.asyncio
    async def test_server_error_triggers_retry(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Server errors trigger retry and eventual success."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=3,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Configure to fail first 2 requests
        mock_transport.fail_next_n = 2

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),  # Skip delays
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)
            await emitter.flush()

        # Should have made 3 requests (2 failures + 1 success)
        assert len(mock_transport.requests) == 3
        # Event should have been received on 3rd attempt
        assert len(mock_transport.received_events) == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_keeps_events(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Events returned to buffer when all retries fail."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=2,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Configure to fail all requests
        mock_transport.fail_next_n = 10

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)
            await emitter.flush()

        # All retries should have been attempted
        assert len(mock_transport.requests) == 2
        # Event should be back in buffer
        assert emitter.buffer_size == 1

    @pytest.mark.asyncio
    async def test_server_returns_4xx_error(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Client errors (4xx) also trigger retries."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            max_retries=2,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        mock_transport.response_code = 400

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with (
            patch("src.config.settings") as mock_settings,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.TELEMETRY.NAMESPACE = "test"
            event = create_representation_event(fixed_timestamp)
            emitter.emit(event)
            await emitter.flush()

        # Should have tried 2 times
        assert len(mock_transport.requests) == 2


# =============================================================================
# Integration Tests - Batching Behavior
# =============================================================================


class TestBatchingBehavior:
    """Tests for batch sending behavior."""

    @pytest.mark.asyncio
    async def test_events_batched_correctly(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Events are batched according to batch_size."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=3,  # Small batch size for testing
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Emit 7 events
            for i in range(7):
                event = create_representation_event(fixed_timestamp, f"_{i}")
                emitter.emit(event)

        await emitter.flush()

        # Should have made 3 requests (batches of 3, 3, 1)
        assert len(mock_transport.requests) == 3
        # All events should be received
        assert len(mock_transport.received_events) == 7

    @pytest.mark.asyncio
    async def test_flush_threshold_triggers_send(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Reaching flush_threshold triggers automatic flush."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=100,
            flush_interval_seconds=10,  # Long interval
            flush_threshold=3,  # Low threshold
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Emit exactly threshold events
            for i in range(3):
                event = create_representation_event(fixed_timestamp, f"_{i}")
                emitter.emit(event)

        # Give time for async flush to trigger
        await asyncio.sleep(0.1)

        # Should have auto-flushed
        assert len(mock_transport.requests) >= 1
        assert len(mock_transport.received_events) == 3


# =============================================================================
# Integration Tests - Idempotency
# =============================================================================


class TestIdempotency:
    """Tests for event ID idempotency."""

    @pytest.mark.asyncio
    async def test_same_event_has_same_id(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Same event parameters produce same event ID."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Create two identical events
            event1 = create_representation_event(fixed_timestamp)
            event2 = create_representation_event(fixed_timestamp)

            emitter.emit(event1)
            emitter.emit(event2)

        await emitter.flush()

        # Both events should have same ID
        assert len(mock_transport.received_events) == 2
        assert (
            mock_transport.received_events[0]["id"]
            == mock_transport.received_events[1]["id"]
        )

    @pytest.mark.asyncio
    async def test_different_events_have_different_ids(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Different event parameters produce different event IDs."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Create events with different suffixes (different latest_message_id)
            event1 = create_representation_event(fixed_timestamp, "_a")
            event2 = create_representation_event(fixed_timestamp, "_b")

            emitter.emit(event1)
            emitter.emit(event2)

        await emitter.flush()

        # Events should have different IDs
        assert len(mock_transport.received_events) == 2
        assert (
            mock_transport.received_events[0]["id"]
            != mock_transport.received_events[1]["id"]
        )


# =============================================================================
# Integration Tests - Shutdown Behavior
# =============================================================================


class TestShutdownBehavior:
    """Tests for graceful shutdown with pending events."""

    @pytest.mark.asyncio
    async def test_shutdown_flushes_pending_events(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Shutdown flushes all pending events before closing."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=100,
            flush_interval_seconds=60,  # Very long interval
            flush_threshold=100,  # High threshold
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True
        emitter._flush_task = asyncio.create_task(asyncio.sleep(100))

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"
            # Emit events without triggering auto-flush
            for i in range(5):
                event = create_representation_event(fixed_timestamp, f"_{i}")
                emitter.emit(event)

        # Verify events are buffered
        assert emitter.buffer_size == 5

        # Shutdown should flush
        await emitter.shutdown()

        # All events should have been sent
        assert len(mock_transport.received_events) == 5
        assert emitter.buffer_size == 0


# =============================================================================
# Integration Tests - Mixed Event Types
# =============================================================================


class TestMixedEventTypes:
    """Tests for sending different event types together."""

    @pytest.mark.asyncio
    async def test_mixed_event_types_in_batch(
        self,
        mock_transport: MockTransport,
        fixed_timestamp: datetime,
    ):
        """Different event types can be batched together."""
        emitter = TelemetryEmitter(
            endpoint="http://test:8001/events",
            batch_size=10,
            flush_interval_seconds=10,
            flush_threshold=100,
            enabled=True,
        )

        # Inject mock transport as client
        emitter._client = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        emitter._running = True

        with patch("src.config.settings") as mock_settings:
            mock_settings.TELEMETRY.NAMESPACE = "test"

            # Emit different event types
            emitter.emit(create_representation_event(fixed_timestamp))
            emitter.emit(create_dream_run_event(fixed_timestamp))
            emitter.emit(create_dialectic_event(fixed_timestamp))
            emitter.emit(create_agent_iteration_event(fixed_timestamp))
            emitter.emit(create_deletion_event(fixed_timestamp))

        await emitter.flush()

        # All events should be received
        assert len(mock_transport.received_events) == 5

        # Check we got different types
        types = {e["type"] for e in mock_transport.received_events}
        assert "representation.completed" in types
        assert "dream.run" in types
        assert "dialectic.completed" in types
        assert "agent.iteration" in types
        assert "deletion.completed" in types
