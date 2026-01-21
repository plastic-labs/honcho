# pyright: reportUnknownMemberType=false
"""Fixtures for telemetry unit tests.

This module provides:
- Sample event fixtures for all 12 event types
- Mock settings fixtures for controlling telemetry configuration
- Mock HTTP client fixtures for testing the emitter without network calls
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telemetry.events.agent import (
    AgentIterationEvent,
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    AgentToolSummaryCreatedEvent,
)
from src.telemetry.events.base import BaseEvent
from src.telemetry.events.deletion import DeletionCompletedEvent
from src.telemetry.events.dialectic import DialecticCompletedEvent
from src.telemetry.events.dream import DreamRunEvent, DreamSpecialistEvent
from src.telemetry.events.reconciliation import (
    CleanupStaleItemsCompletedEvent,
    SyncVectorsCompletedEvent,
)
from src.telemetry.events.representation import RepresentationCompletedEvent

# =============================================================================
# Fixed timestamp for deterministic tests
# =============================================================================


@pytest.fixture
def fixed_timestamp() -> datetime:
    """Return a fixed timestamp for deterministic event ID generation."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Sample event fixtures for all event types
# =============================================================================


@pytest.fixture
def sample_representation_event(
    fixed_timestamp: datetime,
) -> RepresentationCompletedEvent:
    """Create a sample RepresentationCompletedEvent for testing."""
    return RepresentationCompletedEvent(
        timestamp=fixed_timestamp,
        workspace_name="test_workspace",
        session_name="test_session",
        observed="user_peer",
        queue_items_processed=3,
        earliest_message_id="msg_001",
        latest_message_id="msg_010",
        message_count=10,
        explicit_conclusion_count=5,
        context_preparation_ms=50.0,
        llm_call_ms=1200.0,
        total_duration_ms=1300.0,
        input_tokens=5000,
        output_tokens=500,
    )


@pytest.fixture
def sample_dream_run_event(fixed_timestamp: datetime) -> DreamRunEvent:
    """Create a sample DreamRunEvent for testing."""
    return DreamRunEvent(
        timestamp=fixed_timestamp,
        run_id="abc12345",
        workspace_name="test_workspace",
        session_name="test_session",
        observer="assistant",
        observed="user_peer",
        specialists_run=["deduction", "induction"],
        deduction_success=True,
        induction_success=True,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=15,
        total_input_tokens=25000,
        total_output_tokens=3000,
        total_duration_ms=45000.0,
    )


@pytest.fixture
def sample_dream_specialist_event(fixed_timestamp: datetime) -> DreamSpecialistEvent:
    """Create a sample DreamSpecialistEvent for testing."""
    return DreamSpecialistEvent(
        timestamp=fixed_timestamp,
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


@pytest.fixture
def sample_dialectic_event(fixed_timestamp: datetime) -> DialecticCompletedEvent:
    """Create a sample DialecticCompletedEvent for testing."""
    return DialecticCompletedEvent(
        timestamp=fixed_timestamp,
        run_id="def67890",
        workspace_name="test_workspace",
        peer_name="user_peer",
        session_name="test_session",
        reasoning_level="medium",
        total_iterations=3,
        prefetched_conclusion_count=10,
        tool_calls_count=5,
        total_duration_ms=3500.0,
        input_tokens=8000,
        output_tokens=1200,
        cache_read_tokens=500,
        cache_creation_tokens=200,
    )


@pytest.fixture
def sample_agent_iteration_event(fixed_timestamp: datetime) -> AgentIterationEvent:
    """Create a sample AgentIterationEvent for testing."""
    return AgentIterationEvent(
        timestamp=fixed_timestamp,
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
        cache_read_tokens=100,
        cache_creation_tokens=50,
    )


@pytest.fixture
def sample_conclusions_created_event(
    fixed_timestamp: datetime,
) -> AgentToolConclusionsCreatedEvent:
    """Create a sample AgentToolConclusionsCreatedEvent for testing."""
    return AgentToolConclusionsCreatedEvent(
        timestamp=fixed_timestamp,
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


@pytest.fixture
def sample_conclusions_deleted_event(
    fixed_timestamp: datetime,
) -> AgentToolConclusionsDeletedEvent:
    """Create a sample AgentToolConclusionsDeletedEvent for testing."""
    return AgentToolConclusionsDeletedEvent(
        timestamp=fixed_timestamp,
        run_id="abc12345",
        iteration=5,
        parent_category="dream",
        agent_type="deduction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        conclusion_count=3,
    )


@pytest.fixture
def sample_peer_card_updated_event(
    fixed_timestamp: datetime,
) -> AgentToolPeerCardUpdatedEvent:
    """Create a sample AgentToolPeerCardUpdatedEvent for testing."""
    return AgentToolPeerCardUpdatedEvent(
        timestamp=fixed_timestamp,
        run_id="abc12345",
        iteration=7,
        parent_category="dream",
        agent_type="induction",
        workspace_name="test_workspace",
        observer="assistant",
        observed="user_peer",
        facts_count=12,
    )


@pytest.fixture
def sample_summary_created_event(
    fixed_timestamp: datetime,
) -> AgentToolSummaryCreatedEvent:
    """Create a sample AgentToolSummaryCreatedEvent for testing."""
    return AgentToolSummaryCreatedEvent(
        timestamp=fixed_timestamp,
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


@pytest.fixture
def sample_deletion_event(fixed_timestamp: datetime) -> DeletionCompletedEvent:
    """Create a sample DeletionCompletedEvent for testing."""
    return DeletionCompletedEvent(
        timestamp=fixed_timestamp,
        workspace_name="test_workspace",
        deletion_type="workspace",
        resource_id="ws_123abc",
        success=True,
        peers_deleted=5,
        sessions_deleted=10,
        messages_deleted=500,
        conclusions_deleted=200,
    )


@pytest.fixture
def sample_sync_vectors_event(fixed_timestamp: datetime) -> SyncVectorsCompletedEvent:
    """Create a sample SyncVectorsCompletedEvent for testing."""
    return SyncVectorsCompletedEvent(
        timestamp=fixed_timestamp,
        documents_synced=150,
        documents_failed=2,
        message_embeddings_synced=500,
        message_embeddings_failed=0,
        total_duration_ms=12000.0,
    )


@pytest.fixture
def sample_cleanup_event(fixed_timestamp: datetime) -> CleanupStaleItemsCompletedEvent:
    """Create a sample CleanupStaleItemsCompletedEvent for testing."""
    return CleanupStaleItemsCompletedEvent(
        timestamp=fixed_timestamp,
        documents_cleaned=25,
        queue_items_cleaned=100,
        total_duration_ms=5000.0,
    )


# =============================================================================
# All sample events as a collection
# =============================================================================


@pytest.fixture
def all_sample_events(
    sample_representation_event: RepresentationCompletedEvent,
    sample_dream_run_event: DreamRunEvent,
    sample_dream_specialist_event: DreamSpecialistEvent,
    sample_dialectic_event: DialecticCompletedEvent,
    sample_agent_iteration_event: AgentIterationEvent,
    sample_conclusions_created_event: AgentToolConclusionsCreatedEvent,
    sample_conclusions_deleted_event: AgentToolConclusionsDeletedEvent,
    sample_peer_card_updated_event: AgentToolPeerCardUpdatedEvent,
    sample_summary_created_event: AgentToolSummaryCreatedEvent,
    sample_deletion_event: DeletionCompletedEvent,
    sample_sync_vectors_event: SyncVectorsCompletedEvent,
    sample_cleanup_event: CleanupStaleItemsCompletedEvent,
) -> list[BaseEvent]:
    """Return all sample events as a list for parametrized tests."""
    return [
        sample_representation_event,
        sample_dream_run_event,
        sample_dream_specialist_event,
        sample_dialectic_event,
        sample_agent_iteration_event,
        sample_conclusions_created_event,
        sample_conclusions_deleted_event,
        sample_peer_card_updated_event,
        sample_summary_created_event,
        sample_deletion_event,
        sample_sync_vectors_event,
        sample_cleanup_event,
    ]


# =============================================================================
# Mock settings fixtures
# =============================================================================


@pytest.fixture
def mock_telemetry_settings():
    """Fixture to mock telemetry settings.

    Returns a context manager that patches settings with configurable values.
    """

    def _configure(
        enabled: bool = True,
        endpoint: str = "http://test-endpoint:8001/v1/events",
        namespace: str = "test_namespace",
        batch_size: int = 100,
        flush_interval: float = 1.0,
        flush_threshold: int = 50,
        max_retries: int = 3,
        max_buffer_size: int = 10000,
        headers: dict[str, str] | None = None,
    ):
        mock_settings = MagicMock()
        mock_settings.TELEMETRY.ENABLED = enabled
        mock_settings.TELEMETRY.ENDPOINT = endpoint
        mock_settings.TELEMETRY.NAMESPACE = namespace
        mock_settings.TELEMETRY.BATCH_SIZE = batch_size
        mock_settings.TELEMETRY.FLUSH_INTERVAL_SECONDS = flush_interval
        mock_settings.TELEMETRY.FLUSH_THRESHOLD = flush_threshold
        mock_settings.TELEMETRY.MAX_RETRIES = max_retries
        mock_settings.TELEMETRY.MAX_BUFFER_SIZE = max_buffer_size
        mock_settings.TELEMETRY.HEADERS = headers
        mock_settings.OTEL.ENABLED = False
        return patch("src.telemetry.emitter.settings", mock_settings)

    return _configure


# =============================================================================
# Mock HTTP client fixtures
# =============================================================================


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient for testing HTTP calls.

    Returns a factory function that creates configured mocks.
    """

    def _create(
        status_code: int = 200,
        raise_exception: Exception | None = None,
        response_sequence: list[tuple[int, Exception | None]] | None = None,
    ):
        """Create a mock HTTP client.

        Args:
            status_code: Default status code to return
            raise_exception: Exception to raise on post()
            response_sequence: List of (status_code, exception) for sequential calls
        """
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.raise_for_status = MagicMock()

        if raise_exception:
            mock_client.post = AsyncMock(side_effect=raise_exception)
        elif response_sequence:
            # Create side effects for sequential responses
            side_effects = []
            for code, exc in response_sequence:
                if exc:
                    side_effects.append(exc)
                else:
                    resp = MagicMock()
                    resp.status_code = code
                    if code >= 400:
                        import httpx

                        resp.raise_for_status = MagicMock(
                            side_effect=httpx.HTTPStatusError(
                                f"HTTP {code}",
                                request=MagicMock(),
                                response=resp,
                            )
                        )
                    else:
                        resp.raise_for_status = MagicMock()
                    side_effects.append(resp)
            mock_client.post = AsyncMock(side_effect=side_effects)
        else:
            if status_code >= 400:
                import httpx

                mock_response.raise_for_status = MagicMock(
                    side_effect=httpx.HTTPStatusError(
                        f"HTTP {status_code}",
                        request=MagicMock(),
                        response=mock_response,
                    )
                )
            mock_client.post = AsyncMock(return_value=mock_response)

        mock_client.aclose = AsyncMock()
        return mock_client

    return _create


# =============================================================================
# Emitter fixtures
# =============================================================================


@pytest.fixture
def disabled_emitter():
    """Create a disabled TelemetryEmitter for testing disabled state."""
    from src.telemetry.emitter import TelemetryEmitter

    return TelemetryEmitter(
        endpoint=None,
        enabled=False,
    )


@pytest.fixture
def emitter_with_small_buffer():
    """Create an emitter with a small buffer for testing overflow."""
    from src.telemetry.emitter import TelemetryEmitter

    return TelemetryEmitter(
        endpoint="http://test:8001/events",
        max_buffer_size=5,
        flush_threshold=3,
        enabled=True,
    )


@pytest.fixture
def emitter_with_low_threshold():
    """Create an emitter with low flush threshold for testing auto-flush."""
    from src.telemetry.emitter import TelemetryEmitter

    return TelemetryEmitter(
        endpoint="http://test:8001/events",
        flush_threshold=2,
        batch_size=10,
        enabled=True,
    )
