# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false
"""Unit tests for telemetry event classes.

Tests all 12 event types for:
- Correct instantiation with required fields
- event_type(), schema_version(), category() class methods
- get_resource_id() returns expected format
- generate_id() is deterministic (same inputs = same ID)
- Timestamp defaults to UTC now
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.telemetry.events.agent import (
    AgentIterationEvent,
    AgentToolConclusionsCreatedEvent,
    AgentToolConclusionsDeletedEvent,
    AgentToolPeerCardUpdatedEvent,
    AgentToolSummaryCreatedEvent,
)
from src.telemetry.events.base import BaseEvent, generate_event_id
from src.telemetry.events.deletion import DeletionCompletedEvent
from src.telemetry.events.dialectic import DialecticCompletedEvent
from src.telemetry.events.dream import DreamRunEvent, DreamSpecialistEvent
from src.telemetry.events.reconciliation import (
    CleanupStaleItemsCompletedEvent,
    SyncVectorsCompletedEvent,
)
from src.telemetry.events.representation import RepresentationCompletedEvent

# =============================================================================
# Tests for generate_event_id function
# =============================================================================


class TestGenerateEventId:
    """Tests for the generate_event_id function."""

    def test_deterministic_output(self, fixed_timestamp: datetime):
        """Same inputs always produce the same ID."""
        event_type = "test.event"
        resource_id = "resource_123"

        id1 = generate_event_id(event_type, fixed_timestamp, resource_id)
        id2 = generate_event_id(event_type, fixed_timestamp, resource_id)

        assert id1 == id2

    def test_different_event_types_produce_different_ids(
        self, fixed_timestamp: datetime
    ):
        """Different event types produce different IDs."""
        resource_id = "resource_123"

        id1 = generate_event_id("type.a", fixed_timestamp, resource_id)
        id2 = generate_event_id("type.b", fixed_timestamp, resource_id)

        assert id1 != id2

    def test_different_timestamps_produce_different_ids(self):
        """Different timestamps produce different IDs."""
        event_type = "test.event"
        resource_id = "resource_123"
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC)

        id1 = generate_event_id(event_type, ts1, resource_id)
        id2 = generate_event_id(event_type, ts2, resource_id)

        assert id1 != id2

    def test_different_resource_ids_produce_different_ids(
        self, fixed_timestamp: datetime
    ):
        """Different resource IDs produce different IDs."""
        event_type = "test.event"

        id1 = generate_event_id(event_type, fixed_timestamp, "resource_a")
        id2 = generate_event_id(event_type, fixed_timestamp, "resource_b")

        assert id1 != id2

    def test_id_format(self, fixed_timestamp: datetime):
        """Event ID has correct format: evt_{base64_hash}."""
        event_id = generate_event_id("test.event", fixed_timestamp, "resource_123")

        assert event_id.startswith("evt_")
        # Base64url encoded 16 bytes = 22 chars (without padding)
        assert len(event_id) == 4 + 22  # "evt_" + 22 chars


# =============================================================================
# Tests for BaseEvent class
# =============================================================================


class TestBaseEvent:
    """Tests for BaseEvent base class behavior."""

    def test_timestamp_defaults_to_utc_now(self):
        """Events without explicit timestamp get current UTC time."""
        # Create event without timestamp
        event = RepresentationCompletedEvent(
            workspace_name="test",
            session_name="test_session",
            observed="user",
            queue_items_processed=1,
            earliest_message_id="msg_1",
            latest_message_id="msg_2",
            message_count=2,
            explicit_conclusion_count=1,
            context_preparation_ms=10.0,
            llm_call_ms=100.0,
            total_duration_ms=110.0,
            input_tokens=100,
            output_tokens=50,
        )

        # Timestamp should be set and be UTC
        assert event.timestamp is not None
        assert event.timestamp.tzinfo == UTC

    def test_get_resource_id_not_implemented_on_base(self):
        """BaseEvent.get_resource_id raises NotImplementedError."""
        # Can't instantiate BaseEvent directly, but we can test the method
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            BaseEvent.get_resource_id(BaseEvent(timestamp=datetime.now(UTC)))


# =============================================================================
# Tests for RepresentationCompletedEvent
# =============================================================================


class TestRepresentationCompletedEvent:
    """Tests for RepresentationCompletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert RepresentationCompletedEvent.event_type() == "representation.completed"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert RepresentationCompletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert RepresentationCompletedEvent.category() == "representation"

    def test_get_resource_id(
        self, sample_representation_event: RepresentationCompletedEvent
    ):
        """get_resource_id() returns workspace:session:message format."""
        resource_id = sample_representation_event.get_resource_id()
        assert resource_id == "test_workspace:test_session:msg_010"

    def test_generate_id_deterministic(
        self, sample_representation_event: RepresentationCompletedEvent
    ):
        """generate_id() produces same ID for same event."""
        id1 = sample_representation_event.generate_id()
        id2 = sample_representation_event.generate_id()
        assert id1 == id2
        assert id1.startswith("evt_")

    def test_required_fields(self):
        """Event requires all mandatory fields."""
        with pytest.raises(ValidationError):
            RepresentationCompletedEvent()  # pyright: ignore[reportCallIssue]

    def test_model_dump(
        self, sample_representation_event: RepresentationCompletedEvent
    ):
        """Event can be serialized to dict."""
        data = sample_representation_event.model_dump(mode="json")
        assert data["workspace_name"] == "test_workspace"
        assert data["message_count"] == 10
        assert data["explicit_conclusion_count"] == 5


# =============================================================================
# Tests for DreamRunEvent
# =============================================================================


class TestDreamRunEvent:
    """Tests for DreamRunEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert DreamRunEvent.event_type() == "dream.run"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert DreamRunEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert DreamRunEvent.category() == "dream"

    def test_get_resource_id(self, sample_dream_run_event: DreamRunEvent):
        """get_resource_id() returns run_id."""
        assert sample_dream_run_event.get_resource_id() == "abc12345"

    def test_generate_id_deterministic(self, sample_dream_run_event: DreamRunEvent):
        """generate_id() produces same ID for same event."""
        id1 = sample_dream_run_event.generate_id()
        id2 = sample_dream_run_event.generate_id()
        assert id1 == id2

    def test_specialists_run_list(self, sample_dream_run_event: DreamRunEvent):
        """specialists_run contains expected values."""
        assert "deduction" in sample_dream_run_event.specialists_run
        assert "induction" in sample_dream_run_event.specialists_run


# =============================================================================
# Tests for DreamSpecialistEvent
# =============================================================================


class TestDreamSpecialistEvent:
    """Tests for DreamSpecialistEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert DreamSpecialistEvent.event_type() == "dream.specialist"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert DreamSpecialistEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert DreamSpecialistEvent.category() == "dream"

    def test_get_resource_id(self, sample_dream_specialist_event: DreamSpecialistEvent):
        """get_resource_id() returns run_id:specialist_type format."""
        assert sample_dream_specialist_event.get_resource_id() == "abc12345:deduction"

    def test_generate_id_deterministic(
        self, sample_dream_specialist_event: DreamSpecialistEvent
    ):
        """generate_id() produces same ID for same event."""
        id1 = sample_dream_specialist_event.generate_id()
        id2 = sample_dream_specialist_event.generate_id()
        assert id1 == id2


# =============================================================================
# Tests for DialecticCompletedEvent
# =============================================================================


class TestDialecticCompletedEvent:
    """Tests for DialecticCompletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert DialecticCompletedEvent.event_type() == "dialectic.completed"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert DialecticCompletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert DialecticCompletedEvent.category() == "dialectic"

    def test_get_resource_id(self, sample_dialectic_event: DialecticCompletedEvent):
        """get_resource_id() returns run_id."""
        assert sample_dialectic_event.get_resource_id() == "def67890"

    def test_generate_id_deterministic(
        self, sample_dialectic_event: DialecticCompletedEvent
    ):
        """generate_id() produces same ID for same event."""
        id1 = sample_dialectic_event.generate_id()
        id2 = sample_dialectic_event.generate_id()
        assert id1 == id2

    def test_optional_session_fields(self, fixed_timestamp: datetime):
        """session_id and session_name are optional."""
        event = DialecticCompletedEvent(
            timestamp=fixed_timestamp,
            run_id="test123",
            workspace_name="test",
            peer_name="user",
            reasoning_level="low",
            total_duration_ms=1000.0,
            input_tokens=500,
            output_tokens=100,
        )
        assert event.session_name is None

    def test_cache_token_defaults(self, fixed_timestamp: datetime):
        """Cache tokens default to 0."""
        event = DialecticCompletedEvent(
            timestamp=fixed_timestamp,
            run_id="test123",
            workspace_name="test",
            peer_name="user",
            reasoning_level="low",
            total_duration_ms=1000.0,
            input_tokens=500,
            output_tokens=100,
        )
        assert event.cache_read_tokens == 0
        assert event.cache_creation_tokens == 0


# =============================================================================
# Tests for AgentIterationEvent
# =============================================================================


class TestAgentIterationEvent:
    """Tests for AgentIterationEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert AgentIterationEvent.event_type() == "agent.iteration"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert AgentIterationEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert AgentIterationEvent.category() == "agent"

    def test_get_resource_id(self, sample_agent_iteration_event: AgentIterationEvent):
        """get_resource_id() returns run_id:iteration format."""
        assert sample_agent_iteration_event.get_resource_id() == "abc12345:3"

    def test_generate_id_deterministic(
        self, sample_agent_iteration_event: AgentIterationEvent
    ):
        """generate_id() produces same ID for same event."""
        id1 = sample_agent_iteration_event.generate_id()
        id2 = sample_agent_iteration_event.generate_id()
        assert id1 == id2

    def test_tool_calls_list(self, sample_agent_iteration_event: AgentIterationEvent):
        """tool_calls is a list of strings."""
        assert isinstance(sample_agent_iteration_event.tool_calls, list)
        assert "search_memory" in sample_agent_iteration_event.tool_calls

    def test_optional_peer_fields(self, fixed_timestamp: datetime):
        """observer, observed, peer_id are optional."""
        event = AgentIterationEvent(
            timestamp=fixed_timestamp,
            run_id="test123",
            parent_category="dialectic",
            agent_type="dialectic",
            workspace_name="test",
            iteration=1,
            input_tokens=100,
            output_tokens=50,
        )
        assert event.observer is None
        assert event.observed is None


# =============================================================================
# Tests for AgentToolConclusionsCreatedEvent
# =============================================================================


class TestAgentToolConclusionsCreatedEvent:
    """Tests for AgentToolConclusionsCreatedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert (
            AgentToolConclusionsCreatedEvent.event_type()
            == "agent.tool.conclusions.created"
        )

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert AgentToolConclusionsCreatedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert AgentToolConclusionsCreatedEvent.category() == "agent"

    def test_get_resource_id(
        self, sample_conclusions_created_event: AgentToolConclusionsCreatedEvent
    ):
        """get_resource_id() returns run_id:iteration:conclusions_created format."""
        assert (
            sample_conclusions_created_event.get_resource_id()
            == "abc12345:3:conclusions_created"
        )

    def test_levels_list(
        self, sample_conclusions_created_event: AgentToolConclusionsCreatedEvent
    ):
        """levels is a list matching conclusion_count."""
        assert len(sample_conclusions_created_event.levels) == 5
        assert sample_conclusions_created_event.conclusion_count == 5


# =============================================================================
# Tests for AgentToolConclusionsDeletedEvent
# =============================================================================


class TestAgentToolConclusionsDeletedEvent:
    """Tests for AgentToolConclusionsDeletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert (
            AgentToolConclusionsDeletedEvent.event_type()
            == "agent.tool.conclusions.deleted"
        )

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert AgentToolConclusionsDeletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert AgentToolConclusionsDeletedEvent.category() == "agent"

    def test_get_resource_id(
        self, sample_conclusions_deleted_event: AgentToolConclusionsDeletedEvent
    ):
        """get_resource_id() returns run_id:iteration:conclusions_deleted format."""
        assert (
            sample_conclusions_deleted_event.get_resource_id()
            == "abc12345:5:conclusions_deleted"
        )


# =============================================================================
# Tests for AgentToolPeerCardUpdatedEvent
# =============================================================================


class TestAgentToolPeerCardUpdatedEvent:
    """Tests for AgentToolPeerCardUpdatedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert (
            AgentToolPeerCardUpdatedEvent.event_type() == "agent.tool.peer_card.updated"
        )

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert AgentToolPeerCardUpdatedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert AgentToolPeerCardUpdatedEvent.category() == "agent"

    def test_get_resource_id(
        self, sample_peer_card_updated_event: AgentToolPeerCardUpdatedEvent
    ):
        """get_resource_id() returns run_id:iteration:peer_card_updated format."""
        assert (
            sample_peer_card_updated_event.get_resource_id()
            == "abc12345:7:peer_card_updated"
        )


# =============================================================================
# Tests for AgentToolSummaryCreatedEvent
# =============================================================================


class TestAgentToolSummaryCreatedEvent:
    """Tests for AgentToolSummaryCreatedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert AgentToolSummaryCreatedEvent.event_type() == "agent.tool.summary.created"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert AgentToolSummaryCreatedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert AgentToolSummaryCreatedEvent.category() == "agent"

    def test_get_resource_id(
        self, sample_summary_created_event: AgentToolSummaryCreatedEvent
    ):
        """get_resource_id() returns run_id:iteration:summary_created format."""
        assert (
            sample_summary_created_event.get_resource_id()
            == "ghi11111:1:summary_created"
        )

    def test_summary_type_values(self, fixed_timestamp: datetime):
        """summary_type accepts 'short' and 'long'."""
        for summary_type in ["short", "long"]:
            event = AgentToolSummaryCreatedEvent(
                timestamp=fixed_timestamp,
                run_id="test123",
                iteration=1,
                parent_category="representation",
                agent_type="summarizer",
                workspace_name="test",
                session_name="test_session",
                message_id="msg_1",
                message_count=10,
                message_seq_in_session=10,
                summary_type=summary_type,
                input_tokens=100,
                output_tokens=50,
            )
            assert event.summary_type == summary_type


# =============================================================================
# Tests for DeletionCompletedEvent
# =============================================================================


class TestDeletionCompletedEvent:
    """Tests for DeletionCompletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert DeletionCompletedEvent.event_type() == "deletion.completed"

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert DeletionCompletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert DeletionCompletedEvent.category() == "deletion"

    def test_get_resource_id(self, sample_deletion_event: DeletionCompletedEvent):
        """get_resource_id() returns workspace:type:resource format."""
        assert (
            sample_deletion_event.get_resource_id()
            == "test_workspace:workspace:ws_123abc"
        )

    def test_cascade_counts_default_to_zero(self, fixed_timestamp: datetime):
        """Cascade counts default to 0 for non-workspace deletions."""
        event = DeletionCompletedEvent(
            timestamp=fixed_timestamp,
            workspace_name="test",
            deletion_type="session",
            resource_id="sess_456",
            success=True,
        )
        assert event.peers_deleted == 0
        assert event.sessions_deleted == 0
        assert event.messages_deleted == 0
        assert event.conclusions_deleted == 0

    def test_error_message_optional(self, fixed_timestamp: datetime):
        """error_message is optional and defaults to None."""
        event = DeletionCompletedEvent(
            timestamp=fixed_timestamp,
            workspace_name="test",
            deletion_type="session",
            resource_id="sess_456",
            success=True,
        )
        assert event.error_message is None

    def test_failed_deletion_with_error(self, fixed_timestamp: datetime):
        """Failed deletion can include error message."""
        event = DeletionCompletedEvent(
            timestamp=fixed_timestamp,
            workspace_name="test",
            deletion_type="session",
            resource_id="sess_456",
            success=False,
            error_message="Foreign key constraint violation",
        )
        assert event.success is False
        assert event.error_message == "Foreign key constraint violation"


# =============================================================================
# Tests for SyncVectorsCompletedEvent
# =============================================================================


class TestSyncVectorsCompletedEvent:
    """Tests for SyncVectorsCompletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert (
            SyncVectorsCompletedEvent.event_type()
            == "reconciliation.sync_vectors.completed"
        )

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert SyncVectorsCompletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert SyncVectorsCompletedEvent.category() == "reconciliation"

    def test_get_resource_id(
        self, sample_sync_vectors_event: SyncVectorsCompletedEvent
    ):
        """get_resource_id() returns fixed string."""
        assert sample_sync_vectors_event.get_resource_id() == "sync_vectors"

    def test_metrics_default_to_zero(self, fixed_timestamp: datetime):
        """Sync metrics default to 0."""
        event = SyncVectorsCompletedEvent(
            timestamp=fixed_timestamp,
            total_duration_ms=1000.0,
        )
        assert event.documents_synced == 0
        assert event.documents_failed == 0
        assert event.message_embeddings_synced == 0
        assert event.message_embeddings_failed == 0


# =============================================================================
# Tests for CleanupStaleItemsCompletedEvent
# =============================================================================


class TestCleanupStaleItemsCompletedEvent:
    """Tests for CleanupStaleItemsCompletedEvent."""

    def test_event_type(self):
        """event_type() returns correct value."""
        assert (
            CleanupStaleItemsCompletedEvent.event_type()
            == "reconciliation.cleanup_stale_items.completed"
        )

    def test_schema_version(self):
        """schema_version() returns correct value."""
        assert CleanupStaleItemsCompletedEvent.schema_version() == 1

    def test_category(self):
        """category() returns correct value."""
        assert CleanupStaleItemsCompletedEvent.category() == "reconciliation"

    def test_get_resource_id(
        self, sample_cleanup_event: CleanupStaleItemsCompletedEvent
    ):
        """get_resource_id() returns fixed string."""
        assert sample_cleanup_event.get_resource_id() == "cleanup_stale_items"

    def test_cleanup_metrics_default_to_zero(self, fixed_timestamp: datetime):
        """Cleanup metrics default to 0."""
        event = CleanupStaleItemsCompletedEvent(
            timestamp=fixed_timestamp,
            total_duration_ms=500.0,
        )
        assert event.documents_cleaned == 0
        assert event.queue_items_cleaned == 0


# =============================================================================
# Parametrized tests across all event types
# =============================================================================


class TestAllEventTypes:
    """Parametrized tests that run across all event types."""

    def test_all_events_have_timestamp(self, all_sample_events: list[BaseEvent]):
        """All events have a timestamp field."""
        for event in all_sample_events:
            assert hasattr(event, "timestamp")
            assert event.timestamp is not None

    def test_all_events_generate_valid_ids(self, all_sample_events: list[BaseEvent]):
        """All events generate valid event IDs."""
        for event in all_sample_events:
            event_id = event.generate_id()
            assert event_id.startswith("evt_")
            assert len(event_id) == 26  # "evt_" + 22 chars

    def test_all_events_have_event_type(self, all_sample_events: list[BaseEvent]):
        """All events return a non-empty event_type."""
        for event in all_sample_events:
            event_type = event.event_type()
            assert event_type
            assert isinstance(event_type, str)

    def test_all_events_have_category(self, all_sample_events: list[BaseEvent]):
        """All events return a non-empty category."""
        for event in all_sample_events:
            category = event.category()
            assert category
            assert isinstance(category, str)

    def test_all_events_have_schema_version(self, all_sample_events: list[BaseEvent]):
        """All events return a positive schema version."""
        for event in all_sample_events:
            version = event.schema_version()
            assert version >= 1
            assert isinstance(version, int)

    def test_all_events_can_serialize_to_json(self, all_sample_events: list[BaseEvent]):
        """All events can be serialized to JSON-compatible dict."""
        for event in all_sample_events:
            data = event.model_dump(mode="json")
            assert isinstance(data, dict)
            # Timestamp should be serialized as ISO string
            assert "timestamp" in data

    def test_all_events_have_get_resource_id(self, all_sample_events: list[BaseEvent]):
        """All events implement get_resource_id()."""
        for event in all_sample_events:
            resource_id = event.get_resource_id()
            assert resource_id
            assert isinstance(resource_id, str)
