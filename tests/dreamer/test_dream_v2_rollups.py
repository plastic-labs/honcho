# pyright: reportPrivateUsage=false
"""tests: DreamSpecialistEvent + DreamRunEvent v2 fields.

Targets:
- Schema bumps to v2 (DreamRunEvent + DreamSpecialistEvent).
- Specialist rollups come from `tool_result_metadata`, NOT tool-name counting.
- DreamRunEvent carries the scheduler reasons threaded through the dream
  queue payload.
"""

from __future__ import annotations

from src.telemetry.events.dream import DreamRunEvent, DreamSpecialistEvent
from src.utils.queue_payload import DreamPayload


class TestSchemaVersionsBumpedToV2:
    def test_dream_run_event_at_v2(self):
        assert DreamRunEvent.schema_version() == 2

    def test_dream_specialist_event_at_v2(self):
        assert DreamSpecialistEvent.schema_version() == 2


class TestDreamRunEventV2Fields:
    def test_scheduler_fields_default(self):
        """Existing callers that don't supply scheduling fields must
        still construct a valid event — all new fields default to None or 0."""
        event = DreamRunEvent(
            run_id="abc",
            workspace_name="ws",
            session_name=None,
            observer="o",
            observed="user",
            specialists_run=["deduction", "induction"],
            deduction_success=True,
            induction_success=True,
            total_iterations=10,
            total_input_tokens=100,
            total_output_tokens=20,
            total_duration_ms=1000.0,
        )
        assert event.dream_type is None
        assert event.enabled_types_count == 0
        assert event.trigger_reason is None
        assert event.delay_reason is None
        assert event.documents_since_last_dream_at_schedule is None
        assert event.document_threshold is None

    def test_scheduler_reasons_round_trip(self):
        event = DreamRunEvent(
            run_id="abc",
            workspace_name="ws",
            session_name=None,
            observer="o",
            observed="user",
            specialists_run=["deduction"],
            deduction_success=True,
            induction_success=False,
            total_iterations=5,
            total_input_tokens=50,
            total_output_tokens=10,
            total_duration_ms=500.0,
            dream_type="omni",
            enabled_types_count=1,
            trigger_reason="document_threshold",
            delay_reason="idle_timeout",
            documents_since_last_dream_at_schedule=60,
            document_threshold=50,
        )
        assert event.dream_type == "omni"
        assert event.trigger_reason == "document_threshold"
        assert event.delay_reason == "idle_timeout"
        assert event.documents_since_last_dream_at_schedule == 60
        assert event.document_threshold == 50

    def test_threshold_and_delay_are_separate(self):
        """The two scheduler gates are intentionally separate fields. The
        snapshot semantics differ: trigger_reason describes WHY the dream
        was scheduled (which gate tripped); delay_reason describes WHEN it
        will fire (idle vs immediate)."""
        event = DreamRunEvent(
            run_id="abc",
            workspace_name="ws",
            session_name=None,
            observer="o",
            observed="user",
            specialists_run=["induction"],
            deduction_success=False,
            induction_success=True,
            total_iterations=3,
            total_input_tokens=30,
            total_output_tokens=5,
            total_duration_ms=100.0,
            trigger_reason="document_threshold",
            delay_reason="immediate",
        )
        # trigger_reason captures the WHY; delay_reason captures the WHEN.
        # They are separate dimensions — flattening into one field would lose
        # the gate semantics that was specifically designed to expose.
        assert event.trigger_reason != event.delay_reason


class TestDreamSpecialistEventV2Rollups:
    def test_rollup_fields_default(self):
        event = DreamSpecialistEvent(
            run_id="abc",
            specialist_type="deduction",
            workspace_name="ws",
            observer="o",
            observed="user",
            iterations=3,
            tool_calls_count=5,
            input_tokens=100,
            output_tokens=20,
            duration_ms=500.0,
            success=True,
        )
        assert event.created_observation_count == 0
        assert event.deleted_observation_count == 0
        assert event.peer_card_updated is False
        assert event.search_tool_calls_count == 0

    def test_observation_counts_are_observation_truth_not_call_counts(self):
        """The whole point of sourcing rollups from ToolResult.metadata
        instead of tool-name counts: a single `create_observations` call can
        produce N observations (or zero on validation failure). must
        report observation truth, not call truth."""
        event = DreamSpecialistEvent(
            run_id="abc",
            specialist_type="deduction",
            workspace_name="ws",
            observer="o",
            observed="user",
            iterations=2,
            # Two create_observations CALLS, but they produced 7 observations
            # together (e.g. one batch of 5, one batch of 2). reports
            # 7 (the metadata-sourced truth), not 2 (the call count).
            tool_calls_count=2,
            input_tokens=100,
            output_tokens=20,
            duration_ms=500.0,
            success=True,
            created_observation_count=7,
        )
        assert event.tool_calls_count == 2
        assert event.created_observation_count == 7


class TestDreamPayloadSchedulerFields:
    def test_payload_defaults(self):
        from src.schemas import DreamType

        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer="o",
            observed="user",
        )
        assert payload.trigger_reason is None
        assert payload.delay_reason is None
        assert payload.documents_since_last_dream_at_schedule is None
        assert payload.document_threshold is None

    def test_payload_threads_scheduler_reasons(self):
        from src.schemas import DreamType

        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer="o",
            observed="user",
            trigger_reason="document_threshold",
            delay_reason="idle_timeout",
            documents_since_last_dream_at_schedule=55,
            document_threshold=50,
        )
        # Round-trip through serialization → deserialization mimics what
        # happens between scheduler enqueue and consumer dequeue.
        data = payload.model_dump(mode="json")
        restored = DreamPayload(**data)
        assert restored.trigger_reason == "document_threshold"
        assert restored.delay_reason == "idle_timeout"
        assert restored.documents_since_last_dream_at_schedule == 55
        assert restored.document_threshold == 50
