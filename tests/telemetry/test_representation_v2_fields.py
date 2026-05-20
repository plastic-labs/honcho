# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""tests for RepresentationCompletedEvent additive fields + truncation.

Targets:
- Schema stays at v2 (additive, no bump). Existing `input_tokens` semantics
  unchanged.
- fields are defaultable (no breakage for callers that ignore them)
  and round-trip through Pydantic serialization.
- `HonchoLLMCallResponse.hit_input_token_cap` defaults to False but can be
  flipped by the tool-less cap-detection path in `src/llm/api.py`.
"""

from __future__ import annotations

from src.llm.types import HonchoLLMCallResponse
from src.telemetry.events.representation import RepresentationCompletedEvent


class TestRepresentationV2AdditiveFields:
    def test_schema_stays_at_v2(self):
        """is additive — schema_version must NOT bump to 3."""
        assert RepresentationCompletedEvent.schema_version() == 2

    def test_new_fields_are_optional(self):
        """Existing callers must keep working without supplying any new
        fields. All new fields default."""
        event = RepresentationCompletedEvent(
            workspace_name="ws",
            session_name="s",
            observed="user",
            queue_items_processed=1,
            earliest_message_id="m1",
            latest_message_id="m1",
            message_count=1,
            explicit_conclusion_count=0,
            context_preparation_ms=10.0,
            llm_call_ms=100.0,
            total_duration_ms=110.0,
            input_tokens=100,
            total_input_tokens=200,
            output_tokens=50,
        )
        # All fields land with defaults.
        assert event.queued_message_count == 0
        assert event.prompt_message_count == 0
        assert event.prompt_message_tokens == 0
        assert event.extra_context_message_count == 0
        assert event.extra_context_tokens == 0
        assert event.prompt_scaffold_tokens == 0
        assert event.batch_max_tokens == 0
        assert event.max_input_tokens == 0
        assert event.was_flush_enabled is False
        assert event.hit_batch_token_cap is False
        assert event.hit_input_token_cap is False
        assert event.observer_count == 0

    def test_input_tokens_semantics_preserved(self):
        """The downstream metering key must remain 'queued-message tokens'.

        Added many fields, but `input_tokens` MUST stay as the downstream
        metering key for representation.completed. Don't rename or
        repurpose without coordinating with consumers.
        """
        event = RepresentationCompletedEvent(
            workspace_name="ws",
            session_name="s",
            observed="user",
            queue_items_processed=2,
            earliest_message_id="m1",
            latest_message_id="m5",
            message_count=5,
            explicit_conclusion_count=3,
            context_preparation_ms=10.0,
            llm_call_ms=100.0,
            total_duration_ms=110.0,
            input_tokens=300,  # ← queued message tokens; the billing key
            total_input_tokens=900,  # provider-side total
            output_tokens=50,
            queued_message_count=2,
            prompt_message_count=5,
            prompt_message_tokens=800,
            extra_context_message_count=3,
            extra_context_tokens=500,
            prompt_scaffold_tokens=100,
        )
        # input_tokens and queued_message_count should describe the same set
        # (queue items being reasoned about) — assert the conceptual link.
        assert event.input_tokens == 300
        assert event.queued_message_count == 2
        # And the breakdown adds up: extra + scaffold + queued ≈ total provider input
        # (it's an approximation — provider includes formatting overhead).
        assert (
            event.extra_context_tokens
            + event.input_tokens
            + event.prompt_scaffold_tokens
            == 900
        )
        assert event.total_input_tokens == 900

    def test_cap_hit_flags(self):
        event = RepresentationCompletedEvent(
            workspace_name="ws",
            session_name="s",
            observed="user",
            queue_items_processed=10,
            earliest_message_id="m1",
            latest_message_id="m10",
            message_count=10,
            explicit_conclusion_count=5,
            context_preparation_ms=10.0,
            llm_call_ms=100.0,
            total_duration_ms=110.0,
            input_tokens=20_000,
            total_input_tokens=23_000,
            output_tokens=500,
            batch_max_tokens=20_000,
            max_input_tokens=23_000,
            was_flush_enabled=True,
            hit_batch_token_cap=True,
            hit_input_token_cap=True,
            observer_count=2,
        )
        assert event.was_flush_enabled is True
        assert event.hit_batch_token_cap is True
        assert event.hit_input_token_cap is True
        assert event.batch_max_tokens == 20_000
        assert event.max_input_tokens == 23_000
        assert event.observer_count == 2

    def test_model_dump_includes_new_fields(self):
        event = RepresentationCompletedEvent(
            workspace_name="ws",
            session_name="s",
            observed="user",
            queue_items_processed=1,
            earliest_message_id="m1",
            latest_message_id="m1",
            message_count=1,
            explicit_conclusion_count=0,
            context_preparation_ms=10.0,
            llm_call_ms=100.0,
            total_duration_ms=110.0,
            input_tokens=100,
            total_input_tokens=150,
            output_tokens=50,
            hit_batch_token_cap=True,
        )
        data = event.model_dump(mode="json")
        for field in (
            "queued_message_count",
            "prompt_message_count",
            "prompt_message_tokens",
            "extra_context_message_count",
            "extra_context_tokens",
            "prompt_scaffold_tokens",
            "batch_max_tokens",
            "max_input_tokens",
            "was_flush_enabled",
            "hit_batch_token_cap",
            "hit_input_token_cap",
            "observer_count",
        ):
            assert field in data, f"missing field: {field}"
        assert data["hit_batch_token_cap"] is True


class TestHitInputTokenCapFlag:
    """`HonchoLLMCallResponse.hit_input_token_cap` is the bridge between the
    tool-less cap-detection path in src/llm/api.py and the deriver's
    `hit_input_token_cap` field on RepresentationCompletedEvent.

    The flag is token-based — it fires whenever the original input exceeded
    `max_input_tokens`, whether or not message truncation could actually
    shrink the input below cap (the deriver's single-prompt case can't).
    """

    def test_defaults_to_false(self):
        response = HonchoLLMCallResponse(
            content="hi",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
        assert response.hit_input_token_cap is False

    def test_can_be_flipped(self):
        response = HonchoLLMCallResponse(
            content="hi",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
        response.hit_input_token_cap = True
        assert response.hit_input_token_cap is True
