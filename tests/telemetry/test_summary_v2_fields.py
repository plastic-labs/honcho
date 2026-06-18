# pyright: reportPrivateUsage=false
"""tests for AgentToolSummaryCreatedEvent additive token breakdown.

Targets:
- Schema bumps to v2.
- `input_tokens` semantic preserved (provider-side input).
- New breakdown fields (`previous_summary_tokens`, `message_tokens`,
  `prompt_scaffold_tokens`) default cleanly and round-trip.
- The conceptual relationship: message_tokens + previous_summary_tokens
  describes the *user-data* portion of the input; prompt_scaffold_tokens is
  the static instruction portion. Together they approximate the provider
  input_tokens (modulo formatting overhead).
"""

from __future__ import annotations

from src.telemetry.events.agent import AgentToolSummaryCreatedEvent


class TestAdditiveFields:
    def test_new_fields_default_to_zero(self):
        """Callers that omit the breakdown fields must construct valid events."""
        event = AgentToolSummaryCreatedEvent(
            run_id="r",
            iteration=0,
            parent_category="deriver",
            agent_type="summarizer",
            workspace_name="ws",
            session_name="s",
            message_id="m1",
            message_count=10,
            message_seq_in_session=10,
            summary_type="short",
            input_tokens=1000,
            output_tokens=100,
        )
        assert event.previous_summary_tokens == 0
        assert event.message_tokens == 0
        assert event.prompt_scaffold_tokens == 0

    def test_input_tokens_semantic_preserved(self):
        """`input_tokens` continues to be the provider-side LLM input count.
        We deliberately do NOT add a redundant `provider_input_tokens` —
        the existing field already serves that purpose, and a duplicate
        would silently fork downstream queries."""
        event = AgentToolSummaryCreatedEvent(
            run_id="r",
            iteration=0,
            parent_category="deriver",
            agent_type="summarizer",
            workspace_name="ws",
            session_name="s",
            message_id="m1",
            message_count=10,
            message_seq_in_session=10,
            summary_type="long",
            input_tokens=5000,  # provider-reported total
            output_tokens=400,
            previous_summary_tokens=500,
            message_tokens=4000,
            prompt_scaffold_tokens=400,
        )
        # message + prev_summary + scaffold ≈ input_tokens (small drift from
        # provider-side formatting overhead is expected).
        breakdown_sum = (
            event.message_tokens
            + event.previous_summary_tokens
            + event.prompt_scaffold_tokens
        )
        assert breakdown_sum <= event.input_tokens + 200  # allow small overhead
        assert event.input_tokens == 5000

    def test_first_summary_has_zero_previous_summary_tokens(self):
        """When there's no prior summary for the session, the breakdown
        carries `previous_summary_tokens=0` so analytics can distinguish
        first-summary calls from rollup calls."""
        event = AgentToolSummaryCreatedEvent(
            run_id="r",
            iteration=0,
            parent_category="deriver",
            agent_type="summarizer",
            workspace_name="ws",
            session_name="s",
            message_id="m1",
            message_count=5,
            message_seq_in_session=5,
            summary_type="short",
            input_tokens=1500,
            output_tokens=150,
            previous_summary_tokens=0,  # first summary for this session
            message_tokens=1200,
            prompt_scaffold_tokens=300,
        )
        assert event.previous_summary_tokens == 0
        assert event.message_tokens > 0

    def test_model_dump_includes_breakdown_fields(self):
        event = AgentToolSummaryCreatedEvent(
            run_id="r",
            iteration=0,
            parent_category="deriver",
            agent_type="summarizer",
            workspace_name="ws",
            session_name="s",
            message_id="m1",
            message_count=10,
            message_seq_in_session=10,
            summary_type="short",
            input_tokens=1000,
            output_tokens=100,
            previous_summary_tokens=100,
            message_tokens=700,
            prompt_scaffold_tokens=200,
        )
        data = event.model_dump(mode="json")
        for field in (
            "previous_summary_tokens",
            "message_tokens",
            "prompt_scaffold_tokens",
        ):
            assert field in data, f"missing field: {field}"
        assert data["message_tokens"] == 700
