# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""tests for AgentIterationEvent emission.

Targets:
- `src/llm/tool_loop.py::_emit_agent_iteration` fires one event per LLM
  response, including the no-tool terminating iteration and the max-iteration
  synthesis call.
- Emission is skipped when telemetry context is missing or under-specified
  (no agent_type / parent_category / workspace / run_id).
- The caller-supplied `LLMTelemetryContext` is never mutated; per-iteration
  copies set the iteration field on a fresh instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from src.llm.tool_loop import (
    _emit_agent_iteration,
    _telemetry_for_iteration,
)
from src.llm.types import HonchoLLMCallResponse, LLMTelemetryContext
from src.telemetry.events import AgentIterationEvent, BaseEvent


def _response(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 25,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        finish_reasons=["stop"],
        tool_calls_made=tool_calls or [],
    )


class TestTelemetryForIteration:
    def test_returns_none_when_base_is_none(self):
        assert _telemetry_for_iteration(None, 1) is None

    def test_returns_fresh_copy_with_iteration_set(self):
        base = LLMTelemetryContext(
            workspace_name="ws",
            call_purpose="dialectic.answer",
            parent_category="dialectic",
            agent_type="dialectic",
            run_id="run-xyz",
            iteration=None,
            peer_name="user_peer",
        )

        copy_a = _telemetry_for_iteration(base, 3)
        copy_b = _telemetry_for_iteration(base, 4)

        assert copy_a is not None and copy_b is not None
        assert copy_a is not base and copy_b is not base
        # Original is never mutated.
        assert base.iteration is None
        assert copy_a.iteration == 3
        assert copy_b.iteration == 4
        # All other fields round-trip.
        assert copy_a.run_id == "run-xyz"
        assert copy_a.peer_name == "user_peer"


class TestEmitAgentIteration:
    def test_emits_event_with_tool_calls(self):
        emitted: list[BaseEvent] = []
        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            parent_category="dream",
            agent_type="deduction",
            run_id="run-1",
            observer="obs",
            observed="obj",
        )
        response = _response(
            tool_calls=[
                {"name": "search_memory", "id": "t1", "input": {}},
                {"name": "create_observations", "id": "t2", "input": {}},
            ],
            input_tokens=500,
            output_tokens=80,
            cache_read=12,
            cache_creation=5,
        )

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(telemetry, iteration=2, response=response)

        assert len(emitted) == 1
        event = emitted[0]
        assert isinstance(event, AgentIterationEvent)
        assert event.run_id == "run-1"
        assert event.parent_category == "dream"
        assert event.agent_type == "deduction"
        assert event.workspace_name == "ws"
        assert event.observer == "obs"
        assert event.observed == "obj"
        assert event.iteration == 2
        assert event.tool_calls == ["search_memory", "create_observations"]
        assert event.input_tokens == 500
        assert event.output_tokens == 80
        assert event.cache_read_tokens == 12
        assert event.cache_creation_tokens == 5

    def test_emits_terminating_iteration_with_empty_tool_calls(self):
        """The no-tool terminating iteration still counts. Empty tool_calls
        list must produce a valid AgentIterationEvent."""
        emitted: list[BaseEvent] = []
        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            parent_category="dialectic",
            agent_type="dialectic",
            run_id="run-2",
            peer_name="user_peer",
        )

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(telemetry, iteration=4, response=_response())

        assert len(emitted) == 1
        event = emitted[0]
        assert isinstance(event, AgentIterationEvent)
        assert event.tool_calls == []
        assert event.iteration == 4

    def test_skips_when_telemetry_is_none(self):
        emitted: list[BaseEvent] = []
        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(None, iteration=1, response=_response())
        assert emitted == []

    def test_skips_when_run_id_missing(self):
        emitted: list[BaseEvent] = []
        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            parent_category="dialectic",
            agent_type="dialectic",
            run_id=None,
        )
        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(telemetry, iteration=1, response=_response())
        assert emitted == []

    def test_skips_when_agent_type_or_parent_category_missing(self):
        """LLMCallCompletedEvent can fire without agent fields (system call),
        but agent.iteration is by definition an agent-loop event. If agent
        metadata is missing, skip emission rather than send a half-populated
        event."""
        emitted: list[BaseEvent] = []
        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            parent_category=None,
            agent_type="dialectic",
            run_id="run-3",
        )
        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(telemetry, iteration=1, response=_response())
        assert emitted == []

    def test_skips_when_workspace_missing(self):
        emitted: list[BaseEvent] = []
        telemetry = LLMTelemetryContext(
            workspace_name=None,
            parent_category="dream",
            agent_type="induction",
            run_id="run-4",
        )
        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_iteration(telemetry, iteration=1, response=_response())
        assert emitted == []

    def test_swallows_emit_failures(self):
        """Telemetry failures must not bleed into the LLM call path."""

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("emitter wedged")

        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            parent_category="dream",
            agent_type="deduction",
            run_id="run-5",
        )
        with patch("src.telemetry.events.emit", side_effect=explode):
            # Must not raise.
            _emit_agent_iteration(telemetry, iteration=1, response=_response())


def test_volume_class_is_high_volume():
    """emission targets a high-volume event class so the sampler
    can throttle iteration events independently of aggregates."""
    assert AgentIterationEvent.volume_class() == "high_volume"


class TestIterationScope:
    """`iteration_scope()` in src/utils/types.py captures Tokens for the
    per-loop ContextVars and resets them on exit. Defensive against a
    subsequent tool loop in the same asyncio Task seeing stale state from
    a previous loop (worker batches, tests using TestClient).
    """

    def test_resets_iteration_and_tool_call_state_on_exit(self):
        from src.utils.types import (
            get_current_iteration,
            get_current_provider_tool_call_id,
            get_current_tool_call_seq,
            get_last_tool_metadata,
            iteration_scope,
            set_current_iteration,
            set_current_tool_call_seq,
            set_last_tool_metadata,
        )

        # Pre-scope: defaults.
        assert get_current_iteration() == 0
        assert get_current_tool_call_seq() == 0
        assert get_current_provider_tool_call_id() is None
        assert get_last_tool_metadata() == {}

        with iteration_scope():
            set_current_iteration(7)
            set_current_tool_call_seq(3, "toolu_abc")
            set_last_tool_metadata({"k": "v"})
            assert get_current_iteration() == 7
            assert get_current_tool_call_seq() == 3
            assert get_current_provider_tool_call_id() == "toolu_abc"
            assert get_last_tool_metadata() == {"k": "v"}

        # Post-scope: every ContextVar reset to pre-scope state.
        assert get_current_iteration() == 0
        assert get_current_tool_call_seq() == 0
        assert get_current_provider_tool_call_id() is None
        assert get_last_tool_metadata() == {}

    def test_resets_on_exception(self):
        """Exception inside the block still triggers the reset path."""
        import pytest

        from src.utils.types import (
            get_current_iteration,
            iteration_scope,
            set_current_iteration,
        )

        with pytest.raises(RuntimeError, match="boom"), iteration_scope():
            set_current_iteration(5)
            raise RuntimeError("boom")

        assert get_current_iteration() == 0

    def test_sequential_scopes_do_not_leak(self):
        """Two back-to-back scopes (mimicking sequential tool loops) — the
        second sees a clean baseline, not stale values from the first."""
        from src.utils.types import (
            get_current_iteration,
            iteration_scope,
            set_current_iteration,
        )

        with iteration_scope():
            set_current_iteration(9)

        with iteration_scope():
            # Inside scope #2: iteration starts at 0 (the scope reset it),
            # not at 9 from the prior scope.
            assert get_current_iteration() == 0
