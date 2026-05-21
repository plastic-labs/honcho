# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedFunction=false
"""tests for AgentToolCallCompletedEvent emission.

Targets:
- `_emit_agent_tool_call_completed` in `src/utils/agent_tools.py` builds a
  well-formed event from ToolContext + per-call metadata.
- ToolResult dataclass behaves like a string for `in` / `str()` so tests of
  the existing handler contract keep working.
- The `tool_call_seq` ContextVar disambiguates resource ids when the same
  tool is called twice in one iteration.
- event opts into the high-volume sampler.
- Search handlers publish search-specific metadata
  (top_k / used_embedding / query_tokens / results_count) so analytics can
  filter by retrieval intent.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, final
from unittest.mock import patch

import pytest

from src.telemetry.events import AgentToolCallCompletedEvent, BaseEvent
from src.utils.agent_tools import _emit_agent_tool_call_completed
from src.utils.types import (
    ToolResult,
    get_current_provider_tool_call_id,
    get_current_tool_call_seq,
    set_current_tool_call_seq,
)


@pytest.fixture(autouse=True)
def _reset_tool_call_contextvars() -> Generator[None]:
    """Restore the tool-call ContextVars after each test.

    `set_current_tool_call_seq` mutates module-level state; without this
    fixture, tests that read `get_current_tool_call_seq()` expecting the
    default 0 become order-dependent on whichever test ran before.
    """
    prev_seq = get_current_tool_call_seq()
    prev_provider_id = get_current_provider_tool_call_id()
    try:
        yield
    finally:
        set_current_tool_call_seq(prev_seq, prev_provider_id)


@final
class _StubToolContext:
    """Duck-typed ToolContext stand-in — emitter only reads identifiers."""

    workspace_name: str
    run_id: str | None
    agent_type: str | None
    parent_category: str | None

    def __init__(
        self,
        *,
        workspace_name: str = "ws",
        run_id: str | None = "run-1",
        agent_type: str | None = "dialectic",
        parent_category: str | None = "dialectic",
    ):
        self.workspace_name = workspace_name
        self.run_id = run_id
        self.agent_type = agent_type
        self.parent_category = parent_category


class TestToolResult:
    def test_str_returns_content(self):
        result = ToolResult(content="hello", metadata={"x": 1})
        assert str(result) == "hello"

    def test_contains_delegates_to_content(self):
        """ToolResult must satisfy existing 'substring in result' assertions."""
        result = ToolResult(content="Created 3 observations", metadata={})
        assert "Created 3" in result
        assert "missing" not in result

    def test_metadata_defaults_to_empty(self):
        result = ToolResult(content="x")
        assert result.metadata == {}


class TestToolCallSeqContextVar:
    def test_seq_and_provider_id_round_trip(self):
        set_current_tool_call_seq(2, "toolu_abc")
        assert get_current_tool_call_seq() == 2
        assert get_current_provider_tool_call_id() == "toolu_abc"

    def test_provider_id_can_be_none(self):
        set_current_tool_call_seq(0, None)
        assert get_current_tool_call_seq() == 0
        assert get_current_provider_tool_call_id() is None


class TestEmitAgentToolCallCompleted:
    def test_emits_event_with_full_context(self):
        emitted: list[BaseEvent] = []
        ctx = _StubToolContext(
            run_id="run-7", agent_type="deduction", parent_category="dream"
        )

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="search_memory",
                duration_ms=42.5,
                result_str="Found 5 observations",
                metadata={
                    "top_k": 20,
                    "used_embedding": True,
                    "embedding_query_count": 1,
                    "query_tokens": 7,
                    "results_count": 5,
                },
                is_error=False,
                iteration=3,
                tool_call_seq=1,
                provider_tool_call_id="toolu_xyz",
            )

        assert len(emitted) == 1
        ev = emitted[0]
        assert isinstance(ev, AgentToolCallCompletedEvent)
        assert ev.run_id == "run-7"
        assert ev.parent_category == "dream"
        assert ev.agent_type == "deduction"
        assert ev.workspace_name == "ws"
        assert ev.iteration == 3
        assert ev.tool_call_seq == 1
        assert ev.provider_tool_call_id == "toolu_xyz"
        assert ev.tool_name == "search_memory"
        assert ev.duration_ms == 42.5
        assert ev.is_error is False
        assert ev.result_chars == len("Found 5 observations")
        # Search-specific fields surface from metadata.
        assert ev.top_k == 20
        assert ev.used_embedding is True
        assert ev.embedding_query_count == 1
        assert ev.query_tokens == 7
        assert ev.results_count == 5

    def test_resource_id_disambiguates_same_tool_in_iteration(self):
        """Resource id = {run_id}:{iteration}:{tool_call_seq}. Two calls to
        the same tool in one iteration must produce DIFFERENT ids — otherwise
        deterministic id generation would collide and dedupe would drop one
        of the events."""
        ev_a = AgentToolCallCompletedEvent(
            run_id="run-1",
            iteration=2,
            tool_call_seq=0,
            parent_category="dialectic",
            agent_type="dialectic",
            workspace_name="ws",
            tool_name="search_memory",
            duration_ms=1.0,
            result_chars=10,
            result_tokens_estimate=3,
        )
        ev_b = AgentToolCallCompletedEvent(
            run_id="run-1",
            iteration=2,
            tool_call_seq=1,
            parent_category="dialectic",
            agent_type="dialectic",
            workspace_name="ws",
            tool_name="search_memory",
            duration_ms=1.0,
            result_chars=10,
            result_tokens_estimate=3,
        )
        assert ev_a.get_resource_id() != ev_b.get_resource_id()
        # Same timestamp + different resource_id → different deterministic ids.
        ev_b.timestamp = ev_a.timestamp
        assert ev_a.generate_id() != ev_b.generate_id()

    def test_skips_when_run_id_missing(self):
        emitted: list[BaseEvent] = []
        ctx = _StubToolContext(run_id=None)

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="search_memory",
                duration_ms=0.0,
                result_str="",
                metadata={},
                is_error=False,
                iteration=1,
                tool_call_seq=0,
                provider_tool_call_id=None,
            )
        assert emitted == []

    def test_skips_when_agent_metadata_incomplete(self):
        emitted: list[BaseEvent] = []
        ctx = _StubToolContext(agent_type=None)

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="search_memory",
                duration_ms=0.0,
                result_str="",
                metadata={},
                is_error=False,
                iteration=1,
                tool_call_seq=0,
                provider_tool_call_id=None,
            )
        assert emitted == []

    def test_swallows_emit_failures(self):
        """Telemetry must never bleed into the tool path."""

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("emitter wedged")

        ctx = _StubToolContext()
        with patch("src.telemetry.events.emit", side_effect=explode):
            # Must not raise.
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="search_memory",
                duration_ms=0.0,
                result_str="",
                metadata={},
                is_error=False,
                iteration=1,
                tool_call_seq=0,
                provider_tool_call_id=None,
            )

    def test_truncation_metadata_round_trips(self):
        emitted: list[BaseEvent] = []
        ctx = _StubToolContext()

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="get_recent_history",
                duration_ms=1.0,
                result_str="abc",
                metadata={
                    "was_truncated": True,
                    "result_chars_before_truncation": 9000,
                },
                is_error=False,
                iteration=1,
                tool_call_seq=0,
                provider_tool_call_id=None,
            )

        ev = emitted[0]
        assert isinstance(ev, AgentToolCallCompletedEvent)
        assert ev.was_truncated is True
        assert ev.result_chars_before_truncation == 9000
        # Truncation delta = before - after = 9000 - 3 = 8997. Calibration can
        # compute that downstream; we just verify both fields land.
        assert ev.result_chars == 3


def test_volume_class_is_high_volume():
    """event is high-volume — sampled alongside llm.call.completed."""
    assert AgentToolCallCompletedEvent.volume_class() == "high_volume"


class TestMaybeTruncatedResult:
    """`_maybe_truncated_result` should wrap in ToolResult only when the
    helper actually clamps the output, and the wrapping must surface
    the fields the AgentToolCallCompletedEvent reads from metadata.

    These tests guard against the regression where the truncation signal
    used to be discarded by `_truncate_tool_output` returning bare str —
    leaving `was_truncated` / `result_chars_before_truncation` as dead
    fields on the event.
    """

    def test_under_cap_returns_bare_string(self):
        """No-cap path keeps the bare-str contract — no metadata wrapping."""
        from src.utils.agent_tools import _maybe_truncated_result

        out = _maybe_truncated_result("hello")
        assert out == "hello"
        assert not isinstance(out, ToolResult)

    def test_over_cap_returns_tool_result_with_metadata(self):
        """Truncated path wraps in ToolResult carrying the original size."""
        from src.config import settings
        from src.utils.agent_tools import _maybe_truncated_result

        original = "x" * 5000
        with patch.object(settings.LLM, "MAX_TOOL_OUTPUT_CHARS", 100):
            out = _maybe_truncated_result(original)
        assert isinstance(out, ToolResult)
        assert out.metadata["was_truncated"] is True
        assert out.metadata["result_chars_before_truncation"] == 5000
        # Content carries the truncation marker so the LLM knows it was clamped.
        assert "OUTPUT TRUNCATED" in out.content

    def test_end_to_end_truncation_event(self):
        """Wired path: truncated handler output reaches the event with
        was_truncated=True. This is the regression check for the
        previously-dead `was_truncated` / `result_chars_before_truncation`
        fields on AgentToolCallCompletedEvent.
        """
        from src.config import settings
        from src.utils.agent_tools import _maybe_truncated_result

        emitted: list[BaseEvent] = []
        ctx = _StubToolContext()

        original = "y" * 10_000
        with patch.object(settings.LLM, "MAX_TOOL_OUTPUT_CHARS", 200):
            wrapped = _maybe_truncated_result(original)
        assert isinstance(wrapped, ToolResult)

        with patch(
            "src.telemetry.events.emit",
            side_effect=lambda event: emitted.append(event),
        ):
            _emit_agent_tool_call_completed(
                ctx=ctx,
                tool_name="get_recent_history",
                duration_ms=1.0,
                result_str=wrapped.content,
                metadata=wrapped.metadata,
                is_error=False,
                iteration=1,
                tool_call_seq=0,
                provider_tool_call_id=None,
            )

        ev = emitted[0]
        assert isinstance(ev, AgentToolCallCompletedEvent)
        assert ev.was_truncated is True
        assert ev.result_chars_before_truncation == 10_000
