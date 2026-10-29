"""Tests for the single-capture content layer (src/llm/capture.py)."""

from __future__ import annotations

import pytest

from src.llm import capture
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.capture import (
    CapturedLLMCall,
    build_captured_call,
    build_captured_messages,
    canonical_json,
    clip_for_trace,
    compute_content_hash,
)
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from src.llm.types import LLMTelemetryContext


class TestContentHash:
    def test_is_deterministic_and_prefixed(self):
        h1 = compute_content_hash("user", "hello", None)
        h2 = compute_content_hash("user", "hello", None)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_role_is_inside_the_hash(self):
        # Identical text under different roles must never collide — role lives
        # inside the hash, closing the role-in-hash collision bug.
        assert compute_content_hash("user", "hi", None) != compute_content_hash(
            "assistant", "hi", None
        )

    def test_tool_call_id_is_inside_the_hash(self):
        assert compute_content_hash("tool", "ok", "call_1") != compute_content_hash(
            "tool", "ok", "call_2"
        )

    def test_canonical_json_is_order_independent(self):
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


class TestClipForTrace:
    def test_leaves_small_content_untouched(self):
        content, truncated = clip_for_trace("short")
        assert content == "short"
        assert truncated is False

    def test_clips_oversized_string(self, monkeypatch: pytest.MonkeyPatch):
        from src.config import settings

        monkeypatch.setattr(settings.TELEMETRY, "TRACE_MAX_BYTES", 32)
        content, truncated = clip_for_trace("x" * 1000)
        assert truncated is True
        assert content.endswith("…[truncated]")
        assert len(content.encode("utf-8")) <= settings.TELEMETRY.TRACE_MAX_BYTES

    def test_leaves_structured_content_intact(self, monkeypatch: pytest.MonkeyPatch):
        from src.config import settings

        monkeypatch.setattr(settings.TELEMETRY, "TRACE_MAX_BYTES", 4)
        blocks = [{"type": "text", "text": "a long block of structured content"}]
        content, truncated = clip_for_trace(blocks)
        assert content == blocks
        assert truncated is False


class TestBuildCapturedMessages:
    def test_hashes_each_message(self):
        messages: list[ChatMessage] = [
            UserMessage(content="q"),
            AIMessage(text="a"),
        ]
        captured, truncated = build_captured_messages(messages, memo=None)
        assert [m.role for m in captured] == ["user", "assistant"]
        assert all(m.content_hash.startswith("sha256:") for m in captured)
        assert truncated is False

    def test_memo_makes_hashing_on(self, monkeypatch: pytest.MonkeyPatch):
        # The conversation is append-only and messages are reused, so with a
        # shared memo each message is hashed exactly once across iterations.
        calls = {"n": 0}
        real = compute_content_hash

        def counting(
            role: str,
            content: object,
            tool_call_id: str | None,
            tool_calls: list[dict[str, object]] | None = None,
        ) -> str:
            calls["n"] += 1
            return real(role, content, tool_call_id, tool_calls)

        monkeypatch.setattr(capture, "compute_content_hash", counting)

        m1 = UserMessage(content="q1")
        m2 = AIMessage(text="a1")
        m3 = UserMessage(content="q2")
        memo: dict[int, capture.CapturedMessage] = {}

        build_captured_messages([m1, m2], memo)
        assert calls["n"] == 2  # both hashed
        build_captured_messages([m1, m2, m3], memo)
        assert calls["n"] == 3  # only the newly-appended m3 hashed (not re-hashed)


class TestToolTurnsInTraces:
    """Tool calls and results surface in the unified trace shape whichever
    provider produced them."""

    def test_assistant_tool_calls_captured(self):
        msg = AIMessage(
            text="let me look",
            tool_calls=[
                ToolCall(id="call_1", name="search_memory", input={"query": "coffee"})
            ],
        )
        captured, _ = build_captured_messages([msg], memo=None)
        assert captured[0].role == "assistant"
        assert captured[0].content == "let me look"
        assert captured[0].tool_calls == [
            {"id": "call_1", "name": "search_memory", "input": {"query": "coffee"}}
        ]

    def test_tool_result_keyed_by_call_id(self):
        msg = ToolMessage(tool_call_id="call_1", name="grep_messages", content="3 hits")
        captured, _ = build_captured_messages([msg], memo=None)
        assert captured[0].role == "tool"
        assert captured[0].content == "3 hits"
        assert captured[0].tool_call_id == "call_1"

    def test_hash_distinguishes_tool_calls(self):
        # Two empty-content assistant turns with different tool calls must not
        # collide in the dedup store.
        a = AIMessage(tool_calls=[ToolCall(id="c1", name="search_memory", input={})])
        b = AIMessage(tool_calls=[ToolCall(id="c2", name="search_messages", input={})])
        (ca,), _ = build_captured_messages([a], memo=None)
        (cb,), _ = build_captured_messages([b], memo=None)
        assert ca.content_hash != cb.content_hash


class TestBuildCapturedCall:
    def test_maps_telemetry_and_result(self):
        telemetry = LLMTelemetryContext(
            workspace_name="ws",
            call_purpose="dialectic.answer",
            parent_category="dialectic",
            run_id="r1",
            trace_id="r1",
            span_id="r1",
            session_id="sess_abc",
            iteration=2,
            step_seq=2,
        )
        result = CompletionResult(
            content="answer",
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            tool_calls=[ToolCallResult(id="t1", name="search", input={"q": "x"})],
        )
        call = build_captured_call(
            telemetry=telemetry,
            transport="anthropic",
            provider_label=None,
            model="claude-x",
            messages=[UserMessage(content="q")],
            tools=None,
            tool_choice=None,
            result=result,
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="stop",
        )
        assert isinstance(call, CapturedLLMCall)
        assert call.trace_id == "r1" and call.span_id == "r1"
        assert call.iteration == 2 and call.step_seq == 2
        assert call.output_content == "answer"
        assert call.output_tool_calls == [
            {"id": "t1", "name": "search", "input": {"q": "x"}}
        ]
        assert call.input_tokens == 10 and call.output_tokens == 5
        assert call.session_id == "sess_abc"
        assert len(call.input_messages) == 1
        assert call.input_messages[0].content_hash.startswith("sha256:")

    def test_session_id_defaults_none_without_telemetry(self):
        # Sessionless calls (and the no-telemetry path) carry session_id=None so
        # the Langfuse projection emits no session grouping for them.
        telemetry = LLMTelemetryContext(run_id="r1", trace_id="r1", span_id="r1")
        call = build_captured_call(
            telemetry=telemetry,
            transport="anthropic",
            provider_label=None,
            model="claude-x",
            messages=[UserMessage(content="q")],
            tools=None,
            tool_choice=None,
            result=CompletionResult(content="a", finish_reason="stop"),
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="stop",
        )
        assert call.session_id is None

    def test_self_parent_is_normalized_to_none(self):
        # The tool loop sets parent_span_id == span_id on the run span (it
        # doubles as the Langfuse "inside a run" signal). A span that is its own
        # parent is a root, so the EXPORTED parent_span_id must be None — else
        # span-tree consumers file the root as a child of itself.
        telemetry = LLMTelemetryContext(
            run_id="r1", trace_id="r1", span_id="r1", parent_span_id="r1"
        )
        call = build_captured_call(
            telemetry=telemetry,
            transport="anthropic",
            provider_label=None,
            model="claude-x",
            messages=[UserMessage(content="q")],
            tools=None,
            tool_choice=None,
            result=CompletionResult(content="a", finish_reason="stop"),
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="stop",
        )
        assert call.span_id == "r1"
        assert call.parent_span_id is None
        # A genuine distinct parent is preserved.
        telemetry.parent_span_id = "parent-span"
        assert telemetry.exported_parent_span_id() == "parent-span"

    def test_error_path_collapses_output(self):
        call = build_captured_call(
            telemetry=None,
            transport="anthropic",
            provider_label=None,
            model="claude-x",
            messages=[UserMessage(content="q")],
            tools=None,
            tool_choice=None,
            result=None,
            attempt=2,
            was_fallback=True,
            was_stream=False,
            finish_reason="error",
        )
        assert call.output_content is None
        assert call.output_tool_calls == []
        assert call.finish_reason == "error"
        assert call.attempt == 2 and call.was_fallback is True


class TestExporterRegistry:
    def test_register_dispatch_and_clear(self):
        capture.clear_exporters()
        assert capture.has_exporters() is False
        seen: list[CapturedLLMCall] = []

        class _Spy:
            def export(self, call: CapturedLLMCall) -> None:
                seen.append(call)

        capture.register_exporter(_Spy())
        assert capture.has_exporters() is True

        call = build_captured_call(
            telemetry=None,
            transport="anthropic",
            provider_label=None,
            model="m",
            messages=[],
            tools=None,
            tool_choice=None,
            result=None,
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="stop",
        )
        capture.dispatch_captured_call(call)
        assert seen == [call]
        capture.clear_exporters()
        assert capture.has_exporters() is False

    def test_dispatch_swallows_exporter_errors(self):
        capture.clear_exporters()

        class _Boom:
            def export(self, call: CapturedLLMCall) -> None:
                raise RuntimeError(f"nope: {call.model}")

        capture.register_exporter(_Boom())
        call = build_captured_call(
            telemetry=None,
            transport="anthropic",
            provider_label=None,
            model="m",
            messages=[],
            tools=None,
            tool_choice=None,
            result=None,
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="stop",
        )
        # Must not raise — telemetry never breaks the LLM path.
        capture.dispatch_captured_call(call)
        capture.clear_exporters()


class TestThoughtSignatureSerialization:
    """Gemini `thought_signature` is bytes; it must not break trace serialization."""

    def test_bytes_signature_base64_encoded_and_serializes(self):
        import json

        from src.telemetry.events.trace import LLMCallTracedEvent

        result = CompletionResult(
            content=None,
            finish_reason="STOP",
            tool_calls=[
                ToolCallResult(
                    id="call_1",
                    name="grep_messages",
                    input={"text": "coffee"},
                    thought_signature=b"\x0a\x1f\x88\xff\x00sig",
                )
            ],
        )
        call = build_captured_call(
            telemetry=LLMTelemetryContext(trace_id="t1", span_id="s1"),
            transport="gemini",
            provider_label=None,
            model="gemini-2.5-flash",
            messages=[UserMessage(content="q")],
            tools=None,
            tool_choice=None,
            result=result,
            attempt=1,
            was_fallback=False,
            was_stream=False,
            finish_reason="STOP",
        )
        sig = call.output_tool_calls[0]["thought_signature"]
        assert isinstance(sig, str)  # base64, not raw bytes

        # The traced event must serialize to JSON without raising (the emit path
        # calls model_dump(mode="json"), which threw UnicodeDecodeError on bytes).
        event = LLMCallTracedEvent(
            model="gemini-2.5-flash",
            transport="gemini",
            output_tool_calls=call.output_tool_calls,
        )
        json.dumps(event.model_dump(mode="json"))
