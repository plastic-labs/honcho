"""Real executor/capture/export paths, with only the provider replaced."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from tenacity import wait_none

from src.config import (
    ConfiguredModelSettings,
    FallbackModelSettings,
    ModelConfig,
    settings,
)
from src.llm import api, capture, executor, runtime, tool_loop
from src.llm.backend import CompletionResult, ProviderBackend, StreamChunk
from src.llm.types import HonchoLLMCallStreamChunk, LLMTelemetryContext
from src.telemetry import events, trace_exporter
from src.telemetry.events import BaseEvent, LLMCallCompletedEvent
from src.telemetry.events.trace import LLMCallTracedEvent, TraceContentEvent

LOOKUP_TOOL: dict[str, Any] = {
    "name": "lookup",
    "description": "Lookup",
    "parameters": {"type": "object", "properties": {}},
}


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[BaseEvent]:
    recorded: list[BaseEvent] = []
    capture.clear_exporters()
    capture.register_exporter(trace_exporter.TraceExporter())
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PAYLOADS_ENABLED", True)
    monkeypatch.setattr(settings.TELEMETRY, "TRACE_PURPOSES", [])
    monkeypatch.setattr(trace_exporter, "emit_trace", recorded.append)
    monkeypatch.setattr(events, "emit", recorded.append)
    return recorded


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    backend = AsyncMock(spec=ProviderBackend)
    backend.complete.return_value = CompletionResult(content="answer")
    monkeypatch.setattr(executor, "backend_for_provider", Mock(return_value=backend))
    monkeypatch.setattr(runtime, "client_for_model_config", Mock(return_value=Mock()))
    monkeypatch.setattr(api, "wait_exponential", Mock(return_value=wait_none()))
    monkeypatch.setattr(tool_loop, "wait_exponential", Mock(return_value=wait_none()))
    return backend


@pytest.mark.parametrize("system_count", [0, 1, 2])
async def test_export_preserves_identity_and_content(
    system_count: int, backend: AsyncMock, recorded: list[BaseEvent]
) -> None:
    context = LLMTelemetryContext(
        workspace_name="ws",
        session_id="session",
        observer="alice",
        observers=["alice", "bob"],
        observed="carol",
        peer_name="carol",
        agent_type="dialectic",
        track_name="Dialectic",
        run_id="run",
        trace_id="trace",
        span_id="span",
        parent_span_id="parent",
        parent_event_id="upstream-event",
        call_purpose="dialectic.answer",
        parent_category="dialectic",
        source_message_ids=["message"],
        queue_item_ids=[7],
    )
    reasoning = [{"type": "reasoning.text", "text": "provider reasoning"}]
    backend.complete.return_value = CompletionResult(
        content="answer",
        input_tokens=20,
        output_tokens=5,
        reasoning_details=reasoning,
    )
    await api.honcho_llm_call(
        model_config=ModelConfig(
            transport="openai", model="synthetic", max_output_tokens=64
        ),
        prompt="question",
        max_tokens=128,
        telemetry=context,
        messages=[
            {"role": "system", "content": f"instruction {i}"}
            for i in range(system_count)
        ]
        + [{"role": "user", "content": "question"}],
    )
    (trace,) = [e for e in recorded if isinstance(e, LLMCallTracedEvent)]
    (completed,) = [e for e in recorded if isinstance(e, LLMCallCompletedEvent)]
    content = {e.content_hash: e for e in recorded if isinstance(e, TraceContentEvent)}
    for field in (
        "workspace_name",
        "session_id",
        "observers",
        "observed",
        "peer_name",
        "agent_type",
        "track_name",
        "run_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "parent_event_id",
        "source_message_ids",
        "queue_item_ids",
    ):
        assert getattr(trace, field) == getattr(context, field)
    assert trace.schema_version() == 2
    assert len(trace.system_prompt_refs) == system_count
    assert [content[ref].content for ref in trace.system_prompt_refs] == [
        f"instruction {i}" for i in range(system_count)
    ]
    # One ref for the whole system prompt: the message itself when there is one,
    # the concatenation of all of them when there are several.
    if system_count == 0:
        assert trace.system_prompt_ref is None
    elif system_count == 1:
        assert trace.system_prompt_ref == trace.system_prompt_refs[0]
    else:
        assert trace.system_prompt_ref is not None
        assert trace.system_prompt_ref not in trace.system_prompt_refs
        assert content[trace.system_prompt_ref].content == "\n\n".join(
            f"instruction {i}" for i in range(system_count)
        )
    assert trace.output_reasoning_ref is not None
    assert content[trace.output_reasoning_ref].content == capture.canonical_json(
        reasoning
    )
    assert trace.duration_ms == completed.duration_ms
    assert backend.complete.await_args is not None
    assert (
        trace.effective_max_output_tokens
        == backend.complete.await_args.kwargs["max_tokens"]
        == 128
    )
    assert trace.outcome == "success" and trace.error_class is None
    assert trace.was_stream is False
    assert trace.provider_input_tokens == 20 and trace.provider_output_tokens == 5
    assert trace.raw_response_ref is None
    context.observers.append("later")
    context.queue_item_ids.append(8)
    assert trace.observers == ["alice", "bob"] and trace.queue_item_ids == [7]


@pytest.mark.parametrize("ending", ["success", "error", "cancelled", "closed"])
@pytest.mark.parametrize("with_tools", [False, True])
async def test_stream_records_partial_output_and_outcome(
    ending: str, with_tools: bool, backend: AsyncMock, recorded: list[BaseEvent]
) -> None:
    events_at_cleanup: list[list[BaseEvent]] = []

    async def chunks() -> AsyncIterator[StreamChunk]:
        try:
            yield StreamChunk(content="partial", output_tokens=2)
            if ending == "error":
                raise RuntimeError("synthetic disconnect")
            if ending == "cancelled":
                raise asyncio.CancelledError()
            yield StreamChunk(
                content=" answer", is_done=True, finish_reason="length", output_tokens=4
            )
        finally:
            events_at_cleanup.append(
                [
                    event
                    for event in recorded
                    if isinstance(event, LLMCallCompletedEvent | LLMCallTracedEvent)
                    and event.was_stream
                ]
            )

    def setup(**_: Any) -> AsyncIterator[StreamChunk]:
        return chunks()

    backend.stream.side_effect = setup
    stream = await api.honcho_llm_call(
        model_config=ModelConfig(transport="openai", model="synthetic"),
        prompt="question",
        max_tokens=128,
        stream=True,
        stream_final_only=with_tools,
        tools=[LOOKUP_TOOL] if with_tools else None,
        tool_executor=Mock() if with_tools else None,
        telemetry=LLMTelemetryContext(
            trace_id="trace", span_id="span", session_id="session"
        ),
    )
    if ending == "closed":
        iterator = stream.__aiter__()
        assert isinstance(iterator, AsyncGenerator)
        closable = cast(AsyncGenerator[HonchoLLMCallStreamChunk], iterator)
        assert await anext(closable)
        await closable.aclose()
    elif ending in {"error", "cancelled"}:
        with pytest.raises(
            RuntimeError if ending == "error" else asyncio.CancelledError
        ):
            async for _ in stream:
                pass
    else:
        assert "".join([chunk.content async for chunk in stream]) == "partial answer"

    (trace,) = [
        e for e in recorded if isinstance(e, LLMCallTracedEvent) and e.was_stream
    ]
    assert events_at_cleanup == [[]]
    (completed,) = [
        e for e in recorded if isinstance(e, LLMCallCompletedEvent) and e.was_stream
    ]
    content = {
        e.content_hash: e.content for e in recorded if isinstance(e, TraceContentEvent)
    }
    assert trace.output_content_ref is not None
    assert content[trace.output_content_ref] == (
        "partial answer" if ending == "success" else "partial"
    )
    assert (
        trace.outcome
        == completed.outcome
        == ("cancelled" if ending == "closed" else ending)
    )
    assert (
        trace.error_class
        == {
            "success": None,
            "error": "RuntimeError",
            "cancelled": "CancelledError",
            "closed": "GeneratorExit",
        }[ending]
    )
    assert trace.finish_reason == ("length" if ending == "success" else trace.outcome)
    assert completed.finish_reason == ("length" if ending == "success" else None)
    assert trace.provider_output_tokens == (4 if ending == "success" else 2)
    assert trace.was_stream is True and trace.session_id == "session"
    assert trace.duration_ms == completed.duration_ms


@pytest.mark.parametrize("with_tools", [False, True])
async def test_stream_setup_retries_record_each_actual_attempt_once(
    with_tools: bool, backend: AsyncMock, recorded: list[BaseEvent]
) -> None:
    attempts = 0

    async def chunks() -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content="answer", is_done=True, finish_reason="stop")

    def setup(**_: Any) -> AsyncIterator[StreamChunk]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("synthetic setup failure")
        return chunks()

    backend.stream.side_effect = setup
    stream = await api.honcho_llm_call(
        model_config=ConfiguredModelSettings(
            transport="openai",
            model="primary",
            fallback=FallbackModelSettings(transport="openai", model="fallback"),
        ),
        prompt="question",
        max_tokens=64,
        stream=True,
        stream_final_only=with_tools,
        tools=[LOOKUP_TOOL] if with_tools else None,
        tool_executor=Mock() if with_tools else None,
        telemetry=LLMTelemetryContext(
            trace_id="trace", span_id="span", session_id="session"
        ),
    )
    assert "".join([chunk.content async for chunk in stream]) == "answer"
    traces = [e for e in recorded if isinstance(e, LLMCallTracedEvent)]
    streamed = [e for e in traces if e.was_stream]
    assert len(traces) == backend.complete.await_count + attempts
    assert [e.attempt for e in streamed] == [1, 2, 3]
    assert [e.outcome for e in streamed] == ["error", "error", "success"]
    assert [e.is_final_attempt for e in streamed] == [False, False, True]
    assert all(e.retry_attempts == 3 and e.session_id == "session" for e in streamed)
    # Tool-loop final streams retry the selected model; plain calls reselect fallback.
    assert [e.was_fallback for e in streamed] == (
        [False] * 3 if with_tools else [False, False, True]
    )
    assert streamed[-1].model == ("primary" if with_tools else "fallback")


def test_legacy_trace_loads_without_execution_metadata() -> None:
    legacy = LLMCallTracedEvent.model_validate({"transport": "openai", "model": "old"})
    assert legacy.was_stream is None and legacy.outcome is None
    assert legacy.duration_ms is None and legacy.retry_attempts is None
    assert legacy.observers == legacy.source_message_ids == legacy.queue_item_ids == []
    enriched = legacy.model_copy(
        update={"workspace_name": "ws", "source_message_ids": ["m"]}
    )
    assert enriched.generate_id() == legacy.generate_id()
