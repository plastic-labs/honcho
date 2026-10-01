"""A tool loop never mixes providers inside one run.

Failing over on the last retry of a single call spliced the fallback model
into the primary's half-finished tool history: the fallback then replayed
turns it never wrote, and the primary (once healthy) was handed turns from a
model whose thinking blocks it cannot verify. Now a run is one provider's
conversation: when a call exhausts its retries on the primary, the whole run
restarts from the caller's messages on the fallback.
"""

import json
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from tenacity import wait_fixed

from src.config import ModelConfig, ResolvedFallbackConfig
from src.exceptions import UpstreamLLMError, ValidationException
from src.llm import api as api_module
from src.llm import tool_loop as tool_loop_module
from src.llm.runtime import CapturedAgentSpan
from src.llm.types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    LLMTelemetryContext,
    StreamingResponseWithMetadata,
)

TOOLS = [{"name": "noop", "description": "no-op", "input_schema": {"type": "object"}}]
QUERY = [{"role": "user", "content": "hi"}]
TELEMETRY = LLMTelemetryContext(
    workspace_name="ws",
    call_purpose="dialectic.answer",
    parent_category="dialectic",
    agent_type="dialectic",
    run_id="run-1",
    trace_id="run-1",
    span_id="run-1",
    track_name="Dialectic Agent",
    tags=["existing"],
)


def _config(
    with_fallback: bool = True, fallback_thinking_budget: int | None = None
) -> ModelConfig:
    fallback = (
        ResolvedFallbackConfig(
            model="gpt-5",
            transport="openai",
            api_key="test-key",
            thinking_budget_tokens=fallback_thinking_budget,
        )
        if with_fallback
        else None
    )
    return ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
        api_key="test-key",
        fallback=fallback,
    )


def _tool_call_response(provider: str = "anthropic") -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content="",
        output_tokens=1,
        finish_reasons=["tool_use"],
        tool_calls_made=[{"name": "noop", "input": {}, "id": f"{provider}-call"}],
    )


def _answer(
    content: str, *, input_tokens: int = 0, output_tokens: int = 1
) -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reasons=["stop"],
    )


class Recorder:
    """Stand-in for honcho_llm_call_inner that scripts each provider's behavior."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        plan = kwargs["plan"]
        call = {
            "provider": plan.provider,
            "is_fallback": plan.is_fallback,
            "attempt": plan.attempt,
            "messages": [dict(m) for m in kwargs["messages"]],
            "stream": kwargs["stream"],
            "telemetry": kwargs["telemetry"],
            "tools": kwargs["tools"],
            "thinking_budget_tokens": plan.thinking_budget_tokens,
        }
        self.calls.append(call)
        return call

    def by_provider(self, provider: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["provider"] == provider]


def _no_wait(**_kw: Any) -> wait_fixed:
    return wait_fixed(0)


async def _run_tool(_name: str, _input: dict[str, Any]) -> str:
    return "tool output"


def _no_backoff() -> Any:
    return patch.object(tool_loop_module, "wait_exponential", _no_wait)


async def _call(
    recorder_fn: Any,
    *,
    config: ModelConfig,
    retry_attempts: int = 1,
    max_tool_iterations: int = 5,
    thinking_budget_tokens: int | None = None,
    restart_blocked_by: set[str] | None = None,
    run: CapturedAgentSpan | None = None,
) -> Any:
    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=recorder_fn),
        _no_backoff(),
    ):
        return await api_module.honcho_llm_call(
            model_config=config,
            prompt="",
            max_tokens=64,
            tools=TOOLS,
            tool_choice="auto",
            tool_executor=_run_tool,
            max_tool_iterations=max_tool_iterations,
            messages=QUERY,
            enable_retry=True,
            retry_attempts=retry_attempts,
            thinking_budget_tokens=thinking_budget_tokens,
            telemetry=TELEMETRY,
            run=run,
            restart_blocked_by=restart_blocked_by,
        )


def _primary_dies_after_one_tool_round(recorder: "Recorder") -> Any:
    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic":
            if len(call["messages"]) == 1:
                return _tool_call_response()
            raise UpstreamLLMError("Model provider returned HTTP 529")
        return _answer("from-fallback")

    return fake


@pytest.mark.asyncio
async def test_run_restarts_on_fallback_from_the_original_messages() -> None:
    """The primary makes a tool call, then dies; the fallback starts over."""
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic":
            if len(call["messages"]) == 1:
                return _tool_call_response()
            raise UpstreamLLMError("Model provider returned HTTP 529")
        return _answer("from-fallback")

    result = await _call(fake, config=_config(), retry_attempts=2)

    assert isinstance(result, HonchoLLMCallResponse)
    assert cast(str, result.content) == "from-fallback"

    primary = recorder.by_provider("anthropic")
    fallback = recorder.by_provider("openai")
    assert [c["attempt"] for c in primary] == [1, 1, 2]
    assert all(not c["is_fallback"] for c in primary)
    assert fallback[0]["messages"] == QUERY
    assert [c["attempt"] for c in fallback] == [1]
    assert all(c["is_fallback"] for c in fallback)


@pytest.mark.asyncio
async def test_no_provider_sees_another_providers_turns() -> None:
    """Every assistant turn in a call's history was written by that call's provider."""
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic":
            if len(call["messages"]) == 1:
                return _tool_call_response("anthropic")
            raise UpstreamLLMError("Model provider returned HTTP 529")
        if len(call["messages"]) == 1:
            return _tool_call_response("openai")
        return _answer("done")

    await _call(fake, config=_config())

    for call in recorder.calls:
        other = "openai" if call["provider"] == "anthropic" else "anthropic"
        assert f"{other}-call" not in json.dumps(call["messages"]), call
    assert len(recorder.by_provider("openai")) == 2


@pytest.mark.asyncio
async def test_fallback_run_is_tagged_for_langfuse() -> None:
    """The restarted run's telemetry names the tag and what it fell back from."""
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic":
            raise UpstreamLLMError("Model provider returned HTTP 529")
        return _answer("from-fallback")

    await _call(fake, config=_config())

    (primary,) = recorder.by_provider("anthropic")
    (fallback,) = recorder.by_provider("openai")
    assert primary["telemetry"].tags == ["existing"]
    assert "fallback_from" not in primary["telemetry"].metadata
    assert fallback["telemetry"].tags == ["existing", "fallback_restart"]
    assert fallback["telemetry"].metadata == {
        "fallback_from": "anthropic/claude-haiku-4-5",
        "fallback_reason": "UpstreamLLMError",
    }
    assert fallback["telemetry"].run_id == "run-1"


@pytest.mark.asyncio
async def test_primary_success_never_touches_the_fallback() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if len(call["messages"]) == 1:
            return _tool_call_response()
        return _answer("from-primary")

    result = await _call(fake, config=_config())

    assert isinstance(result, HonchoLLMCallResponse)
    assert cast(str, result.content) == "from-primary"
    assert recorder.by_provider("openai") == []
    assert all(not c["is_fallback"] for c in recorder.calls)


@pytest.mark.asyncio
async def test_fallback_run_failing_surfaces_the_provider_error() -> None:
    """Two runs, then the caller gets the fallback's own error, not a RetryError."""
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        raise UpstreamLLMError(f"{call['provider']} is down")

    with pytest.raises(UpstreamLLMError, match="openai is down") as caught:
        await _call(fake, config=_config())

    assert caught.value.status_code == 503
    assert [c["provider"] for c in recorder.calls] == ["anthropic", "openai"]


@pytest.mark.asyncio
async def test_without_a_fallback_the_primary_error_surfaces() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        recorder.record(kwargs)
        raise UpstreamLLMError("anthropic is down")

    with pytest.raises(UpstreamLLMError, match="anthropic is down"):
        await _call(fake, config=_config(with_fallback=False))

    assert [c["provider"] for c in recorder.calls] == ["anthropic"]


async def _chunks(*texts: str) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    for text in texts:
        yield HonchoLLMCallStreamChunk(content=text)


async def _stream_call(
    fake: Any, *, config: ModelConfig
) -> StreamingResponseWithMetadata:
    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=fake),
        _no_backoff(),
    ):
        result = await api_module.honcho_llm_call(
            model_config=config,
            prompt="",
            max_tokens=64,
            stream=True,
            stream_final_only=True,
            tools=TOOLS,
            tool_choice="auto",
            tool_executor=_run_tool,
            max_tool_iterations=5,
            messages=QUERY,
            enable_retry=True,
            retry_attempts=1,
        )
    assert isinstance(result, StreamingResponseWithMetadata)
    return result


@pytest.mark.asyncio
async def test_final_stream_that_cannot_open_restarts_the_run_on_fallback() -> None:
    """Stream setup runs inside the tool loop, so it is still restartable."""
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> Any:
        call = recorder.record(kwargs)
        if not call["stream"]:
            return _answer("ok")
        if call["provider"] == "anthropic":
            raise UpstreamLLMError("anthropic cannot stream")
        return _chunks("from ", "fallback")

    result = await _stream_call(fake, config=_config())
    streamed = "".join([chunk.content async for chunk in result])

    assert streamed == "from fallback"
    assert [c["stream"] for c in recorder.by_provider("openai")] == [False, True]


@pytest.mark.asyncio
async def test_failure_after_the_first_chunk_is_not_restarted() -> None:
    """Content already reached the client; the error propagates to the drainer."""
    recorder = Recorder()

    async def _breaks_midway() -> AsyncIterator[HonchoLLMCallStreamChunk]:
        yield HonchoLLMCallStreamChunk(content="partial")
        raise UpstreamLLMError("connection reset")

    async def fake(*_args: Any, **kwargs: Any) -> Any:
        call = recorder.record(kwargs)
        if not call["stream"]:
            return _answer("ok")
        return _breaks_midway()

    result = await _stream_call(fake, config=_config())
    received: list[str] = []
    with pytest.raises(UpstreamLLMError, match="connection reset"):
        async for chunk in result:
            received.append(chunk.content)

    assert received == ["partial"]
    assert recorder.by_provider("openai") == []


@pytest.mark.asyncio
async def test_fallback_calls_get_the_full_retry_budget() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic" or call["attempt"] < 3:
            raise UpstreamLLMError(f"{call['provider']} is busy")
        return _answer("third time lucky")

    result = await _call(fake, config=_config(), retry_attempts=3)

    assert cast(str, result.content) == "third time lucky"
    assert [c["attempt"] for c in recorder.by_provider("anthropic")] == [1, 2, 3]
    assert [c["attempt"] for c in recorder.by_provider("openai")] == [1, 2, 3]


@pytest.mark.asyncio
async def test_synthesis_after_max_iterations_restarts_on_fallback() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["tools"]:
            return _tool_call_response(call["provider"])
        if call["provider"] == "anthropic":
            raise UpstreamLLMError("anthropic cannot synthesize")
        return _answer("synthesized")

    result = await _call(fake, config=_config(), max_tool_iterations=1)

    assert cast(str, result.content) == "synthesized"
    assert [bool(c["tools"]) for c in recorder.by_provider("anthropic")] == [
        True,
        False,
    ]
    assert [bool(c["tools"]) for c in recorder.by_provider("openai")] == [True, False]
    assert result.iterations == 2


@pytest.mark.asyncio
async def test_caller_owned_run_span_is_left_open_and_marked() -> None:
    recorder = Recorder()
    run = CapturedAgentSpan(telemetry=TELEMETRY, kind="run")

    await _call(_primary_dies_after_one_tool_round(recorder), config=_config(), run=run)

    assert run._ended is False  # pyright: ignore[reportPrivateUsage]
    assert run.telemetry.tags == ["existing", "fallback_restart"]
    assert run.telemetry.metadata["fallback_from"] == "anthropic/claude-haiku-4-5"
    assert len(recorder.by_provider("openai")) == 1


@pytest.mark.asyncio
async def test_fallback_run_uses_its_own_thinking_budget() -> None:
    recorder = Recorder()

    await _call(
        _primary_dies_after_one_tool_round(recorder),
        config=_config(fallback_thinking_budget=512),
        thinking_budget_tokens=2048,
    )

    assert {c["thinking_budget_tokens"] for c in recorder.by_provider("anthropic")} == {
        2048
    }
    assert {c["thinking_budget_tokens"] for c in recorder.by_provider("openai")} == {
        512
    }


@pytest.mark.asyncio
async def test_validation_errors_are_not_restarted() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        recorder.record(kwargs)
        raise ValidationException("thinking cannot be forced with tool_choice")

    with pytest.raises(ValidationException):
        await _call(fake, config=_config())

    assert recorder.by_provider("openai") == []


@pytest.mark.asyncio
async def test_primary_spend_is_folded_into_the_fallback_result() -> None:
    recorder = Recorder()

    async def fake(*_args: Any, **kwargs: Any) -> HonchoLLMCallResponse[str]:
        call = recorder.record(kwargs)
        if call["provider"] == "anthropic":
            if len(call["messages"]) == 1:
                response = _tool_call_response()
                response.input_tokens = 10
                response.output_tokens = 5
                return response
            raise UpstreamLLMError("Model provider returned HTTP 529")
        return _answer("from-fallback", input_tokens=7, output_tokens=3)

    result = await _call(fake, config=_config())

    assert result.input_tokens == 17
    assert result.output_tokens == 8
    assert [tc["tool_name"] for tc in result.tool_calls_made] == ["noop"]


@pytest.mark.asyncio
async def test_a_run_that_already_wrote_is_not_restarted() -> None:
    recorder = Recorder()

    with pytest.raises(UpstreamLLMError, match="529"):
        await _call(
            _primary_dies_after_one_tool_round(recorder),
            config=_config(),
            restart_blocked_by={"noop"},
        )

    assert recorder.by_provider("openai") == []


@pytest.mark.asyncio
async def test_blocked_tools_that_never_ran_do_not_prevent_a_restart() -> None:
    recorder = Recorder()

    result = await _call(
        _primary_dies_after_one_tool_round(recorder),
        config=_config(),
        restart_blocked_by={"delete_observations"},
    )

    assert cast(str, result.content) == "from-fallback"
