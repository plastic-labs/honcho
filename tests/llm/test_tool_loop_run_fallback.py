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
from src.exceptions import UpstreamLLMError
from src.llm import api as api_module
from src.llm import tool_loop as tool_loop_module
from src.llm.types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)

TOOLS = [{"name": "noop", "description": "no-op", "input_schema": {"type": "object"}}]
QUERY = [{"role": "user", "content": "hi"}]


def _config(with_fallback: bool = True) -> ModelConfig:
    fallback = (
        ResolvedFallbackConfig(model="gpt-5", transport="openai", api_key="test-key")
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


def _answer(content: str) -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content=content, output_tokens=1, finish_reasons=["stop"]
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
        }
        self.calls.append(call)
        return call

    def by_provider(self, provider: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["provider"] == provider]


def _no_wait(**_kw: Any) -> wait_fixed:
    return wait_fixed(0)


def _no_backoff() -> Any:
    return patch.object(tool_loop_module, "wait_exponential", _no_wait)


async def _call(
    recorder_fn: Any, *, config: ModelConfig, retry_attempts: int = 1
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
            tool_executor=lambda _name, _input: "tool output",
            max_tool_iterations=5,
            messages=QUERY,
            enable_retry=True,
            retry_attempts=retry_attempts,
        )


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
    # Iteration 1 succeeded, iteration 2 burned the whole retry budget.
    assert [c["attempt"] for c in primary] == [1, 1, 2]
    assert all(not c["is_fallback"] for c in primary)
    # The fallback owns the run from the first message: nothing the primary
    # wrote is in its history, and it never sees an attempt it did not make.
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

    # Tool-call ids are tagged with the provider that issued them, so a
    # history that names the other provider's tag was spliced together.
    for call in recorder.calls:
        other = "openai" if call["provider"] == "anthropic" else "anthropic"
        assert f"{other}-call" not in json.dumps(call["messages"]), call
    assert len(recorder.by_provider("openai")) == 2


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
            tool_executor=lambda _name, _input: "tool output",
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
    # The fallback re-ran the tool phase, not just the stream.
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
