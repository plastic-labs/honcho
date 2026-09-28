# pyright: reportPrivateUsage=false, reportUnusedParameter=false
"""The DialecticAgent owns its Langfuse run: opened before prefetch, ended with the answer."""

import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.dialectic.core import DialecticAgent
from src.llm import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)


class FakeRun:
    def __init__(self) -> None:
        self.ends: list[dict[str, Any]] = []

    def end(self, *, output: Any = None, is_error: bool = False) -> None:
        self.ends.append({"output": output, "is_error": is_error})


def _make_agent() -> DialecticAgent:
    return DialecticAgent(
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        reasoning_level="low",
    )


@pytest.mark.asyncio
async def test_prepare_query_opens_the_run_before_prefetch() -> None:
    agent = _make_agent()
    order: list[str] = []
    run = FakeRun()

    def start_run(kind: str, telemetry: Any, *, input: Any = None) -> FakeRun:  # noqa: A002
        assert kind == "run"
        order.append("run")
        assert input == "what does alice do?"
        return run

    async def prefetch(query: str) -> None:
        order.append("prefetch")

    with (
        patch("src.dialectic.core.start_captured_span", new=start_run),
        patch.object(agent, "_initialize_session_history", new=AsyncMock()),
        patch.object(agent, "_prefetch_relevant_observations", new=prefetch),
        patch.object(agent, "_create_tool_executor", new=AsyncMock()),
    ):
        *_, returned = await agent._prepare_query(
            "what does alice do?",
            agent._telemetry_context("Dialectic Agent"),
        )

    assert order == ["run", "prefetch"]
    assert returned is run
    assert run.ends == []


@pytest.mark.asyncio
async def test_prepare_query_failure_ends_the_run_as_error() -> None:
    agent = _make_agent()
    run = FakeRun()

    with (
        patch("src.dialectic.core.start_captured_span", return_value=run),
        patch.object(agent, "_initialize_session_history", new=AsyncMock()),
        patch.object(
            agent, "_prefetch_relevant_observations", new=AsyncMock(return_value=None)
        ),
        patch.object(
            agent, "_create_tool_executor", new=AsyncMock(side_effect=RuntimeError)
        ),
        pytest.raises(RuntimeError),
    ):
        await agent._prepare_query("q", agent._telemetry_context())

    assert run.ends == [{"output": None, "is_error": True}]


def _prepare(run: FakeRun):
    return patch.object(
        DialecticAgent,
        "_prepare_query",
        new=AsyncMock(
            return_value=(AsyncMock(), "task", "run", time.perf_counter(), run)
        ),
    )


@pytest.mark.asyncio
async def test_answer_ends_the_run_with_the_answer() -> None:
    run = FakeRun()
    llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="robots", input_tokens=1, output_tokens=1, finish_reasons=["stop"]
        )
    )
    with (
        _prepare(run),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch("src.dialectic.core.honcho_llm_call", new=llm_call),
    ):
        assert await _make_agent().answer("q") == "robots"

    assert llm_call.await_args.kwargs["run"] is run  # pyright: ignore[reportOptionalMemberAccess]
    assert run.ends == [{"output": "robots", "is_error": False}]


@pytest.mark.asyncio
async def test_answer_failure_ends_the_run_as_error() -> None:
    run = FakeRun()
    with (
        _prepare(run),
        patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(side_effect=RuntimeError),
        ),
        pytest.raises(RuntimeError),
    ):
        await _make_agent().answer("q")

    assert run.ends == [{"output": None, "is_error": True}]


@pytest.mark.asyncio
async def test_answer_stream_ends_the_run_with_the_streamed_text() -> None:
    run = FakeRun()

    async def _stream():
        yield HonchoLLMCallStreamChunk(content="rob")
        yield HonchoLLMCallStreamChunk(content="ots")

    llm_call = AsyncMock(
        return_value=StreamingResponseWithMetadata(
            _stream(),
            tool_calls_made=[],
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=1,
        )
    )
    with (
        _prepare(run),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch("src.dialectic.core.honcho_llm_call", new=llm_call),
    ):
        chunks = [chunk async for chunk in _make_agent().answer_stream("q")]

    assert chunks == ["rob", "ots"]
    assert run.ends == [{"output": "robots", "is_error": False}]
