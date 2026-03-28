import time
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.dialectic.core import DialecticAgent
from src.utils.clients import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)


async def _stream_chunks() -> StreamingResponseWithMetadata:
    async def _stream():
        yield HonchoLLMCallStreamChunk(content="streamed")
        yield HonchoLLMCallStreamChunk(content="", is_done=True)

    return StreamingResponseWithMetadata(
        _stream(),
        tool_calls_made=[],
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        iterations=1,
    )


@pytest.mark.asyncio
async def test_dialectic_answer_uses_level_model_config() -> None:
    agent = DialecticAgent(
        db=cast(AsyncSession, Mock()),
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        reasoning_level="medium",
    )

    mock_response = HonchoLLMCallResponse(
        content="answer",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["stop"],
    )

    with (
        patch.object(
            DialecticAgent,
            "_prepare_query",
            new=AsyncMock(
                return_value=(AsyncMock(), "task", "run", time.perf_counter())
            ),
        ),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ) as mock_llm_call,
    ):
        result = await agent.answer("What do you know?")

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected dialectic LLM call")
    kwargs = await_args.kwargs
    expected_config = settings.DIALECTIC.LEVELS["medium"].MODEL_CONFIG
    if expected_config is None:
        raise AssertionError("Expected DIALECTIC medium MODEL_CONFIG to be resolved")

    assert result == "answer"
    assert kwargs["model_config"] == expected_config
    assert "llm_settings" not in kwargs
    assert "thinking_budget_tokens" not in kwargs


@pytest.mark.asyncio
async def test_dialectic_answer_stream_uses_level_model_config() -> None:
    agent = DialecticAgent(
        db=cast(AsyncSession, Mock()),
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        reasoning_level="medium",
    )

    with (
        patch.object(
            DialecticAgent,
            "_prepare_query",
            new=AsyncMock(
                return_value=(AsyncMock(), "task", "run", time.perf_counter())
            ),
        ),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=await _stream_chunks()),
        ) as mock_llm_call,
    ):
        chunks = [chunk async for chunk in agent.answer_stream("What do you know?")]

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected dialectic streaming LLM call")
    kwargs = await_args.kwargs
    expected_config = settings.DIALECTIC.LEVELS["medium"].MODEL_CONFIG
    if expected_config is None:
        raise AssertionError("Expected DIALECTIC medium MODEL_CONFIG to be resolved")

    assert chunks == ["streamed"]
    assert kwargs["model_config"] == expected_config
    assert "llm_settings" not in kwargs
    assert "thinking_budget_tokens" not in kwargs
