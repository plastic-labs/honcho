import time
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dialectic.core import DialecticAgent
from src.llm import (
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


def test_build_dialectic_langfuse_metadata_is_pure_and_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "myah-")
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "myah")
    agent = DialecticAgent(
        workspace_name="myah-user-123",
        session_name="chat-abc",
        observer="myah",
        observed="user-123",
        reasoning_level="medium",
    )

    metadata, trace_attrs = agent._build_langfuse_metadata_for_call()

    assert metadata["honcho_operation"] == "dialectic_chat"
    assert metadata["honcho_workspace_id"] == "myah-user-123"
    assert metadata["honcho_session_id"] == "chat-abc"
    assert metadata["honcho_observer_peer"] == "myah"
    assert metadata["honcho_observed_peer"] == "user-123"
    assert metadata["honcho_reasoning_level"] == "medium"
    assert metadata["tenant_user_id"] == "user-123"
    assert metadata["tenant_platform"] == "myah"
    assert "honcho_run_id" in metadata
    assert trace_attrs == {
        "user_id": "user-123",
        "session_id": "chat-abc",
        "tags": ["honcho", "memory", "dialectic_chat", "myah"],
    }
    assert "query" not in metadata
    assert "prompt" not in metadata
    assert "content" not in str(metadata).lower()


@pytest.mark.asyncio
async def test_dialectic_answer_passes_langfuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "myah-")
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "myah")
    agent = DialecticAgent(
        workspace_name="myah-user-123",
        session_name="chat-abc",
        observer="myah",
        observed="user-123",
        reasoning_level="low",
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

    assert result == "answer"
    assert kwargs["track_name"] == "Dialectic Agent"
    assert kwargs["langfuse_metadata"]["honcho_operation"] == "dialectic_chat"
    assert kwargs["langfuse_metadata"]["honcho_workspace_id"] == "myah-user-123"
    assert kwargs["langfuse_metadata"]["honcho_session_id"] == "chat-abc"
    assert kwargs["langfuse_trace_user_id"] == "user-123"
    assert kwargs["langfuse_trace_session_id"] == "chat-abc"
    assert kwargs["langfuse_trace_tags"] == [
        "honcho",
        "memory",
        "dialectic_chat",
        "myah",
    ]


@pytest.mark.asyncio
async def test_dialectic_answer_stream_passes_langfuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "myah-")
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "myah")
    agent = DialecticAgent(
        workspace_name="myah-user-123",
        session_name="chat-abc",
        observer="myah",
        observed="user-123",
        reasoning_level="low",
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

    assert chunks == ["streamed"]
    assert kwargs["track_name"] == "Dialectic Agent Stream"
    assert kwargs["langfuse_metadata"]["honcho_operation"] == "dialectic_chat"
    assert kwargs["langfuse_trace_user_id"] == "user-123"
    assert kwargs["langfuse_trace_session_id"] == "chat-abc"
    assert kwargs["langfuse_trace_tags"] == [
        "honcho",
        "memory",
        "dialectic_chat",
        "myah",
    ]
