"""Tests for response_model threading through the DialecticAgent."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.dialectic.core import DialecticAgent
from src.exceptions import StructuredResponseError
from src.llm import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)
from src.llm.structured_output import StructuredOutputError


class FoodPreferences(BaseModel):
    favorite: str
    confidence: float


def _make_agent() -> DialecticAgent:
    return DialecticAgent(
        workspace_name="workspace",
        session_name="session",
        observer="observer",
        observed="observed",
        reasoning_level="low",
    )


def _patches(mock_llm_call: AsyncMock):
    return (
        patch.object(
            DialecticAgent,
            "_prepare_query",
            new=AsyncMock(
                return_value=(AsyncMock(), "task", "run", time.perf_counter(), None)
            ),
        ),
        patch.object(DialecticAgent, "_log_response_metrics"),
        patch("src.dialectic.core.honcho_llm_call", new=mock_llm_call),
    )


@pytest.mark.asyncio
async def test_answer_passes_response_model_and_serializes() -> None:
    """answer() threads response_model to the LLM call and serializes the
    parsed model instance back to a JSON string."""
    agent = _make_agent()
    parsed = FoodPreferences(favorite="sushi", confidence=0.9)
    mock_llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content=parsed,
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
    )

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3:
        result = await agent.answer("query", response_model=FoodPreferences)

    kwargs = mock_llm_call.await_args.kwargs  # pyright: ignore
    assert kwargs["response_model"] is FoodPreferences
    assert isinstance(result, str)
    assert FoodPreferences.model_validate_json(result) == parsed


@pytest.mark.asyncio
async def test_answer_without_response_model_returns_plain_text() -> None:
    agent = _make_agent()
    mock_llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="plain answer",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
    )

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3:
        result = await agent.answer("query")

    assert result == "plain answer"
    assert mock_llm_call.await_args.kwargs["response_model"] is None  # pyright: ignore


def _schema_failure() -> ValidationError:
    try:
        FoodPreferences.model_validate_json("I found sushi.")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [_schema_failure(), StructuredOutputError("No parsed content")],
    ids=["schema-mismatch", "no-content"],
)
async def test_answer_maps_structured_output_failure(failure: Exception) -> None:
    """A final answer that fails response_format surfaces as a 422, not a 500."""
    agent = _make_agent()
    mock_llm_call = AsyncMock(side_effect=failure)

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3, pytest.raises(StructuredResponseError) as caught:
        await agent.answer("query", response_model=FoodPreferences)

    assert caught.value.status_code == 422
    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "I can't help with that."])
async def test_answer_rejects_unparsed_content(content: str) -> None:
    """A backend that degrades to a plain string instead of raising (json_object
    mode's empty answer or a refusal) must still surface as a 422."""
    agent = _make_agent()
    mock_llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content=content,
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["stop"],
        )
    )

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3, pytest.raises(StructuredResponseError):
        await agent.answer("query", response_model=FoodPreferences)


@pytest.mark.asyncio
async def test_answer_without_response_model_does_not_map_validation_errors() -> None:
    agent = _make_agent()
    failure = _schema_failure()
    mock_llm_call = AsyncMock(side_effect=failure)

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3, pytest.raises(ValidationError):
        await agent.answer("query")


@pytest.mark.asyncio
async def test_answer_stream_passes_response_model() -> None:
    """answer_stream() threads response_model; chunks stay raw text."""
    agent = _make_agent()

    async def _stream():
        yield HonchoLLMCallStreamChunk(content='{"favorite":"sushi",')
        yield HonchoLLMCallStreamChunk(content='"confidence":0.9}')
        yield HonchoLLMCallStreamChunk(content="", is_done=True)

    mock_llm_call = AsyncMock(
        return_value=StreamingResponseWithMetadata(
            _stream(),
            tool_calls_made=[],
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=1,
        )
    )

    p1, p2, p3 = _patches(mock_llm_call)
    with p1, p2, p3:
        chunks = [
            chunk
            async for chunk in agent.answer_stream(
                "query", response_model=FoodPreferences
            )
        ]

    kwargs = mock_llm_call.await_args.kwargs  # pyright: ignore
    assert kwargs["response_model"] is FoodPreferences
    assert kwargs["stream_final_only"] is True
    accumulated = "".join(chunks)
    assert FoodPreferences.model_validate_json(accumulated).favorite == "sushi"
