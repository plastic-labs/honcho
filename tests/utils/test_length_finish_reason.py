"""
Tests for JSON repair handling across all providers in honcho_llm_call_inner,
and Gemini thinking budget support.

Verifies that when an LLM hits the max token limit or returns malformed JSON,
the truncated output is repaired and returned instead of crashing.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, Usage
from openai import AsyncOpenAI, LengthFinishReasonError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from src.utils.clients import CLIENTS, HonchoLLMCallResponse, honcho_llm_call_inner
from src.utils.representation import PromptRepresentation

# --- Test models ---


class SimpleModel(BaseModel):
    """Non-PromptRepresentation model for testing re-raise behavior."""

    items: list[str]


# --- Helpers ---

VALID_REPR_JSON = {
    "explicit": [
        {"content": "hermes is 25 years old"},
        {"content": "hermes has a dog"},
    ]
}


def _make_truncated_completion(content: str) -> ChatCompletion:
    """Build a ChatCompletion with finish_reason='length' and the given content."""
    return ChatCompletion(
        id="test-truncated",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="length",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=1000, completion_tokens=2000, total_tokens=3000
        ),
    )


def _raise_length_error(content: str) -> AsyncMock:
    """Return an AsyncMock that raises LengthFinishReasonError with truncated content."""
    completion = _make_truncated_completion(content)
    return AsyncMock(side_effect=LengthFinishReasonError(completion=completion))


def _make_anthropic_mock(text: str, stop_reason: str = "end_turn") -> AsyncMock:
    """Build a mocked AsyncAnthropic client returning the given text."""
    mock_client = AsyncMock(spec=AsyncAnthropic)
    mock_response = Mock()
    mock_response.content = [TextBlock(text=text, type="text")]
    mock_response.usage = Usage(input_tokens=100, output_tokens=50)
    mock_response.stop_reason = stop_reason
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


def _make_gemini_mock(
    text: str | None = None,
    parsed: Any = None,
    finish_reason_name: str = "STOP",
) -> Mock:
    """Build a mocked genai.Client returning the given text/parsed content."""
    mock_client = Mock()

    # Build response
    mock_response = Mock()
    mock_response.parsed = parsed

    # Candidates
    mock_candidate = Mock()
    mock_finish_reason = Mock()
    mock_finish_reason.name = finish_reason_name
    mock_candidate.finish_reason = mock_finish_reason

    # Content parts
    if text is not None:
        mock_part = Mock()
        mock_part.text = text
        mock_part.function_call = None
        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
    else:
        mock_candidate.content = None

    mock_response.candidates = [mock_candidate]

    # Usage
    mock_usage = Mock()
    mock_usage.prompt_token_count = 200
    mock_usage.candidates_token_count = 100
    mock_response.usage_metadata = mock_usage

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return mock_client


# ---------------------------------------------------------------------------
# OpenAI / Custom provider tests (LengthFinishReasonError path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenAILengthFinishReasonRepair:
    """Tests that LengthFinishReasonError is caught and truncated JSON is repaired."""

    async def test_truncated_prompt_representation_repaired_openai(self) -> None:
        """Truncated but repairable PromptRepresentation JSON should be repaired (openai)."""
        truncated_json = json.dumps(VALID_REPR_JSON)[:-2]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error(truncated_json)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response, HonchoLLMCallResponse)
        assert isinstance(response.content, PromptRepresentation)
        assert len(response.content.explicit) >= 1
        assert response.finish_reasons == ["length"]
        assert response.output_tokens == 2000

    async def test_truncated_prompt_representation_repaired_openai_with_custom_base(
        self,
    ) -> None:
        """Truncated but repairable PromptRepresentation JSON should be repaired."""
        truncated_json = json.dumps(VALID_REPR_JSON)[:-2]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error(truncated_json)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response, HonchoLLMCallResponse)
        assert isinstance(response.content, PromptRepresentation)
        assert len(response.content.explicit) >= 1
        assert response.finish_reasons == ["length"]

    async def test_completely_broken_json_falls_back_to_empty(self) -> None:
        """Completely unrepairable JSON should fall back to empty PromptRepresentation."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error(
            "this is not json at all just random text"
        )

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert response.content.explicit == []
        assert response.finish_reasons == ["length"]

    async def test_empty_content_falls_back_to_empty(self) -> None:
        """Empty/null content should fall back to empty PromptRepresentation."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error("")

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert response.content.explicit == []

    async def test_non_prompt_representation_reraises_on_unfixable(self) -> None:
        """Non-PromptRepresentation with unrepairable JSON should raise ValidationError."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error("not json")

        with (
            patch.dict(CLIENTS, {"openai": mock_client}),
            pytest.raises(ValidationError),
        ):
            await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Generate items",
                max_tokens=2000,
                response_model=SimpleModel,
                json_mode=True,
            )

    async def test_token_counts_preserved(self) -> None:
        """Token counts from the truncated completion should be preserved."""
        truncated_json = '{"explicit": [{"content": "fact one"}'

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error(truncated_json)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert response.input_tokens == 1000
        assert response.output_tokens == 2000

    async def test_valid_json_with_length_finish_reason(self) -> None:
        """Valid JSON despite length truncation should parse fine."""
        valid_json = json.dumps(VALID_REPR_JSON)

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.chat.completions.parse = _raise_length_error(valid_json)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="test-model",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert len(response.content.explicit) == 2
        assert response.content.explicit[0].content == "hermes is 25 years old"


# ---------------------------------------------------------------------------
# Anthropic provider tests (JSON parse failure -> repair path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAnthropicJsonRepair:
    """Tests that Anthropic response_model parse failures trigger JSON repair."""

    async def test_truncated_anthropic_response_repaired(self) -> None:
        """Truncated Anthropic JSON response should be repaired."""
        # Anthropic prefills "{" so the response text starts after that
        # The code prepends "{" back: json_content = "{" + text_content
        truncated_text = json.dumps(VALID_REPR_JSON)[
            1:-2
        ]  # Remove leading { and trailing }]

        mock_client = _make_anthropic_mock(truncated_text, stop_reason="max_tokens")

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert len(response.content.explicit) >= 1

    async def test_broken_anthropic_response_falls_back_to_empty(self) -> None:
        """Completely broken Anthropic JSON should fall back to empty PromptRepresentation."""
        mock_client = _make_anthropic_mock(
            "random gibberish that is not json", stop_reason="max_tokens"
        )

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert response.content.explicit == []

    async def test_non_prompt_representation_reraises(self) -> None:
        """Non-PromptRepresentation with broken JSON should raise."""
        mock_client = _make_anthropic_mock("not json", stop_reason="max_tokens")

        with (
            patch.dict(CLIENTS, {"anthropic": mock_client}),
            pytest.raises(ValidationError),
        ):
            await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Generate items",
                max_tokens=2000,
                response_model=SimpleModel,
                json_mode=True,
            )


# ---------------------------------------------------------------------------
# Gemini provider tests (parsed=None or type mismatch -> repair path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGeminiJsonRepair:
    """Tests that Gemini response_model parse failures trigger JSON repair."""

    async def test_gemini_unparsed_response_repaired(self) -> None:
        """Gemini returning text but no parsed object should repair from raw text."""
        from google import genai

        valid_text = json.dumps(VALID_REPR_JSON)
        mock_client = _make_gemini_mock(
            text=valid_text, parsed=None, finish_reason_name="MAX_TOKENS"
        )

        with (
            patch.dict(CLIENTS, {"gemini": mock_client}),
            patch.object(genai.Client, "__instancecheck__", return_value=True),
        ):
            # We need the match statement to hit the genai.Client case
            mock_client.__class__ = genai.Client  # pyright: ignore[reportAttributeAccessIssue]
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert len(response.content.explicit) == 2

    async def test_gemini_broken_text_falls_back_to_empty(self) -> None:
        """Gemini with broken text and no parsed content should fall back."""
        from google import genai

        mock_client = _make_gemini_mock(
            text="broken json", parsed=None, finish_reason_name="MAX_TOKENS"
        )
        mock_client.__class__ = genai.Client  # pyright: ignore[reportAttributeAccessIssue]

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Analyze messages",
                max_tokens=2000,
                response_model=PromptRepresentation,
                json_mode=True,
            )

        assert isinstance(response.content, PromptRepresentation)
        assert response.content.explicit == []


# ---------------------------------------------------------------------------
# Gemini thinking budget tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGeminiThinkingBudget:
    """Tests that thinking_budget_tokens is passed to Gemini via ThinkingConfig."""

    async def test_thinking_budget_passed_to_gemini(self) -> None:
        """thinking_budget_tokens should be included in Gemini config."""
        from google import genai

        mock_client = _make_gemini_mock(text="Hello", parsed=None)
        mock_client.__class__ = genai.Client  # pyright: ignore[reportAttributeAccessIssue]

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Think about this",
                max_tokens=2000,
                thinking_budget_tokens=4096,
            )

        # Verify generate_content was called with thinking_config
        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")
        assert config is not None
        assert "thinking_config" in config
        assert config["thinking_config"]["thinking_budget"] == 4096

    async def test_no_thinking_config_when_budget_is_none(self) -> None:
        """When thinking_budget_tokens is None, thinking_config should not be set."""
        from google import genai

        mock_client = _make_gemini_mock(text="Hello", parsed=None)
        mock_client.__class__ = genai.Client  # pyright: ignore[reportAttributeAccessIssue]

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="No thinking needed",
                max_tokens=2000,
            )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")
        if config:
            assert "thinking_config" not in config
