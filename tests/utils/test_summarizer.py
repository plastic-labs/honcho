"""
Tests for src/utils/summarizer.py

Covers the _create_summary function's handling of empty, blocked, and
normal LLM responses, ensuring fallback logic prevents empty summaries
from being persisted.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.utils.clients import HonchoLLMCallResponse
from src.utils.summarizer import (
    Summary,
    SummaryType,
    _create_summary,  # pyright: ignore[reportPrivateUsage]
)

# Common test arguments for _create_summary
_FORMATTED_MESSAGES = "user: hello\nassistant: hi there"
_INPUT_TOKENS = 100
_MESSAGE_PUBLIC_ID = "msg_abc123"
_LAST_MESSAGE_ID = 42
_LAST_MESSAGE_CONTENT_PREVIEW = "hello there how are you"
_MESSAGE_COUNT = 5


async def _call_create_summary(
    summary_type: SummaryType,
    *,
    message_count: int = _MESSAGE_COUNT,
    input_tokens: int = _INPUT_TOKENS,
) -> tuple[Summary, bool, int, int]:
    """Helper to call _create_summary with standard test arguments."""
    return await _create_summary(
        formatted_messages=_FORMATTED_MESSAGES,
        previous_summary_text=None,
        summary_type=summary_type,
        input_tokens=input_tokens,
        message_public_id=_MESSAGE_PUBLIC_ID,
        last_message_id=_LAST_MESSAGE_ID,
        last_message_content_preview=_LAST_MESSAGE_CONTENT_PREVIEW,
        message_count=message_count,
    )


@pytest.mark.asyncio
class TestCreateSummary:
    """Tests for the _create_summary function."""

    async def test_normal_response_succeeds(self):
        """Normal LLM response with content is preserved as-is."""
        mock_response = HonchoLLMCallResponse(
            content="User greeted the assistant and asked about the weather.",
            input_tokens=100,
            output_tokens=15,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            (
                summary,
                is_fallback,
                input_tokens,
                output_tokens,
            ) = await _call_create_summary(SummaryType.SHORT)

        assert is_fallback is False
        assert (
            summary["content"]
            == "User greeted the assistant and asked about the weather."
        )
        assert input_tokens == 100
        assert output_tokens == 15

    async def test_empty_response_uses_fallback(self):
        """Empty LLM response triggers fallback text instead of saving empty string."""
        mock_response = HonchoLLMCallResponse(
            content="",
            input_tokens=100,
            output_tokens=0,
            finish_reasons=["SAFETY"],
        )

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            (
                summary,
                is_fallback,
                input_tokens,
                output_tokens,
            ) = await _call_create_summary(SummaryType.SHORT)

        assert is_fallback is True
        assert "Conversation with 5 messages" in summary["content"]
        assert summary["content"] != ""
        assert input_tokens == 0
        assert output_tokens == 0

    async def test_whitespace_response_uses_fallback(self):
        """Whitespace-only LLM response is treated as empty."""
        mock_response = HonchoLLMCallResponse(
            content="   \n  \t  ",
            input_tokens=100,
            output_tokens=3,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            (
                summary,
                is_fallback,
                input_tokens,
                output_tokens,
            ) = await _call_create_summary(SummaryType.SHORT)

        assert is_fallback is True
        assert "Conversation with 5 messages" in summary["content"]
        assert input_tokens == 0
        assert output_tokens == 0

    async def test_exception_uses_fallback(self):
        """LLM exception triggers the existing fallback path."""
        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API timeout"),
        ):
            (
                summary,
                is_fallback,
                input_tokens,
                output_tokens,
            ) = await _call_create_summary(SummaryType.SHORT)

        assert is_fallback is True
        assert "Conversation with 5 messages" in summary["content"]
        assert input_tokens == 0
        assert output_tokens == 0

    async def test_long_type_routes_to_long_summary(self):
        """SummaryType.LONG calls create_long_summary, not create_short_summary."""
        mock_response = HonchoLLMCallResponse(
            content="A comprehensive summary of the conversation.",
            input_tokens=100,
            output_tokens=10,
            finish_reasons=["STOP"],
        )

        with (
            patch(
                "src.utils.summarizer.create_long_summary",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_long,
            patch(
                "src.utils.summarizer.create_short_summary",
                new_callable=AsyncMock,
            ) as mock_short,
        ):
            summary, is_fallback, _, _ = await _call_create_summary(SummaryType.LONG)

        assert is_fallback is False
        assert summary["content"] == "A comprehensive summary of the conversation."
        mock_long.assert_called_once()
        mock_short.assert_not_called()

    async def test_non_stop_finish_with_content_keeps_content(self):
        """Non-STOP finish reason with actual content is preserved (not a false positive)."""
        mock_response = HonchoLLMCallResponse(
            content="User discussed their project deadlines and asked for help prioritizing",
            input_tokens=100,
            output_tokens=12,
            finish_reasons=["MAX_TOKENS"],
        )

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, _, _ = await _call_create_summary(SummaryType.SHORT)

        assert is_fallback is False
        assert "project deadlines" in summary["content"]

    async def test_zero_message_count_empty_fallback(self):
        """Empty response with zero messages produces empty fallback text."""
        mock_response = HonchoLLMCallResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            finish_reasons=["SAFETY"],
        )

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, _, _ = await _call_create_summary(
                SummaryType.SHORT, message_count=0, input_tokens=0
            )

        assert is_fallback is True
        assert summary["content"] == ""
        assert summary["token_count"] == 0
