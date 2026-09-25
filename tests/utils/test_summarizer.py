"""
Tests for src/utils/summarizer.py

Covers the _create_summary function's handling of empty, blocked, and
normal LLM responses, ensuring fallback logic prevents empty summaries
from being persisted.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.llm import HonchoLLMCallResponse
from src.utils.summarizer import (
    Summary,
    SummaryType,
    _create_summary,  # pyright: ignore[reportPrivateUsage]
    create_long_summary,
    create_short_summary,
)

# Common test arguments for _create_summary
_FORMATTED_MESSAGES = "user: hello\nassistant: hi there"
_INPUT_TOKENS = 100
_MESSAGE_PUBLIC_ID = "msg_abc123"
_LAST_MESSAGE_ID = 42
_LAST_MESSAGE_CONTENT_PREVIEW = "hello there how are you"
_MESSAGE_COUNT = 5
# Degenerate long-summary loop from #899 — non-empty, highly repetitive.
_DEGENERATE_TEXT = "Human. Forever. Human. Always. Human value. Always. " * 300
# Clean prose that can still hit the output cap (must not be rejected).
# Distinct-4 ratio must stay well above 0.35 — do not build this by repeating
# a single paragraph (that scores ~0.05 and would false-trigger the guard).
_CLEAN_CAP_HIT_TOPICS = [
    "project planning",
    "deadline prioritization",
    "communication preferences",
    "quarterly goals",
    "beta launch readiness",
    "analytics rewrite deferral",
    "stakeholder alignment",
    "risk mitigation",
    "capacity planning",
    "design review feedback",
    "API contract changes",
    "migration sequencing",
    "observability gaps",
    "incident response drills",
    "onboarding materials",
    "vendor evaluation",
    "budget reforecast",
    "security audit findings",
    "customer interviews",
    "feature flag rollout",
    "performance baselines",
    "dependency upgrades",
    "test coverage targets",
    "release checklist",
    "team rituals",
    "documentation debt",
    "support handoff notes",
    "partner integrations",
    "data retention policy",
    "accessibility fixes",
]
_CLEAN_CAP_HIT_TEXT = " ".join(
    f"In discussion segment {i + 1}, the participants covered {topic}. "
    f"They agreed on concrete next steps, owners, and a follow-up date. "
    f"Open questions around {topic} were parked for the next working session."
    for i, topic in enumerate(_CLEAN_CAP_HIT_TOPICS)
)


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

    @pytest.mark.parametrize("finish_reason", ["max_tokens", "length", "MAX_TOKENS"])
    async def test_degenerate_cap_hit_response_uses_fallback(self, finish_reason: str):
        """A summary that loops until the output cap is discarded, not persisted (#899)."""
        mock_response = HonchoLLMCallResponse(
            content=_DEGENERATE_TEXT,
            input_tokens=20000,
            output_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
            finish_reasons=[finish_reason],
        )
        with patch(
            "src.utils.summarizer.create_long_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, in_tok, out_tok = await _call_create_summary(
                SummaryType.LONG
            )

        assert is_fallback is True
        assert "Human. Forever." not in summary["content"]
        assert (in_tok, out_tok) == (0, 0)

    async def test_degenerate_cap_hit_short_summary_uses_fallback(self):
        """Shared guard covers SHORT as well as LONG (#899)."""
        mock_response = HonchoLLMCallResponse(
            content=_DEGENERATE_TEXT,
            input_tokens=5000,
            output_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
            finish_reasons=["max_tokens"],
        )
        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, in_tok, out_tok = await _call_create_summary(
                SummaryType.SHORT
            )

        assert is_fallback is True
        assert "Human. Forever." not in summary["content"]
        assert (in_tok, out_tok) == (0, 0)

    async def test_degenerate_stop_finish_keeps_content(self):
        """Cap-hit conjunct is required — stop + degenerate text is not rejected (#899)."""
        mock_response = HonchoLLMCallResponse(
            content=_DEGENERATE_TEXT,
            input_tokens=20000,
            output_tokens=500,
            finish_reasons=["stop"],
        )
        with patch(
            "src.utils.summarizer.create_long_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, _, _ = await _call_create_summary(SummaryType.LONG)

        assert is_fallback is False
        assert "Human. Forever." in summary["content"]

    async def test_degenerate_empty_finish_reasons_keeps_content(self):
        """Empty finish_reasons must not reject (#899)."""
        mock_response = HonchoLLMCallResponse(
            content=_DEGENERATE_TEXT,
            input_tokens=20000,
            output_tokens=500,
            finish_reasons=[],
        )
        with patch(
            "src.utils.summarizer.create_long_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, _, _ = await _call_create_summary(SummaryType.LONG)

        assert is_fallback is False
        assert "Human. Forever." in summary["content"]

    async def test_clean_prose_cap_hit_keeps_content(self):
        """Dense valid summary that hits the cap is still valid (#899)."""
        mock_response = HonchoLLMCallResponse(
            content=_CLEAN_CAP_HIT_TEXT,
            input_tokens=20000,
            output_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
            finish_reasons=["max_tokens"],
        )
        with patch(
            "src.utils.summarizer.create_long_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            summary, is_fallback, in_tok, out_tok = await _call_create_summary(
                SummaryType.LONG
            )

        assert is_fallback is False
        assert "project planning" in summary["content"]
        assert in_tok == 20000
        assert out_tok == settings.SUMMARY.MAX_TOKENS_LONG


@pytest.mark.asyncio
class TestSummaryCallerMigration:
    async def test_create_short_summary_uses_model_config(self):
        mock_response = HonchoLLMCallResponse(
            content="short summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await create_short_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                input_tokens=_INPUT_TOKENS,
                previous_summary=None,
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        kwargs = await_args.kwargs
        expected_config = settings.SUMMARY.MODEL_CONFIG
        assert "model_config" in kwargs
        assert kwargs["model_config"].model == expected_config.model
        assert "llm_settings" not in kwargs

    async def test_create_long_summary_uses_model_config(self):
        mock_response = HonchoLLMCallResponse(
            content="long summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await create_long_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                previous_summary=None,
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        kwargs = await_args.kwargs
        expected_config = settings.SUMMARY.MODEL_CONFIG
        assert "model_config" in kwargs
        assert kwargs["model_config"].model == expected_config.model
        assert "llm_settings" not in kwargs
