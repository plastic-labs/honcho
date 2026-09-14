"""
Tests for src/utils/summarizer.py

Covers the _create_summary function's handling of empty, blocked, and
normal LLM responses, ensuring fallback logic prevents empty summaries
from being persisted.
"""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.exceptions import EmptySummaryError
from src.llm import HonchoLLMCallResponse
from src.utils.summarizer import (
    Summary,
    SummaryType,
    _create_summary,  # pyright: ignore[reportPrivateUsage]
    create_long_summary,
    create_short_summary,
    summary_prompt,
)

# Common test arguments for _create_summary
_FORMATTED_MESSAGES = "user: hello\nassistant: hi there"
_INPUT_TOKENS = 100
_MESSAGE_PUBLIC_ID = "msg_abc123"
_LAST_MESSAGE_ID = 42
_LAST_MESSAGE_CONTENT_PREVIEW = "hello there how are you"
_MESSAGE_COUNT = 5

# The real coroutines, bound at import time (before the autouse
# `mock_llm_call_functions` fixture swaps the module attributes for stubs).
# `TestSummaryPromptIdentity` restores them so the real prompt builder runs.
_REAL_CREATE_SHORT_SUMMARY = create_short_summary
_REAL_CREATE_LONG_SUMMARY = create_long_summary


async def _call_create_summary(
    summary_type: SummaryType,
    *,
    message_count: int = _MESSAGE_COUNT,
    input_tokens: int = _INPUT_TOKENS,
    previous_summary_text: str | None = None,
) -> tuple[Summary, bool, int, int]:
    """Helper to call _create_summary with standard test arguments."""
    return await _create_summary(
        formatted_messages=_FORMATTED_MESSAGES,
        previous_summary_text=previous_summary_text,
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

    async def test_degraded_empty_response_is_refused(self):
        """An empty response the provider cut off is refused, not papered over.

        The placeholder fallback is never persisted (the only `_save_summary`
        call is gated on `not is_fallback`), so accepting it would consume the
        summary boundary with nothing stored. Raising lets the queue re-claim
        the item under its bounded retry budget.
        """
        mock_response = HonchoLLMCallResponse(
            content="",
            input_tokens=100,
            output_tokens=0,
            finish_reasons=["SAFETY"],
        )

        with (
            patch(
                "src.utils.summarizer.create_short_summary",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(EmptySummaryError, match="empty summary"),
        ):
            await _call_create_summary(SummaryType.SHORT)

    async def test_truncated_empty_response_is_refused(self):
        """`length` with nothing in the body is the measured production cause."""
        mock_response = HonchoLLMCallResponse(
            content="   \n",
            input_tokens=4000,
            output_tokens=16000,
            finish_reasons=["length"],
        )

        with (
            patch(
                "src.utils.summarizer.create_short_summary",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            pytest.raises(
                EmptySummaryError, match="finish_reasons=\\['length'\\]"
            ) as exc,
        ):
            await _call_create_summary(SummaryType.SHORT)

        failure = exc.value
        assert failure.parse_class == "truncated"
        assert failure.prompt_bytes > 0
        assert len(failure.prompt_digest) == 16
        assert failure.provider and failure.model

    async def test_legitimately_empty_response_uses_fallback(self):
        """A clean stop with tokens spent is a real (empty) answer: no retry."""
        mock_response = HonchoLLMCallResponse(
            content="  ",
            input_tokens=100,
            output_tokens=12,
            finish_reasons=["stop"],
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


@pytest.mark.asyncio
class TestSummaryPromptIdentity:
    """The prompt the provider saw is the prompt the error triage fields name.

    `EmptySummaryError.prompt_digest` / `prompt_bytes` exist to correlate a
    repeated failure, so they must describe the string actually handed to
    `honcho_llm_call` -- instructions, previous summary, messages and the hard
    word limit -- using the deriver's 16-hex convention
    (`deriver.py` raises `EmptyRepresentationError` the same way). Hashing only
    `formatted_messages` makes two different prompts collide.

    The autouse `mock_llm_call_functions` fixture replaces the *module
    attributes* `create_short_summary` / `create_long_summary` with stubs that
    return a plain string, so `_create_summary` would never reach the prompt
    builder. These tests restore the real coroutines (bound at import time) for
    the duration of the test, because the real prompt-building path is the
    subject under test.
    """

    @staticmethod
    def _degraded() -> HonchoLLMCallResponse[str]:
        """A response the provider cut off: empty body, `length`, no tokens."""
        return HonchoLLMCallResponse(
            content="",
            input_tokens=100,
            output_tokens=0,
            finish_reasons=["length"],
        )

    async def _run_degraded(
        self,
        summary_type: SummaryType,
        previous_summary_text: str | None = None,
    ) -> tuple[str, EmptySummaryError]:
        short = summary_type is SummaryType.SHORT
        with (
            patch(
                "src.utils.summarizer.create_short_summary"
                if short
                else "src.utils.summarizer.create_long_summary",
                side_effect=(
                    _REAL_CREATE_SHORT_SUMMARY if short else _REAL_CREATE_LONG_SUMMARY
                ),
            ),
            patch(
                "src.utils.summarizer.honcho_llm_call",
                new_callable=AsyncMock,
                return_value=self._degraded(),
            ) as mock_llm_call,
            pytest.raises(EmptySummaryError, match="empty summary") as exc,
        ):
            await _call_create_summary(
                summary_type, previous_summary_text=previous_summary_text
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        sent_prompt = await_args.kwargs["prompt"]
        return sent_prompt, exc.value

    async def test_create_short_summary_prompt_comes_from_the_builder(self):
        """Single source of truth: the caller and the error path agree."""
        response = HonchoLLMCallResponse(
            content="short summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )
        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_llm_call:
            await create_short_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                input_tokens=_INPUT_TOKENS,
                previous_summary=None,
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        assert await_args.kwargs["prompt"] == summary_prompt(
            SummaryType.SHORT, _FORMATTED_MESSAGES, None, _INPUT_TOKENS
        )

    async def test_create_long_summary_prompt_comes_from_the_builder(self):
        response = HonchoLLMCallResponse(
            content="long summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )
        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_llm_call:
            await create_long_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                previous_summary=None,
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        assert await_args.kwargs["prompt"] == summary_prompt(
            SummaryType.LONG, _FORMATTED_MESSAGES, None
        )

    async def test_short_summary_digest_identifies_the_sent_prompt(self):
        sent_prompt, failure = await self._run_degraded(SummaryType.SHORT)

        assert sent_prompt == summary_prompt(
            SummaryType.SHORT, _FORMATTED_MESSAGES, None, _INPUT_TOKENS
        )
        assert failure.prompt_bytes == len(sent_prompt.encode("utf-8"))
        assert (
            failure.prompt_digest
            == hashlib.sha256(sent_prompt.encode("utf-8")).hexdigest()[:16]
        )
        assert len(failure.prompt_digest) == 16
        # The prompt is not just the messages: a digest over the bare messages
        # would be both shorter and a different value.
        assert failure.prompt_bytes > len(_FORMATTED_MESSAGES.encode("utf-8"))
        assert (
            failure.prompt_digest
            != hashlib.sha256(_FORMATTED_MESSAGES.encode("utf-8")).hexdigest()[:16]
        )

    async def test_long_summary_digest_identifies_the_sent_prompt(self):
        sent_prompt, failure = await self._run_degraded(SummaryType.LONG)

        assert sent_prompt == summary_prompt(
            SummaryType.LONG, _FORMATTED_MESSAGES, None
        )
        assert failure.prompt_bytes == len(sent_prompt.encode("utf-8"))
        assert (
            failure.prompt_digest
            == hashlib.sha256(sent_prompt.encode("utf-8")).hexdigest()[:16]
        )
        assert (
            failure.prompt_digest
            != hashlib.sha256(_FORMATTED_MESSAGES.encode("utf-8")).hexdigest()[:16]
        )

    async def test_different_previous_summaries_do_not_share_a_digest(self):
        """The regression the finding names: only the previous summary differs."""
        results: list[tuple[str, str]] = []
        for previous in ("previous summary A", "previous summary B"):
            sent_prompt, failure = await self._run_degraded(
                SummaryType.SHORT, previous_summary_text=previous
            )
            assert previous in sent_prompt
            assert (
                failure.prompt_digest
                == hashlib.sha256(sent_prompt.encode("utf-8")).hexdigest()[:16]
            )
            results.append((sent_prompt, failure.prompt_digest))

        assert results[0][0] != results[1][0]
        assert results[0][1] != results[1][1]
