"""Pure classifier for the deriver's empty-extraction log level.

No database and no runtime mocks: the classifier only reads a response object, so
it lives in the isolated set (see _RUNTIME_MOCK_TEST_BLOCKLIST_PREFIXES in
tests/conftest.py).
"""

from typing import Any

from src.deriver.deriver import _extraction_was_contentless
from src.llm import HonchoLLMCallResponse
from src.utils.representation import PromptRepresentation


def _response(output_tokens: int, *finish_reasons: str) -> HonchoLLMCallResponse[Any]:
    """A minimal response carrying the given finish reasons and token count."""
    return HonchoLLMCallResponse(
        content=PromptRepresentation(explicit=[]),
        input_tokens=10,
        output_tokens=output_tokens,
        finish_reasons=list(finish_reasons),
    )


class TestExtractionWasContentless:
    """An empty representation is a fault only when the model returned nothing.

    A truncated completion, or a response with no output tokens, means the model
    emitted nothing at all - what a reasoning model does when its thinking is left
    on and it spends the whole output budget thinking. That deserves to be loud. A
    completed call that emits a valid but empty extraction found nothing worth
    recording, which is normal.
    """

    def test_truncated_completion_is_contentless(self) -> None:
        """An OpenAI-style 'length' finish reason means the model was cut off."""
        assert _extraction_was_contentless(_response(1500, "length"))

    def test_anthropic_truncation_is_contentless(self) -> None:
        """Anthropic reports the same truncation as 'max_tokens'."""
        assert _extraction_was_contentless(_response(1500, "max_tokens"))

    def test_gemini_truncation_is_contentless(self) -> None:
        """Gemini reports the enum name, which lowercases to the same value."""
        assert _extraction_was_contentless(_response(1500, "MAX_TOKENS"))

    def test_finish_reason_is_case_insensitive(self) -> None:
        """Provider casing must not decide whether a fault is reported."""
        assert _extraction_was_contentless(_response(1500, "LENGTH"))

    def test_reason_list_containing_truncation_is_contentless(self) -> None:
        """Truncation is detected anywhere in the reason list."""
        assert _extraction_was_contentless(_response(900, "stop", "length"))

    def test_no_output_tokens_is_contentless(self) -> None:
        """A response with no output tokens at all is a provider fault."""
        assert _extraction_was_contentless(_response(0, "stop"))

    def test_valid_but_empty_extraction_is_not_contentless(self) -> None:
        """A model that emitted valid, empty JSON found nothing to record."""
        assert not _extraction_was_contentless(_response(12, "stop"))

    def test_normal_extraction_is_not_contentless(self) -> None:
        """A normal extraction is never classified as contentless."""
        assert not _extraction_was_contentless(_response(64, "stop"))
