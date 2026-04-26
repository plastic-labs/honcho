"""
Tests for the deriver's metadata.type filter that excludes non-prose messages
from peer representation extraction.

Background: messages tagged metadata.type ∈ {"user_paste_not_speech",
"tool_action"} are uploaded by clients (e.g. plastic-labs/claude-honcho) for
archival/replay but should not contribute to peer representation. They ride
role: "user" per the Anthropic Messages API but contain pasted code/diffs or
explicit tool actions performed on the user's behalf.

See plastic-labs/claude-honcho#34 for the upstream plugin tags.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.deriver.deriver import (
    NON_PROSE_METADATA_TYPES,
    _is_extraction_eligible,
    process_representation_tasks_batch,
)
from src.llm import HonchoLLMCallResponse
from src.utils.representation import PromptRepresentation


def _make_message(msg_id: int, content: str, metadata: dict | None = None) -> Mock:
    return Mock(
        id=msg_id,
        public_id=f"msg_{msg_id}",
        session_name="session-1",
        workspace_name="workspace-1",
        peer_name="alice",
        content=content,
        token_count=5,
        created_at=datetime.now(timezone.utc),
        h_metadata=metadata or {},
    )


class TestExtractionEligibility:
    """Direct unit tests for the _is_extraction_eligible predicate."""

    def test_default_metadata_is_eligible(self):
        msg = _make_message(1, "hello world")
        assert _is_extraction_eligible(msg) is True

    def test_empty_metadata_is_eligible(self):
        msg = _make_message(1, "hello world", metadata={})
        assert _is_extraction_eligible(msg) is True

    def test_unrelated_metadata_type_is_eligible(self):
        msg = _make_message(1, "hello", metadata={"type": "assistant_prose"})
        assert _is_extraction_eligible(msg) is True

    def test_user_paste_not_speech_is_filtered(self):
        msg = _make_message(1, "[diff removed]", metadata={"type": "user_paste_not_speech"})
        assert _is_extraction_eligible(msg) is False

    def test_tool_action_is_filtered(self):
        msg = _make_message(1, "[Tool] Edited foo.ts", metadata={"type": "tool_action"})
        assert _is_extraction_eligible(msg) is False

    def test_non_prose_set_contains_known_tags(self):
        assert "user_paste_not_speech" in NON_PROSE_METADATA_TYPES
        assert "tool_action" in NON_PROSE_METADATA_TYPES


@pytest.mark.asyncio
class TestProcessRepresentationFiltering:
    """End-to-end tests for the deriver's batch processing with the filter."""

    async def test_skips_llm_call_when_all_messages_non_prose(self):
        """When every message in the batch is tagged non-prose, no LLM call is made."""
        messages = [
            _make_message(1, "[code block removed]", metadata={"type": "user_paste_not_speech"}),
            _make_message(2, "[Tool] Wrote foo.ts", metadata={"type": "tool_action"}),
        ]
        configuration = Mock()
        configuration.reasoning.enabled = True

        with patch(
            "src.deriver.deriver.honcho_llm_call",
            new_callable=AsyncMock,
        ) as mock_llm_call:
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=configuration,
                observers=["bob"],
                observed="alice",
                queue_item_message_ids=[1, 2],
            )

        mock_llm_call.assert_not_called()

    async def test_filters_non_prose_from_llm_prompt(self):
        """Non-prose messages are excluded from the LLM-facing prompt; prose messages remain."""
        messages = [
            _make_message(1, "I want to refactor the auth middleware."),
            _make_message(2, "[diff removed]", metadata={"type": "user_paste_not_speech"}),
            _make_message(3, "Look for race conditions please."),
            _make_message(4, "[Tool] Edited middleware.ts", metadata={"type": "tool_action"}),
        ]
        configuration = Mock()
        configuration.reasoning.enabled = True

        mock_response = HonchoLLMCallResponse(
            content=PromptRepresentation(explicit=[]),
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.deriver.deriver.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=configuration,
                observers=["bob"],
                observed="alice",
                queue_item_message_ids=[1, 2, 3, 4],
            )

        mock_llm_call.assert_called_once()
        prompt = mock_llm_call.await_args.kwargs["prompt"]
        assert "I want to refactor the auth middleware" in prompt
        assert "Look for race conditions please" in prompt
        assert "[diff removed]" not in prompt
        assert "[Tool] Edited middleware.ts" not in prompt

    async def test_passes_prose_messages_unchanged(self):
        """Prose messages without tags are passed to the LLM as before."""
        messages = [_make_message(1, "I prefer fail-closed defaults.")]
        configuration = Mock()
        configuration.reasoning.enabled = True

        mock_response = HonchoLLMCallResponse(
            content=PromptRepresentation(explicit=[]),
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.deriver.deriver.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=configuration,
                observers=["bob"],
                observed="alice",
                queue_item_message_ids=[1],
            )

        mock_llm_call.assert_called_once()
        prompt = mock_llm_call.await_args.kwargs["prompt"]
        assert "I prefer fail-closed defaults" in prompt
