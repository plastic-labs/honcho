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


def _make_message(
    msg_id: int,
    content: str,
    metadata: dict | None = None,
    peer_name: str = "alice",
) -> Mock:
    return Mock(
        id=msg_id,
        public_id=f"msg_{msg_id}",
        session_name="session-1",
        workspace_name="workspace-1",
        peer_name=peer_name,
        content=content,
        token_count=5,
        created_at=datetime.now(timezone.utc),
        h_metadata=metadata if metadata is not None else {},
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

    def test_non_dict_metadata_is_eligible(self):
        """JSONB columns can technically hold scalars (rogue write or
        botched migration). The eligibility check must default to eligible
        rather than raise AttributeError on .get()."""
        for bad_value in ("stray-string", 42, ["a", "list"], True):
            msg = _make_message(1, "content")
            msg.h_metadata = bad_value
            assert _is_extraction_eligible(msg) is True, f"failed for {bad_value!r}"


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

    async def test_emits_minimal_completion_event_on_early_return(self):
        """All-non-prose batches must still emit a RepresentationCompletedEvent
        and increment metrics so the observability layer captures every
        batch outcome uniformly. Otherwise dashboards/replay systems lose
        all trace of these batches.
        """
        messages = [
            _make_message(1, "[diff removed]", metadata={"type": "user_paste_not_speech"}),
            _make_message(2, "[Tool] Edited foo.ts", metadata={"type": "tool_action"}),
        ]
        configuration = Mock()
        configuration.reasoning.enabled = True

        with patch(
            "src.deriver.deriver.honcho_llm_call",
            new_callable=AsyncMock,
        ) as mock_llm_call, patch(
            "src.deriver.deriver.emit",
        ) as mock_emit:
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=configuration,
                observers=["bob"],
                observed="alice",
                queue_item_message_ids=[1, 2],
            )

        mock_llm_call.assert_not_called()
        # Exactly one completion event should have been emitted, with
        # zero conclusions and zero token usage.
        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        assert event.explicit_conclusion_count == 0
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        assert event.message_count == 2
        assert event.queue_items_processed == 2

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

    async def test_skips_llm_when_target_messages_non_prose_with_eligible_other_peer_context(self):
        """Non-prose target queue items must skip LLM even when other-peer
        context messages in the same batch are eligible. Otherwise the
        deriver runs on context messages and produces facts about the
        observed peer based on the other peer's content (which is
        exactly the misattribution this PR is trying to prevent).
        """
        messages = [
            # Eligible context from a different peer (bob) — passes filter
            _make_message(10, "earlier turn from bob", peer_name="bob"),
            # Target queue item from observed peer (alice), tagged non-prose
            _make_message(11, "[diff removed]", metadata={"type": "user_paste_not_speech"}, peer_name="alice"),
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
                queue_item_message_ids=[11],  # only msg 11 is the target
            )

        mock_llm_call.assert_not_called()

    async def test_processes_when_target_is_eligible_with_context_in_batch(self):
        """Eligible target message must trigger the LLM even when the batch
        also contains other-peer context messages.
        """
        messages = [
            _make_message(10, "earlier turn from bob", peer_name="bob"),
            _make_message(11, "I want to ship the auth migration tonight.", peer_name="alice"),
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
                queue_item_message_ids=[11],
            )

        mock_llm_call.assert_called_once()
        prompt = mock_llm_call.await_args.kwargs["prompt"]
        # Both messages should appear (eligible context is preserved)
        assert "earlier turn from bob" in prompt
        assert "I want to ship the auth migration tonight" in prompt

    async def test_observation_provenance_excludes_filtered_messages(self):
        """Saved observations must cite only eligible source messages
        (target queue items for the observed peer that survived the
        non-prose filter), not tagged messages.
        """
        messages = [
            _make_message(20, "real prose from alice", peer_name="alice"),
            _make_message(21, "[diff removed]", metadata={"type": "user_paste_not_speech"}, peer_name="alice"),
            _make_message(22, "[Tool] Edited foo.ts", metadata={"type": "tool_action"}, peer_name="alice"),
        ]
        configuration = Mock()
        configuration.reasoning.enabled = True

        # Build a Representation result that won't be empty (so save_representation runs)
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
        ) as mock_llm_call, patch(
            "src.deriver.deriver.Representation.from_prompt_representation",
        ) as mock_from_prompt:
            await process_representation_tasks_batch(
                messages=messages,
                message_level_configuration=configuration,
                observers=["bob"],
                observed="alice",
                queue_item_message_ids=[20, 21, 22],
            )

        mock_llm_call.assert_called_once()
        # The provenance message_ids passed to the Representation factory
        # must contain ONLY message 20 (the eligible target). 21 and 22
        # are tagged non-prose and must not appear as evidence for any
        # derived fact.
        from_prompt_args = mock_from_prompt.call_args
        passed_message_ids = from_prompt_args[0][1]  # second positional arg
        assert passed_message_ids == [20]

    async def test_handles_non_dict_metadata_safely(self):
        """If h_metadata is somehow a non-dict scalar (rogue write,
        botched migration), the eligibility check must default to
        eligible rather than crash with AttributeError.
        """
        messages = [
            _make_message(1, "hello", metadata=None, peer_name="alice"),  # None
            _make_message(2, "world", peer_name="alice"),
        ]
        # Simulate a JSONB scalar by setting h_metadata directly to a non-dict
        messages[0].h_metadata = "stray-string-from-bad-write"

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
                queue_item_message_ids=[1, 2],
            )

        # Must not crash. Both messages should be eligible (non-dict
        # metadata defaults to eligible).
        mock_llm_call.assert_called_once()
