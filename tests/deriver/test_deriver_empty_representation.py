"""Retry-on-empty-representation behaviour for the minimal deriver.

The loss mode this covers: a structured response that parses to a *valid but
empty* representation used to be logged (`Deriver generated zero observations
...`) and then consumed -- the batch was marked processed with nothing derived
from it. These tests pin the replacement behaviour:

* an empty parse is re-requested in-line, bounded by
  ``DERIVER.EMPTY_PARSE_MAX_ATTEMPTS``;
* a response that still looks degraded afterwards (`length`/`content_filter`
  finish reason, or zero output tokens) raises ``EmptyRepresentationError`` so
  the queue item is *not* consumed;
* a legitimately empty batch (clean ``stop`` with tokens spent) keeps today's
  semantics -- warning, consumed, no requeue;
* the non-empty path stays a single call.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src import crud
from src.config import settings
from src.crud.representation import RepresentationManager
from src.deriver.deriver import process_representation_tasks_batch
from src.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS
from src.exceptions import EmptyRepresentationError
from src.llm import HonchoLLMCallResponse
from src.llm.types import degraded_response_parse_class, empty_response_is_degraded
from src.utils.representation import (
    ExplicitObservationBase,
    PromptRepresentation,
)


def _message(message_id: int = 1, peer_name: str = "alice") -> Mock:
    return Mock(
        id=message_id,
        public_id=f"msg_{message_id}",
        session_name="session-1",
        workspace_name="workspace-1",
        peer_name=peer_name,
        content="hello",
        token_count=5,
        created_at=datetime.now(UTC),
    )


def _response(
    explicit: list[str] | None = None,
    *,
    finish_reasons: list[str] | None = None,
    output_tokens: int = 5,
) -> HonchoLLMCallResponse[PromptRepresentation]:
    return HonchoLLMCallResponse(
        content=PromptRepresentation(
            explicit=[
                ExplicitObservationBase(content=content) for content in (explicit or [])
            ]
        ),
        input_tokens=100,
        output_tokens=output_tokens,
        finish_reasons=finish_reasons or ["stop"],
    )


EMPTY_TRUNCATED = _response([], finish_reasons=["length"], output_tokens=0)
EMPTY_LEGITIMATE = _response([], finish_reasons=["stop"], output_tokens=7)
NON_EMPTY = _response(["The user has a dog named Rover"])


async def _run_batch(
    llm_responses: list[HonchoLLMCallResponse[PromptRepresentation]],
    *,
    messages: list[Mock] | None = None,
    emitted: list[Any] | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Drive one batch through the deriver with scripted LLM responses.

    Returns (llm_call_mock, save_representation_mock).
    """
    configuration = Mock()
    configuration.reasoning.enabled = True
    configuration.reasoning.custom_instructions = None

    llm_call = AsyncMock(side_effect=llm_responses)
    save = AsyncMock(return_value=crud.CreateDocumentsResult())
    captured: list[Any] = emitted if emitted is not None else []

    with (
        patch("src.deriver.deriver.honcho_llm_call", new=llm_call),
        patch.object(RepresentationManager, "save_representation", save),
        patch("src.deriver.deriver.emit", side_effect=captured.append),
    ):
        await process_representation_tasks_batch(
            messages=messages or [_message()],  # pyright: ignore[reportArgumentType]
            message_level_configuration=configuration,
            observers=["bob"],
            observed="alice",
            queue_item_message_ids=[1],
        )
    return llm_call, save


@pytest.mark.asyncio
class TestEmptyRepresentationRetry:
    async def test_discriminator_flags_degraded_responses_only(self):
        """The suspicion signal is the provider's, not the parser's."""
        assert empty_response_is_degraded(EMPTY_TRUNCATED) is True
        assert (
            empty_response_is_degraded(
                _response([], finish_reasons=["content_filter"], output_tokens=9)
            )
            is True
        )
        assert (
            empty_response_is_degraded(
                _response([], finish_reasons=["stop"], output_tokens=0)
            )
            is True
        )
        assert empty_response_is_degraded(EMPTY_LEGITIMATE) is False

    async def test_degraded_parse_class_classifies_the_cause(self):
        """Triage class for alerting: truncated vs filtered vs empty body."""
        assert degraded_response_parse_class(EMPTY_TRUNCATED) == "truncated"
        assert (
            degraded_response_parse_class(
                _response([], finish_reasons=["content_filter"], output_tokens=9)
            )
            == "content_filtered"
        )
        assert (
            degraded_response_parse_class(
                _response([], finish_reasons=["stop"], output_tokens=0)
            )
            == "empty_body"
        )
        assert degraded_response_parse_class(EMPTY_LEGITIMATE) == "unknown"

    async def test_empty_then_non_empty_is_saved(self, monkeypatch):
        """The acceptance shape: 1st response empty, 2nd non-empty -> saved."""
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        emitted: list[Any] = []
        llm_call, save = await _run_batch([EMPTY_TRUNCATED, NON_EMPTY], emitted=emitted)

        assert llm_call.await_count == 2
        save.assert_awaited_once()
        assert emitted, "expected a telemetry event"
        assert emitted[-1].explicit_conclusion_count == 1
        assert emitted[-1].empty_parse_attempts == 2

    async def test_retries_are_bounded_and_refused(self, monkeypatch):
        """A persistently degraded parse raises instead of consuming the batch."""
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_MAX_ATTEMPTS", 2)
        emitted: list[Any] = []

        with pytest.raises(
            EmptyRepresentationError, match="empty representation"
        ) as exc:
            await _run_batch(
                [EMPTY_TRUNCATED, EMPTY_TRUNCATED, EMPTY_TRUNCATED],
                emitted=emitted,
            )

        # The attempt count is observable in the emitted event before the refusal.
        assert emitted and emitted[-1].empty_parse_attempts == 3
        assert emitted[-1].empty_parse_refused is True
        assert emitted[-1].explicit_conclusion_count == 0

        # ...and the failure itself is typed for triage (issue #993 shape),
        # carrying a prompt digest rather than the prompt text.
        failure = exc.value
        assert failure.parse_class == "truncated"
        assert failure.attempts == 3
        assert failure.prompt_bytes > 0
        assert len(failure.prompt_digest) == 16
        assert failure.provider and failure.model
        assert "hello" not in str(failure), "the failure must not carry message text"

    async def test_zero_output_tokens_is_refused(self, monkeypatch):
        """A provider that returns literally nothing is not a legitimate empty."""
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        nothing = _response([], finish_reasons=["stop"], output_tokens=0)
        with pytest.raises(EmptyRepresentationError, match="output_tokens=0"):
            await _run_batch([nothing, nothing])

    async def test_legitimately_empty_batch_is_consumed(self, monkeypatch):
        """Clean stop + tokens spent = the model asserted nothing: no refusal.

        The batch is still re-requested once (an empty parse is indistinguishable
        at this layer until the response signals are read), but it is *not*
        refused, so it never costs a work-unit requeue.
        """
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        emitted: list[Any] = []
        llm_call, save = await _run_batch(
            [EMPTY_LEGITIMATE, EMPTY_LEGITIMATE], emitted=emitted
        )

        assert llm_call.await_count == 2
        save.assert_not_awaited()
        assert emitted[-1].empty_parse_attempts == 2

    async def test_non_empty_path_is_a_single_call(self, monkeypatch):
        """Regression guard: ordinary batches pay for exactly one call."""
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        emitted: list[Any] = []
        llm_call, save = await _run_batch([NON_EMPTY], emitted=emitted)

        assert llm_call.await_count == 1
        save.assert_awaited_once()
        assert emitted[-1].empty_parse_attempts == 1

    async def test_no_observed_peer_messages_skips_retries(self, monkeypatch):
        """Nothing from the observed peer -> a re-request cannot add anything."""
        monkeypatch.setattr(settings.DERIVER, "EMPTY_PARSE_BACKOFF_SECONDS", 0.0)
        emitted: list[Any] = []
        llm_call, _save = await _run_batch(
            [EMPTY_TRUNCATED],
            messages=[_message(peer_name="carol")],
            emitted=emitted,
        )

        assert llm_call.await_count == 1
        assert emitted[-1].empty_parse_attempts == 1

    async def test_worst_case_call_budget_is_the_documented_product(self):
        """The no-infinite-loop guarantee, asserted as arithmetic.

        In-line attempts per claim times the work-unit requeue budget. Both are
        small constants, so a degraded provider is bounded at 6 provider calls
        per batch with the shipped defaults.
        """
        assert (
            1 + settings.DERIVER.EMPTY_PARSE_MAX_ATTEMPTS
        ) * MAX_RETRYABLE_ATTEMPTS == 2 * MAX_RETRYABLE_ATTEMPTS
        assert MAX_RETRYABLE_ATTEMPTS == 3
