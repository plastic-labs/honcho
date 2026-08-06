"""Deriver silent-loss benchmark — before/after quantification for #728.

Measures, against whatever ``src/`` is currently checked out:

- ``silent_loss_rate`` — fraction of "all observer saves fail" batches that are
  silently marked processed (the deriver returns normally with zero documents
  saved) instead of surfacing the failure. This is the #728 data-loss symptom.
- ``failure_visibility_rate`` — fraction of those failing batches that produce an
  *alertable* signal (the call raises so the queue marks it errored, or telemetry
  reports a failed observer).

It needs **no database and no real provider**: the deriver's LLM call and the
representation save are mocked. The same script runs unmodified against any
checkout, so the before/after contrast is produced by running it on ``main``
(before) and on the fix branch (after):

    PYTHONPATH=. uv run python tests/bench/bench_deriver_silent_loss.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

# Make `src` importable when run directly (uv run python tests/bench/<this>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TRIALS = 20


def _message() -> Mock:
    return Mock(
        id=1,
        public_id="msg_1",
        session_name="session-1",
        workspace_name="workspace-1",
        peer_name="alice",
        content="hello",
        token_count=5,
        created_at=datetime.now(timezone.utc),
    )


async def measure_silent_loss_and_visibility(
    trials: int = TRIALS,
) -> tuple[float, float]:
    """Drive process_representation_tasks_batch with every observer save failing.

    Returns (silent_loss_rate, failure_visibility_rate).
    """
    from src.crud.representation import RepresentationManager
    from src.deriver.deriver import process_representation_tasks_batch
    from src.llm import HonchoLLMCallResponse
    from src.utils.representation import ExplicitObservationBase, PromptRepresentation

    # RepresentationSaveError only exists on the fix branch; pre-fix `main`
    # swallows the failure and never raises, so the name is absent there. Fall
    # back to the base HonchoException so this script runs unmodified on either
    # checkout — on `main` the except below simply never fires.
    try:
        from src.exceptions import RepresentationSaveError as _SaveFailure
    except ImportError:  # pragma: no cover - pre-fix main
        from src.exceptions import HonchoException as _SaveFailure

    silent = 0
    alertable = 0
    for i in range(trials):
        configuration = Mock()
        configuration.reasoning.enabled = True
        response = HonchoLLMCallResponse(
            content=PromptRepresentation(
                explicit=[ExplicitObservationBase(content=f"observation {i}")]
            ),
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )
        emitted: list[Any] = []
        raised = False
        with (
            patch(
                "src.deriver.deriver.honcho_llm_call",
                new=AsyncMock(return_value=response),
            ),
            patch.object(
                RepresentationManager,
                "save_representation",
                new=AsyncMock(side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")),
            ),
            patch("src.deriver.deriver.emit", side_effect=emitted.append),
        ):
            try:
                await process_representation_tasks_batch(
                    messages=[_message()],
                    message_level_configuration=configuration,
                    observers=["bob"],
                    observed="alice",
                    queue_item_message_ids=[1],
                )
            # Count only the expected fail-loud save failure. Keying off the
            # typed exception (not the message text) means an unrelated
            # regression raises a different type, escapes, and fails the
            # benchmark instead of inflating the visibility rate.
            except _SaveFailure:
                raised = True

        # The save failed for the only observer. "Silent loss" = the deriver
        # returned normally (queue would mark it processed) with nothing saved.
        if not raised:
            silent += 1
        # Alertable = the failure is detectable downstream: either it raised
        # (queue marks errored), or telemetry carries a non-zero failed count.
        if raised or any(getattr(e, "failed_observer_count", 0) for e in emitted):
            alertable += 1

    return silent / trials, alertable / trials


def _supports_fail_loud() -> bool:
    from src.telemetry.events.representation import RepresentationCompletedEvent

    return "failed_observer_count" in RepresentationCompletedEvent.model_fields


async def main() -> None:
    silent_loss, visibility = await measure_silent_loss_and_visibility()
    mode = "AFTER (fail-loud)" if _supports_fail_loud() else "BEFORE (swallowed)"

    print("=" * 60)
    print(f"Deriver silent-loss benchmark — {mode}")
    print(f"  trials={TRIALS}")
    print("-" * 60)
    print(f"  silent_loss_rate        : {silent_loss:6.1%}   (target: 0%)")
    print(f"  failure_visibility_rate : {visibility:6.1%}   (target: 100%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
