"""What the dream specialists are told about the peer card.

Only deduction writes the card. A peer with no card yet gets a prompt that
says so, so deduction creates one rather than skipping it, and deduction is
told to carry a knowledge update over to the card.
"""

import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.crud.document import DocumentLevelSummary
from src.dreamer.specialists import (
    BaseSpecialist,
    DeductionSpecialist,
    InductionSpecialist,
)
from src.dreamer.working_memory import DreamState, NotesCursor, WorkItem, WorkList
from src.llm import HonchoLLMCallResponse

NO_CARD_YET = "This peer has no peer card yet"
CARD_UPDATE = "If the old value is on the peer card"


async def _user_prompt(
    specialist: BaseSpecialist, stored_card: list[str] | None, **run_kwargs: Any
) -> tuple[str, AsyncMock]:
    """Run the specialist with the LLM and database mocked; return its user
    prompt and the get_peer_card mock."""
    response = HonchoLLMCallResponse(
        content="DONE", input_tokens=10, output_tokens=5, finish_reasons=["stop"]
    )
    get_peer_card = AsyncMock(return_value=stored_card)
    llm_call = AsyncMock(return_value=response)
    with (
        patch("src.dreamer.specialists.tracked_db") as tracked_db,
        patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
        patch("src.dreamer.specialists.crud.get_peer_card", new=get_peer_card),
        patch(
            "src.dreamer.specialists.crud.summarize_documents_by_level",
            new=AsyncMock(
                return_value=DocumentLevelSummary(
                    counts_by_level={"explicit": 7}, earliest=None, latest=None
                )
            ),
        ),
        patch(
            "src.dreamer.specialists.load_notes_for_pass",
            new=AsyncMock(return_value=None),
        ),
        patch("src.dreamer.specialists.emit"),
        patch(
            "src.dreamer.specialists.create_tool_executor",
            new=AsyncMock(return_value=AsyncMock()),
        ),
        patch("src.dreamer.specialists.honcho_llm_call", new=llm_call),
    ):
        tracked_db.return_value.__aenter__.return_value = AsyncMock()
        result = await specialist.run(
            workspace_name="workspace",
            observer="assistant",
            observed="maya",
            session_name=None,
            **run_kwargs,
        )
    assert result.success is True
    assert llm_call.await_args is not None
    return llm_call.await_args.kwargs["messages"][1]["content"], get_peer_card


@pytest.fixture(autouse=True)
def no_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


@pytest.mark.asyncio
async def test_deduction_is_told_to_create_a_missing_card() -> None:
    prompt, _ = await _user_prompt(DeductionSpecialist(), None)

    assert "CURRENT PEER CARD" in prompt
    assert NO_CARD_YET in prompt
    assert "update_peer_card" in prompt


@pytest.mark.asyncio
async def test_a_stored_empty_card_reads_as_none_yet() -> None:
    prompt, _ = await _user_prompt(DeductionSpecialist(), [])

    assert NO_CARD_YET in prompt


@pytest.mark.asyncio
async def test_with_working_memory_on_deduction_is_still_told(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Brief, budget and notes on, as deployments run them: the work list
    # replaces the usual opening, and the card note must survive that.
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    monkeypatch.setattr(settings.DREAM, "BUDGET_ENABLED", True)
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", "on")
    state = DreamState(
        cursor=NotesCursor(),
        work_list=WorkList(
            items=[
                WorkItem(
                    id="n1",
                    content="maya moved to Seattle",
                    created_at=datetime.datetime(2026, 11, 1, tzinfo=datetime.UTC),
                )
            ],
            remaining=0,
        ),
    )

    prompt, _ = await _user_prompt(DeductionSpecialist(), None, state=state)

    assert "## NEW SINCE THE LAST DREAM" in prompt
    assert NO_CARD_YET in prompt


@pytest.mark.asyncio
async def test_deduction_sees_an_existing_card() -> None:
    prompt, _ = await _user_prompt(
        DeductionSpecialist(), ["IDENTITY: Name: Zorblax-Prime"]
    )

    assert "IDENTITY: Name: Zorblax-Prime" in prompt
    assert NO_CARD_YET not in prompt


@pytest.mark.asyncio
async def test_induction_never_sees_the_card() -> None:
    prompt, get_peer_card = await _user_prompt(InductionSpecialist(), None)

    get_peer_card.assert_not_awaited()
    assert "CURRENT PEER CARD" not in prompt


def test_deduction_carries_knowledge_updates_to_the_card() -> None:
    specialist = DeductionSpecialist()

    assert CARD_UPDATE in specialist.build_system_prompt("maya")
    assert CARD_UPDATE not in specialist.build_system_prompt(
        "maya", peer_card_enabled=False
    )


def test_a_card_that_is_not_shown_says_nothing() -> None:
    # `None` is "not shown" (rebuild mode, or a specialist that can't write
    # the card), unlike an empty card.
    prompt = DeductionSpecialist().build_user_prompt("maya", hints=None, peer_card=None)

    assert "CURRENT PEER CARD" not in prompt
