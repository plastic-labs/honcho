from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.crud.document import DocumentLevelSummary
from src.dreamer.specialists import (
    BaseSpecialist,
    CardRefreshSpecialist,
    DeductionSpecialist,
    InductionSpecialist,
)
from src.llm import HonchoLLMCallResponse

_SUMMARY = DocumentLevelSummary(
    counts_by_level={"explicit": 7}, earliest=None, latest=None
)


async def _run(specialist: BaseSpecialist, **run_kwargs: Any) -> list[dict[str, Any]]:
    """Run the specialist with the LLM and DB mocked; return the messages sent."""
    llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="DONE", input_tokens=1, output_tokens=1, finish_reasons=["stop"]
        )
    )
    with (
        patch("src.dreamer.specialists.tracked_db") as tracked_db,
        patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
        patch("src.dreamer.specialists.crud.get_peer_card", new=AsyncMock()),
        patch(
            "src.dreamer.specialists.crud.summarize_documents_by_level",
            new=AsyncMock(return_value=_SUMMARY),
        ),
        patch(
            "src.dreamer.specialists.create_tool_executor",
            new=AsyncMock(return_value=AsyncMock()),
        ),
        patch("src.dreamer.specialists.honcho_llm_call", new=llm_call),
    ):
        tracked_db.return_value.__aenter__.return_value = AsyncMock()
        await specialist.run(
            workspace_name="w",
            observer="alice",
            observed="alice",
            session_name=None,
            **run_kwargs,
        )
    return llm_call.call_args.kwargs["messages"]


@pytest.fixture(autouse=True)
def _no_metrics(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("specialist", [DeductionSpecialist(), InductionSpecialist()])
async def test_brief_adds_layout_and_counts(
    specialist: BaseSpecialist, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    system, user = await _run(specialist)
    assert "## HOW MEMORY IS ORGANIZED" in system["content"]
    assert "`get_recent_observations`" in system["content"]
    assert user["content"].startswith("Memory counts: 7 explicit,")


@pytest.mark.asyncio
async def test_brief_off_by_default():
    system, user = await _run(DeductionSpecialist())
    assert "HOW MEMORY IS ORGANIZED" not in system["content"]
    assert "Memory counts" not in user["content"]


@pytest.mark.asyncio
async def test_card_refresh_never_gets_the_brief(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    system, user = await _run(CardRefreshSpecialist())
    assert "HOW MEMORY IS ORGANIZED" not in system["content"]
    assert "Memory counts" not in user["content"]


@pytest.mark.asyncio
async def test_specialist_event_reports_its_iteration_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "DEDUCTION_MAX_TOOL_ITERATIONS", 9)
    with patch("src.dreamer.specialists.emit") as emit:
        await _run(DeductionSpecialist())
    event = emit.call_args.args[0]
    assert event.iteration_limit == 9
    assert event.hit_iteration_limit is False
    assert event.budget_line_enabled is False


@pytest.mark.asyncio
async def test_brief_includes_the_work_list(monkeypatch: pytest.MonkeyPatch):
    import datetime

    from src.dreamer.working_memory import DreamState, NotesCursor, WorkItem, WorkList

    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    work_list = WorkList(
        items=[
            WorkItem(
                id="n1",
                content="moved to Lisbon",
                created_at=datetime.datetime(2026, 10, 1, tzinfo=datetime.UTC),
            )
        ],
        remaining=0,
    )
    state = DreamState(cursor=NotesCursor(), work_list=work_list)
    with patch("src.dreamer.specialists.emit") as emit:
        _, user = await _run(InductionSpecialist(), state=state)
    assert "## NEW SINCE THE LAST DREAM" in user["content"]
    # The list replaces "start with get_recent_observations", which would
    # re-fetch it.
    assert (
        "Start from the conclusions under NEW SINCE THE LAST DREAM" in user["content"]
    )
    assert "Start with `get_recent_observations`" not in user["content"]
    assert "[id:n1] [2026-10-01] moved to Lisbon" in user["content"]
    assert emit.call_args.args[0].new_conclusions_shown == 1


@pytest.mark.parametrize("specialist", [DeductionSpecialist(), InductionSpecialist()])
@pytest.mark.parametrize("hints", [None, ["coffee"]])
def test_start_instruction_follows_the_work_list(
    specialist: BaseSpecialist, hints: list[str] | None
):
    plain = specialist.build_user_prompt("alice", hints)
    listed = specialist.build_user_prompt("alice", hints, work_list_shown=True)
    assert "Start from the conclusions under NEW SINCE THE LAST DREAM" in listed
    assert "get_recent_observations" not in listed
    assert "NEW SINCE THE LAST DREAM" not in plain
