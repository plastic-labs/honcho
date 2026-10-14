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


async def _run(specialist: BaseSpecialist) -> list[dict[str, Any]]:
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
            workspace_name="w", observer="alice", observed="alice", session_name=None
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
