from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dreamer.specialists import (
    DeductionSpecialist,
    InductionSpecialist,
    _handler_resolver,  # pyright: ignore[reportPrivateUsage]
    dream_budget_line,
    write_window_size,
)
from src.llm import HonchoLLMCallResponse
from src.utils.agent_tools import get_tool_handler
from src.utils.types import ToolResult, set_current_iteration


@pytest.mark.parametrize(
    ("limit", "window"), [(12, 2), (10, 2), (5, 1), (1, 1), (20, 3)]
)
def test_default_window_is_15_percent_rounded_min_one(limit: int, window: int):
    assert write_window_size(limit) == window


def test_window_override_leaves_a_search_round(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "WRITE_WINDOW_ITERATIONS", 4)
    assert write_window_size(12) == 4
    assert write_window_size(3) == 2
    assert write_window_size(1) == 1


def test_budget_line_wording():
    line = dream_budget_line(2)
    assert line(7, 12) == (
        "Tool calls used: 7 of 12. Calls 11 and 12 are for writing; "
        + "searches will be refused."
    )
    assert "Searches are now refused" in (line(10, 12) or "")
    assert line(12, 12) is None
    assert dream_budget_line(1)(3, 10) == (
        "Tool calls used: 3 of 10. Call 10 is for writing; searches will be refused."
    )
    assert "Calls 8 to 10 are" in (dream_budget_line(3)(2, 10) or "")


@pytest.mark.asyncio
async def test_resolver_refuses_searches_only_inside_the_window():
    resolve = _handler_resolver(11, {})

    set_current_iteration(10)
    assert resolve("search_memory") is get_tool_handler("search_memory")

    set_current_iteration(11)
    refuse = resolve("search_memory")
    assert refuse is not None and refuse is not get_tool_handler("search_memory")
    result = await refuse(AsyncMock(), {})
    assert isinstance(result, ToolResult) and result.metadata == {"refused": True}
    # Writes are never refused.
    assert resolve("create_observations_deductive") is get_tool_handler(
        "create_observations_deductive"
    )


async def _run(
    specialist: Any, content: str | None = "DONE"
) -> tuple[Any, dict[str, Any]]:
    llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content=content, input_tokens=1, output_tokens=1, finish_reasons=["stop"]
        )
    )
    make_executor = AsyncMock(return_value=AsyncMock())
    with (
        patch("src.dreamer.specialists.tracked_db") as tracked_db,
        patch("src.dreamer.specialists.crud.get_peer", new=AsyncMock()),
        patch("src.dreamer.specialists.crud.get_peer_card", new=AsyncMock()),
        patch("src.dreamer.specialists.create_tool_executor", new=make_executor),
        patch("src.dreamer.specialists.honcho_llm_call", new=llm_call),
    ):
        tracked_db.return_value.__aenter__.return_value = AsyncMock()
        result = await specialist.run(
            workspace_name="w", observer="alice", observed="alice", session_name=None
        )
    return result, {
        **llm_call.call_args.kwargs,
        "handler_resolver": make_executor.call_args.kwargs["handler_resolver"],
    }


@pytest.fixture(autouse=True)
def _no_metrics(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("specialist", "write_tools"),
    [
        (
            DeductionSpecialist(),
            [
                "create_observations_deductive",
                "delete_observations",
                "update_peer_card",
            ],
        ),
        (InductionSpecialist(), ["create_observations_inductive"]),
    ],
)
async def test_budget_enabled_wires_the_window(
    specialist: Any, write_tools: list[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.DREAM, "BUDGET_ENABLED", True)
    _, kwargs = await _run(specialist)
    assert kwargs["budget_line"] is not None
    assert [t["name"] for t in kwargs["final_call_tools"]] == write_tools
    assert kwargs["handler_resolver"] is not None
    assert "## BUDGET" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_budget_off_by_default():
    _, kwargs = await _run(DeductionSpecialist())
    assert kwargs["budget_line"] is None
    assert kwargs["final_call_tools"] is None
    assert kwargs["handler_resolver"] is None
    assert "## BUDGET" not in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_write_only_final_call_reports_empty_content(
    monkeypatch: pytest.MonkeyPatch,
):
    # OpenAI returns content=None when the final call only makes tool calls;
    # run_dream slices the result's content, so it must stay a string.
    monkeypatch.setattr(settings.DREAM, "BUDGET_ENABLED", True)
    result, _ = await _run(DeductionSpecialist(), content=None)
    assert result.success is True
    assert result.content == ""
