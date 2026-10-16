"""Budget line and memory brief for the dialectic."""

import datetime
import time
from types import SimpleNamespace
from typing import final
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ReasoningLevel, settings
from src.dialectic.core import DialecticAgent, dialectic_budget_line
from src.dialectic.workspace import WorkspaceDialecticAgent
from src.dreamer.working_memory import NoteEntry
from src.llm import HonchoLLMCallResponse


def _agent(
    level: ReasoningLevel = "low", session_allowlist: list[str] | None = None
) -> DialecticAgent:
    return DialecticAgent(
        workspace_name="w",
        session_name=None,
        observer="alice",
        observed="bob",
        reasoning_level=level,
        session_allowlist=session_allowlist,
    )


def test_budget_line_wording():
    assert dialectic_budget_line(1, 4) == "Tool calls used: 1 of 4."
    assert dialectic_budget_line(3, 4) == (
        "Tool calls used: 3 of 4. On the next call, answer from what you have."
    )
    assert dialectic_budget_line(4, 4) is None


@pytest.mark.parametrize(
    ("iterations", "shown"), [(1, False), (2, False), (3, True), (10, True)]
)
def test_budget_line_only_with_three_or_more_iterations(
    iterations: int, shown: bool, monkeypatch: pytest.MonkeyPatch
):
    # Set the count explicitly: deployments (and CI) override per-level limits.
    monkeypatch.setattr(settings.DIALECTIC, "BUDGET_ENABLED", True)
    level_settings = settings.DIALECTIC.LEVELS["low"].model_copy(
        update={"MAX_TOOL_ITERATIONS": iterations}
    )
    assert (_agent()._budget_line(level_settings) is not None) is shown  # pyright: ignore[reportPrivateUsage]


def test_budget_line_off_by_default():
    level_settings = settings.DIALECTIC.LEVELS["max"]
    assert _agent("max")._budget_line(level_settings) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_answer_passes_the_budget_line(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DIALECTIC, "BUDGET_ENABLED", True)
    llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="answer", input_tokens=1, output_tokens=1, finish_reasons=["stop"]
        )
    )
    with (
        patch.object(
            DialecticAgent,
            "_prepare_query",
            new=AsyncMock(
                return_value=(AsyncMock(), "task", "run", time.perf_counter(), None)
            ),
        ),
        patch.object(DialecticAgent, "_log_response_metrics") as log_metrics,
        patch("src.dialectic.core.honcho_llm_call", new=llm_call),
    ):
        await _agent("high").answer("q")
    assert llm_call.call_args.kwargs["budget_line"] is dialectic_budget_line
    assert log_metrics.call_args.kwargs["budget_line_enabled"] is True


@final
class _Stream:
    """Stands in for the streamed response: two chunks, then metadata."""

    tool_calls_made: list[dict[str, str]] = []
    input_tokens = 1
    output_tokens = 1
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    thinking_content = None
    iterations = 4
    hit_input_token_cap = False
    hit_iteration_limit = True

    async def __aiter__(self):
        for text in ("ans", "wer"):
            yield SimpleNamespace(content=text)


@pytest.mark.asyncio
async def test_answer_stream_passes_the_budget_line(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DIALECTIC, "BUDGET_ENABLED", True)
    level_settings = settings.DIALECTIC.LEVELS["high"].model_copy(
        update={"MAX_TOOL_ITERATIONS": 4}
    )
    monkeypatch.setitem(settings.DIALECTIC.LEVELS, "high", level_settings)
    llm_call = AsyncMock(return_value=_Stream())
    with (
        patch.object(
            DialecticAgent,
            "_prepare_query",
            new=AsyncMock(
                return_value=(AsyncMock(), "task", "run", time.perf_counter(), None)
            ),
        ),
        patch.object(DialecticAgent, "_log_response_metrics") as log_metrics,
        patch("src.dialectic.core.honcho_llm_call", new=llm_call),
    ):
        chunks = [chunk async for chunk in _agent("high").answer_stream("q")]
    assert chunks == ["ans", "wer"]
    kwargs = llm_call.call_args.kwargs
    assert kwargs["budget_line"] is dialectic_budget_line
    assert kwargs["max_tool_iterations"] == 4
    metrics = log_metrics.call_args.kwargs
    assert metrics["hit_iteration_limit"] is True
    assert metrics["iteration_limit"] == 4
    assert metrics["budget_line_enabled"] is True


def test_layout_section_follows_the_flag(monkeypatch: pytest.MonkeyPatch):
    assert "HOW MEMORY IS ORGANIZED" not in _agent().messages[0]["content"]
    monkeypatch.setattr(settings.DIALECTIC, "BRIEF_ENABLED", True)
    system = _agent().messages[0]["content"]
    assert "## HOW MEMORY IS ORGANIZED" in system
    # The dialectic doesn't cite ids, so the id instructions are left out.
    assert "To cite one" not in system


def test_workspace_agent_never_gets_the_pair_brief(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DIALECTIC, "BRIEF_ENABLED", True)
    agent = WorkspaceDialecticAgent(workspace_name="w")
    assert "HOW MEMORY IS ORGANIZED" not in agent.messages[0]["content"]
    assert agent._pair_brief_enabled() is False  # pyright: ignore[reportPrivateUsage]


def _map_entry(text: str) -> NoteEntry:
    return NoteEntry(
        id="n1",
        kind="map",
        specialist="deduction",
        text=text,
        created_dream=1,
        updated_at=datetime.datetime(2026, 10, 1, tzinfo=datetime.UTC),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("notes_mode", ["on", "shadow"])
async def test_pair_brief_has_counts_and_map_notes_only_when_on(
    notes_mode: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.DIALECTIC, "BRIEF_ENABLED", True)
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", notes_mode)
    with (
        patch(
            "src.dialectic.core.load_memory_counts",
            new=AsyncMock(return_value="Memory counts: 3 explicit."),
        ),
        patch(
            "src.dialectic.core.load_map_entries",
            new=AsyncMock(return_value=[_map_entry("Mostly about the sales pipeline")]),
        ),
    ):
        brief = await _agent()._pair_brief()  # pyright: ignore[reportPrivateUsage]
    assert brief is not None and brief.startswith("Memory counts: 3 explicit.")
    assert ("- Mostly about the sales pipeline" in brief) is (notes_mode == "on")


@pytest.mark.asyncio
async def test_session_allowlist_rules_out_the_pair_brief(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DIALECTIC, "BRIEF_ENABLED", True)
    counts = AsyncMock(return_value="Memory counts: 3 explicit.")
    with patch("src.dialectic.core.load_memory_counts", new=counts):
        brief = await _agent(session_allowlist=["s1"])._pair_brief()  # pyright: ignore[reportPrivateUsage]
    assert brief is None
    counts.assert_not_awaited()
