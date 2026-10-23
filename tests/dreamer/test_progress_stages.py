"""Progress callbacks follow existing orchestration without changing outcomes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import models
from src.config import settings
from src.dreamer.orchestrator import run_card_refresh_dream, run_dream
from src.dreamer.specialists import SpecialistResult


def result(stage: str) -> SpecialistResult:
    return SpecialistResult(
        run_id=stage,
        specialist_type=stage,
        iterations=0,
        tool_calls_count=0,
        input_tokens=0,
        output_tokens=0,
        duration_ms=0,
        success=True,
        content="test",
    )


@pytest.mark.asyncio
async def test_omni_reports_deduction_then_induction_even_after_specialist_error(
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, peer = sample_data
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    monkeypatch.setattr(settings.DREAM.SURPRISAL, "ENABLED", False)
    report = AsyncMock()
    with patch.dict(
        "src.dreamer.orchestrator.SPECIALISTS",
        {
            "deduction": MagicMock(
                run=AsyncMock(side_effect=RuntimeError("test failure"))
            ),
            "induction": MagicMock(run=AsyncMock(return_value=result("induction"))),
        },
    ):
        outcome = await run_dream(workspace.name, peer.name, peer.name, progress=report)
    assert [call.args[0] for call in report.await_args_list] == [
        "deduction",
        "induction",
    ]
    assert outcome is not None
    assert outcome.deduction_success is False
    assert outcome.induction_success is True


@pytest.mark.asyncio
async def test_card_refresh_reports_its_actual_specialist(
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, peer = sample_data
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    report = AsyncMock()
    with patch("src.dreamer.orchestrator.CardRefreshSpecialist") as specialist:
        specialist.return_value.run = AsyncMock(return_value=result("card_refresh"))
        outcome = await run_card_refresh_dream(
            workspace.name, peer.name, peer.name, progress=report
        )
    report.assert_awaited_once_with("card_refresh")
    assert outcome is not None and outcome.deduction_success
