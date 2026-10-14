from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings
from src.dreamer.orchestrator import DreamResult, process_dream
from src.schemas import DreamType
from src.utils.queue_payload import DreamPayload


def _result(deduction: bool, induction: bool) -> DreamResult:
    return DreamResult(
        run_id="r1",
        specialists_run=["deduction", "induction"],
        deduction_success=deduction,
        induction_success=induction,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=0,
        total_duration_ms=0.0,
        input_tokens=0,
        output_tokens=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deduction", "induction", "advanced"),
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
async def test_cursor_advances_only_when_a_specialist_succeeded(
    deduction: bool, induction: bool, advanced: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    advance = AsyncMock()
    collection = MagicMock(internal_metadata={})
    db = AsyncMock()
    db.scalar.return_value = 0
    with (
        patch(
            "src.dreamer.orchestrator.run_dream",
            new=AsyncMock(return_value=_result(deduction, induction)),
        ),
        patch("src.dreamer.orchestrator.tracked_db") as tracked_db,
        patch(
            "src.dreamer.orchestrator.crud.get_collection",
            new=AsyncMock(return_value=collection),
        ),
        patch(
            "src.dreamer.orchestrator.crud.update_collection_internal_metadata",
            new=AsyncMock(),
        ),
        patch("src.dreamer.orchestrator.advance_cursor", new=advance),
    ):
        tracked_db.return_value.__aenter__.return_value = db
        await process_dream(
            DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="a"),
            "w",
        )
    assert advance.await_count == (1 if advanced else 0)
