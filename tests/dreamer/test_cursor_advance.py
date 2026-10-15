from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings
from src.dreamer.orchestrator import DreamResult, process_dream
from src.dreamer.working_memory import (
    NOTES_KEY,
    DreamState,
    NotesCursor,
    WorkItem,
    WorkList,
)
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
    ("deduction", "induction", "recorded", "moves_position"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, True, False),
        (False, False, False, False),
    ],
)
async def test_cursor_moves_only_when_both_specialists_succeeded(
    deduction: bool,
    induction: bool,
    recorded: bool,
    moves_position: bool,
    monkeypatch: pytest.MonkeyPatch,
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
    assert advance.await_count == (1 if recorded else 0)
    if recorded:
        assert advance.await_args is not None
        kwargs = advance.await_args.kwargs
        assert kwargs["all_specialists_succeeded"] is moves_position
        assert kwargs["any_specialist_succeeded"] is True


async def _run_process_dream(
    result: DreamResult, internal_metadata: dict[str, Any], explicit_count: int
) -> tuple[AsyncMock, AsyncMock]:
    advance = AsyncMock()
    update = AsyncMock()
    db = AsyncMock()
    db.scalar.return_value = explicit_count
    with (
        patch("src.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=result)),
        patch("src.dreamer.orchestrator.tracked_db") as tracked_db,
        patch(
            "src.dreamer.orchestrator.crud.get_collection",
            new=AsyncMock(return_value=MagicMock(internal_metadata=internal_metadata)),
        ),
        patch(
            "src.dreamer.orchestrator.crud.update_collection_internal_metadata",
            new=update,
        ),
        patch("src.dreamer.orchestrator.advance_cursor", new=advance),
    ):
        tracked_db.return_value.__aenter__.return_value = db
        await process_dream(
            DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="a"),
            "w",
        )
    return advance, update


@pytest.mark.asyncio
async def test_working_memory_off_clears_a_stored_position(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", False)
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", "off")
    stored = {
        "entries": [{"id": "n1"}],
        "cursor": {
            "through_created_at": "2026-10-01T00:00:00+00:00",
            "through_id": "x",
            "dreams": 2,
        },
    }
    advance, update = await _run_process_dream(
        _result(True, True), {NOTES_KEY: stored}, explicit_count=4
    )
    assert advance.await_count == 0
    assert update.await_args is not None
    written = update.await_args.kwargs["update_data"]
    assert written[NOTES_KEY]["entries"] == [{"id": "n1"}]
    assert written[NOTES_KEY]["cursor"]["through_id"] is None
    assert written[NOTES_KEY]["cursor"]["dreams"] == 2


@pytest.mark.asyncio
async def test_unshown_backlog_stays_counted_as_new(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "BRIEF_ENABLED", True)
    result = _result(True, True)
    result.state = DreamState(
        cursor=NotesCursor(),
        work_list=WorkList(
            items=[WorkItem(id="x", content="c", created_at=datetime.now(UTC))],
            remaining=30,
        ),
    )
    _, update = await _run_process_dream(result, {}, explicit_count=100)
    assert update.await_args is not None
    written = update.await_args.kwargs["update_data"]
    assert written["dream"]["last_dream_document_count"] == 70
    assert NOTES_KEY not in written
