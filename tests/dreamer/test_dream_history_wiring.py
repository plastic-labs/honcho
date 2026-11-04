"""Representation History wiring: orchestrator → specialist → tool handlers."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.crud.dream_history import DreamHistoryTarget
from src.dreamer import orchestrator
from src.dreamer.specialists import SPECIALISTS, DeductionSpecialist, SpecialistResult
from src.llm import HonchoLLMCallResponse
from src.models import Peer, Workspace
from src.utils.agent_tools import (
    ToolContext,
    _handle_create_observations_deductive,  # pyright: ignore[reportPrivateUsage]
    _handle_delete_observations,  # pyright: ignore[reportPrivateUsage]
    _handle_update_peer_card,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(autouse=True)
def _no_metrics(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


async def test_dream_tool_writes_record_to_the_run(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    explicit_ids = [str(generate_nanoid()) for _ in range(2)]
    db_session.add(
        models.Collection(
            workspace_name=workspace_name, observer=peer_name, observed=peer_name
        )
    )
    db_session.add_all(
        models.Document(
            id=doc_id,
            workspace_name=workspace_name,
            observer=peer_name,
            observed=peer_name,
            content=content,
            level="explicit",
            embedding=[0.1] * 1536,
        )
        for doc_id, content in zip(
            explicit_ids, ["Ann drinks coffee", "Ann works nights"], strict=True
        )
    )
    await db_session.commit()
    run = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=peer_name,
        observed=peer_name,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert run is not None
    ctx = ToolContext(
        workspace_name=workspace_name,
        observer=peer_name,
        observed=peer_name,
        session_name=None,
        current_messages=None,
        include_observation_ids=True,
        db_lock=asyncio.Lock(),
        run_id=run.run_id,
        agent_type="deduction",
        parent_category="dream",
        history=DreamHistoryTarget(run_id=run.run_id, specialist="deduction"),
    )

    await _handle_create_observations_deductive(
        ctx,
        {
            "observations": [
                {
                    "content": "Ann relies on caffeine at night",
                    "source_ids": explicit_ids,
                    "premises": ["Ann drinks coffee", "Ann works nights"],
                }
            ]
        },
    )
    await _handle_delete_observations(ctx, {"observation_ids": [explicit_ids[0]]})
    await _handle_update_peer_card(ctx, {"content": ["IDENTITY: Name: Ann"]})

    db_session.expire_all()
    detail = await crud.get_run_detail(db_session, workspace_name, run.run_id)
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine at night"
    assert [source.id for source in created.sources] == explicit_ids
    assert created.specialist == "deduction"
    [removed] = detail.removed
    assert (removed.id, removed.content) == (explicit_ids[0], "Ann drinks coffee")
    assert detail.peer_card is not None
    assert detail.peer_card.after == ["IDENTITY: Name: Ann"]
    assert (detail.created_count, detail.removed_count) == (1, 1)


async def test_specialist_hands_its_name_and_the_run_to_the_executor():
    llm_call = AsyncMock(
        return_value=HonchoLLMCallResponse(
            content="DONE", input_tokens=1, output_tokens=1, finish_reasons=["stop"]
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
        await DeductionSpecialist().run(
            workspace_name="w",
            observer="alice",
            observed="alice",
            session_name=None,
            parent_run_id="run-1",
            history=DreamHistoryTarget(run_id="run-1"),
        )
        await DeductionSpecialist().run(
            workspace_name="w", observer="alice", observed="alice", session_name=None
        )

    with_history, without_history = make_executor.call_args_list
    assert with_history.kwargs["history"] == DreamHistoryTarget(
        run_id="run-1", specialist="deduction"
    )
    assert without_history.kwargs["history"] is None


def _result(name: str, *, success: bool) -> SpecialistResult:
    return SpecialistResult(
        run_id="r",
        specialist_type=name,
        iterations=1,
        tool_calls_count=0,
        input_tokens=1,
        output_tokens=1,
        duration_ms=1.0,
        success=success,
        content="done",
    )


@pytest.mark.parametrize(
    ("induction_succeeds", "status"), [(True, "completed"), (False, "failed")]
)
async def test_run_dream_opens_and_closes_its_run(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
    induction_succeeds: bool,
    status: str,
):
    workspace, peer = sample_data
    workspace_name, peer_name = workspace.name, peer.name
    monkeypatch.setattr(settings.DREAM.SURPRISAL, "ENABLED", False)
    seen: dict[str, Any] = {}

    def _specialist(name: str, *, success: bool) -> Any:
        async def run(**kwargs: Any) -> SpecialistResult:
            seen[name] = kwargs["history"]
            return _result(name, success=success)

        return type("Stub", (), {"run": staticmethod(run)})()

    with patch.dict(
        SPECIALISTS,
        {
            "deduction": _specialist("deduction", success=True),
            "induction": _specialist("induction", success=induction_succeeds),
        },
    ):
        result = await orchestrator.run_dream(
            workspace_name,
            peer_name,
            peer_name,
            dream_type="omni",
            trigger_reason="manual",
        )

    assert result is not None
    assert (
        seen["deduction"]
        == seen["induction"]
        == DreamHistoryTarget(run_id=result.run_id)
    )
    run = await db_session.scalar(
        select(models.DreamRun).where(models.DreamRun.id == result.run_id)
    )
    assert run is not None
    assert (run.status, run.dream_type, run.trigger_reason) == (
        status,
        "omni",
        "manual",
    )
    assert run.finished_at is not None


async def test_run_dream_without_history_passes_none(
    sample_data: tuple[Workspace, Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    workspace, peer = sample_data
    monkeypatch.setattr(settings.DREAM.SURPRISAL, "ENABLED", False)
    monkeypatch.setattr(settings.DREAM, "HISTORY_ENABLED", False)
    seen: list[Any] = []

    async def run(**kwargs: Any) -> SpecialistResult:
        seen.append(kwargs["history"])
        return _result("x", success=True)

    stub = type("Stub", (), {"run": staticmethod(run)})()
    with patch.dict(SPECIALISTS, {"deduction": stub, "induction": stub}):
        await orchestrator.run_dream(workspace.name, peer.name, peer.name)

    assert seen == [None, None]
