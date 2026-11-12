"""A dream's writes run behind the lifecycle barrier.

A dream reads a representation and then writes many times across many model
calls: new conclusions, removals, the peer card, its notes. All of them are
checked against one snapshot captured when the dream started. Each test lets a
deletion land mid-dream, after the snapshot and before a write.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.crud.lifecycle_barrier import WriteGate, capture_write_snapshot
from src.dreamer import orchestrator
from src.dreamer.specialists import SPECIALISTS, DeductionSpecialist
from src.dreamer.working_memory import (
    NOTES_KEY,
    NoteEntry,
    _store_entries,  # pyright: ignore[reportPrivateUsage]
)
from src.exceptions import (
    LifecycleBarrierError,
    WriteGenerationStaleError,
    WriteParentGoneError,
)
from src.llm import HonchoLLMCallResponse
from src.schemas import DreamType
from src.utils.agent_tools import create_tool_executor
from src.utils.queue_payload import DreamPayload
from tests.lifecycle_barrier_graph import Graph, seed_graph, tombstone

ToolExecutor = Callable[[str, dict[str, Any]], Any]

DEDUCTION = "the observed peer relies on caffeine at night"
CARD = ["IDENTITY: Name: Ann"]


@pytest.fixture(autouse=True)
def _no_metrics(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


async def _seed_explicit(db: AsyncSession, graph: Graph) -> list[str]:
    """Two explicit conclusions for the dream to read and build on."""
    ids = [str(generate_nanoid()) for _ in range(2)]
    db.add_all(
        models.Document(
            id=doc_id,
            workspace_name=graph.workspace.name,
            observer=graph.observer.name,
            observed=graph.observed.name,
            session_name=graph.session.name,
            content=content,
            level="explicit",
            embedding=[0.1] * 1536,
        )
        for doc_id, content in zip(
            ids, ["Ann drinks coffee", "Ann works nights"], strict=True
        )
    )
    await db.commit()
    return ids


async def _start_dream(
    db: AsyncSession, graph: Graph
) -> tuple[WriteGate, ToolExecutor]:
    """What a dream holds once it started: its snapshot and its tools."""
    gate = WriteGate(
        await capture_write_snapshot(
            db,
            graph.workspace.name,
            representations=[(graph.observer.name, graph.observed.name)],
        )
    )
    await db.commit()
    executor = await create_tool_executor(
        workspace_name=graph.workspace.name,
        observer=graph.observer.name,
        observed=graph.observed.name,
        include_observation_ids=True,
        barrier=gate,
    )
    return gate, executor


def _create_call(source_ids: list[str]) -> tuple[str, dict[str, Any]]:
    return (
        "create_observations_deductive",
        {
            "observations": [
                {
                    "content": DEDUCTION,
                    "source_ids": source_ids,
                    "premises": ["Ann drinks coffee", "Ann works nights"],
                }
            ]
        },
    )


async def _live_conclusions(db: AsyncSession, graph: Graph) -> set[str]:
    return set(
        await db.scalars(
            select(models.Document.content).where(
                models.Document.workspace_name == graph.workspace.name,
                models.Document.deleted_at.is_(None),
            )
        )
    )


async def _card(db: AsyncSession, graph: Graph) -> Any:
    metadata = await db.scalar(
        select(models.Peer.internal_metadata).where(
            models.Peer.workspace_name == graph.workspace.name,
            models.Peer.name == graph.observer.name,
        )
    )
    return None if metadata is None else metadata.get("observed_peer_card")


async def test_an_undisturbed_dream_writes(db_session: AsyncSession) -> None:
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate, call = await _start_dream(db_session, graph)

    await call(*_create_call(explicit_ids))
    await call("update_peer_card", {"content": CARD})
    await call("delete_observations", {"observation_ids": [explicit_ids[0]]})

    assert gate.refusal is None
    assert await _live_conclusions(db_session, graph) == {DEDUCTION, "Ann works nights"}
    assert await _card(db_session, graph) == CARD


@pytest.mark.parametrize("write", ["create", "remove", "card"])
async def test_a_conclusion_erased_mid_dream_refuses_the_dreams_writes(
    db_session: AsyncSession, write: str
) -> None:
    """The dream read a conclusion that a deletion then erased."""
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate, call = await _start_dream(db_session, graph)

    await crud.delete_document_by_id(db_session, graph.workspace.name, explicit_ids[0])

    if write == "create":
        result = await call(*_create_call(explicit_ids))
    elif write == "remove":
        result = await call(
            "delete_observations", {"observation_ids": [explicit_ids[1]]}
        )
    else:
        result = await call("update_peer_card", {"content": CARD})

    assert str(result).startswith("ERROR:")
    assert isinstance(gate.refusal, WriteGenerationStaleError)
    assert gate.refusal.scope == "collection"
    assert await _live_conclusions(db_session, graph) == {"Ann works nights"}
    assert await _card(db_session, graph) is None


async def test_after_one_refusal_the_dream_reads_and_writes_nothing_more(
    db_session: AsyncSession,
) -> None:
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate, call = await _start_dream(db_session, graph)
    await crud.delete_document_by_id(db_session, graph.workspace.name, explicit_ids[0])
    refused = await call("update_peer_card", {"content": CARD})
    first_refusal = gate.refusal

    # A read is refused too, and a later write does not reach the database.
    with patch("src.utils.agent_tools.tracked_db") as tracked_db:
        assert await call("get_recent_observations", {}) == refused
        assert await call(*_create_call(explicit_ids)) == refused
    tracked_db.assert_not_called()
    assert gate.refusal is first_refusal


@pytest.mark.parametrize("kind", ["workspace", "observer", "observed"])
async def test_a_parent_deleted_mid_dream_gets_no_conclusions_and_no_card(
    db_session: AsyncSession, kind: str
) -> None:
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate, call = await _start_dream(db_session, graph)

    await tombstone(db_session, graph, kind)

    # Conclusions under a deleted parent are hidden from reads, so the sources
    # no longer resolve and the create writes nothing.
    await call(*_create_call(explicit_ids))
    assert DEDUCTION not in await _live_conclusions(db_session, graph)

    # The card names no sources. Only the barrier stands in its way.
    await call("update_peer_card", {"content": CARD})
    assert isinstance(gate.refusal, WriteParentGoneError)
    assert await _card(db_session, graph) is None


async def test_a_card_write_does_not_recreate_a_purged_observer(
    db_session: AsyncSession,
) -> None:
    """The card write used to get-or-create its observer."""
    graph = await seed_graph(db_session)
    gate, call = await _start_dream(db_session, graph)

    for model in (models.Collection, models.Peer):
        column = model.observer if model is models.Collection else model.name
        await db_session.execute(
            delete(model).where(
                model.workspace_name == graph.workspace.name,
                column == graph.observer.name,
            )
        )
    await db_session.commit()

    await call("update_peer_card", {"content": CARD})

    assert isinstance(gate.refusal, WriteParentGoneError)
    assert gate.refusal.scope == "peer"
    assert (
        await db_session.scalar(
            select(models.Peer.id).where(
                models.Peer.workspace_name == graph.workspace.name,
                models.Peer.name == graph.observer.name,
            )
        )
        is None
    )


async def test_notes_are_not_written_from_a_stale_reading(
    db_session: AsyncSession,
) -> None:
    """Notes are free text a dream writes from what it read."""
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate, _ = await _start_dream(db_session, graph)
    entry = NoteEntry(
        id="n1",
        kind="open",
        specialist="deduction",
        text="Ann drinks coffee, per the erased conclusion",
        created_dream=1,
        updated_at=datetime.now(UTC),
    )
    pair = (graph.workspace.name, graph.observer.name, graph.observed.name)

    assert await _store_entries(*pair, [], epoch=None, barrier=gate.snapshot)

    await crud.delete_document_by_id(db_session, graph.workspace.name, explicit_ids[0])
    with pytest.raises(WriteGenerationStaleError):
        await _store_entries(*pair, [entry], epoch=None, barrier=gate.snapshot)

    stored = await crud.get_collection_internal_metadata_value(
        db_session, *pair, NOTES_KEY
    )
    assert stored is not None
    assert stored["entries"] == []


async def test_a_specialist_pass_fails_when_one_of_its_writes_was_refused(
    db_session: AsyncSession,
) -> None:
    """The model sees an error string. The pass itself must not count as done."""
    graph = await seed_graph(db_session)
    explicit_ids = await _seed_explicit(db_session, graph)
    gate = WriteGate(
        await capture_write_snapshot(
            db_session,
            graph.workspace.name,
            representations=[(graph.observer.name, graph.observed.name)],
        )
    )
    await db_session.commit()
    workspace_name = graph.workspace.name

    async def model_loop(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        # The deletion lands while the model is thinking, then it writes.
        await crud.delete_document_by_id(db_session, workspace_name, explicit_ids[0])
        await kwargs["tool_executor"](*_create_call(explicit_ids))
        return HonchoLLMCallResponse(
            content="DONE", input_tokens=1, output_tokens=1, finish_reasons=["stop"]
        )

    with (
        patch("src.dreamer.specialists.honcho_llm_call", model_loop),
        pytest.raises(WriteGenerationStaleError),
    ):
        await DeductionSpecialist().run(
            workspace_name=workspace_name,
            observer=graph.observer.name,
            observed=graph.observed.name,
            session_name=None,
            barrier=gate,
        )

    assert DEDUCTION not in await _live_conclusions(db_session, graph)


async def test_a_refused_deduction_ends_the_dream_before_induction(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    monkeypatch.setattr(settings.DREAM.SURPRISAL, "ENABLED", False)
    graph = await seed_graph(db_session)
    seen: dict[str, Any] = {}

    async def refused_deduction(**kwargs: Any) -> Any:
        seen["gate"] = kwargs["barrier"]
        raise WriteGenerationStaleError("collection", "a deletion landed")

    induction = AsyncMock()
    with (
        patch.object(SPECIALISTS["deduction"], "run", refused_deduction),
        patch.object(SPECIALISTS["induction"], "run", induction),
        patch("src.dreamer.orchestrator.working_memory_enabled", return_value=False),
        pytest.raises(WriteGenerationStaleError),
    ):
        await orchestrator.run_dream(
            workspace_name=graph.workspace.name,
            observer=graph.observer.name,
            observed=graph.observed.name,
        )

    induction.assert_not_called()
    # The dream captured its snapshot before any specialist ran.
    gate = seen["gate"]
    assert isinstance(gate, WriteGate)
    assert [c.uid for c in gate.snapshot.collections] == [graph.collection.id]
    assert {p.uid for p in gate.snapshot.peers} == {
        graph.observer.id,
        graph.observed.id,
    }


async def test_a_dream_for_a_deleted_peer_does_not_start(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    graph = await seed_graph(db_session)
    await tombstone(db_session, graph, "observed")

    deduction = AsyncMock()
    with (
        patch.object(SPECIALISTS["deduction"], "run", deduction),
        pytest.raises(WriteParentGoneError),
    ):
        await orchestrator.run_dream(
            workspace_name=graph.workspace.name,
            observer=graph.observer.name,
            observed=graph.observed.name,
        )
    deduction.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        WriteGenerationStaleError("collection", "a deletion landed"),
        WriteParentGoneError("peer", "the observed peer was deleted"),
    ],
    ids=["stale", "gone"],
)
async def test_process_dream_hands_a_refusal_to_the_queue(
    error: LifecycleBarrierError,
) -> None:
    """Other dream failures are logged and swallowed. A refusal must reach the
    queue, which drops the task or runs it again."""
    payload = DreamPayload(
        dream_type=DreamType.OMNI, observer="observer", observed="observed"
    )
    with (
        patch("src.dreamer.orchestrator.run_dream", AsyncMock(side_effect=error)),
        pytest.raises(type(error)),
    ):
        await orchestrator.process_dream(payload, "workspace")

    with patch(
        "src.dreamer.orchestrator.run_dream",
        AsyncMock(side_effect=RuntimeError("provider down")),
    ):
        await orchestrator.process_dream(payload, "workspace")
