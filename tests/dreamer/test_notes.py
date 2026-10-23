import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dreamer.specialists import DeductionSpecialist
from src.dreamer.working_memory import (
    NOTES_KEY,
    DreamState,
    NoteEntry,
    NotesCursor,
    apply_note_operations,
    entry_expired,
    make_update_notes_handler,
    parse_entries,
)
from src.llm import HonchoLLMCallResponse
from src.utils.types import ToolResult

NOW = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)


def _entry(entry_id: str, kind: str, created_dream: int = 1) -> NoteEntry:
    return NoteEntry.model_validate(
        {
            "id": entry_id,
            "kind": kind,
            "specialist": "deduction",
            "text": f"text {entry_id}",
            "created_dream": created_dream,
            "updated_at": NOW,
        }
    )


def _apply(entries: list[NoteEntry], operations: Any, *, specialist: str = "deduction"):
    return apply_note_operations(
        entries,
        operations,
        specialist=specialist,
        run_id="r9",
        dream_number=4,
        now=NOW,
    )


def test_operations_add_replace_remove_one_entry_each():
    entries = [_entry("n1", "open"), _entry("n2", "map")]
    updated, edit = _apply(
        entries,
        [
            {"op": "add", "kind": "pending", "text": "write X", "ids": ["id:a1", "b2"]},
            {"op": "replace", "entry_id": "n1", "text": "still open"},
            {"op": "remove", "entry_id": "n2"},
        ],
    )
    assert [e.id for e in updated] == ["n1", "n3"]
    added = updated[1]
    assert (added.kind, added.ids, added.created_dream, added.specialist) == (
        "pending",
        ["a1", "b2"],
        4,
        "deduction",
    )
    assert updated[0].text == "still open"
    assert (edit.added, edit.replaced, edit.removed) == (["n3"], ["n1"], ["n2"])
    # The input is not mutated.
    assert entries[0].text == "text n1"


@pytest.mark.parametrize(
    "operations",
    [
        [],
        [{"op": "rewrite_all"}],
        [{"op": "add", "kind": "fact", "text": "x"}],
        [{"op": "add", "kind": "open", "text": "  "}],
        [{"op": "replace", "entry_id": "n9", "text": "x"}],
        [{"op": "remove", "entry_id": "n9"}],
        # A bad operation after a good one rejects the whole call.
        [{"op": "add", "kind": "open", "text": "ok"}, {"op": "remove"}],
    ],
)
def test_malformed_operations_change_nothing(operations: Any):
    with pytest.raises(ValueError):
        _apply([_entry("n1", "open")], operations)


@pytest.mark.parametrize(
    ("kind", "created", "dream", "expired"),
    [
        ("pending", 1, 3, False),
        ("pending", 1, 4, True),
        ("covered", 1, 5, False),
        ("covered", 1, 6, True),
        ("open", 1, 100, False),
        ("map", 1, 100, False),
    ],
)
def test_expiry_by_kind(kind: str, created: int, dream: int, expired: bool):
    assert entry_expired(_entry("n1", kind, created), dream) is expired


async def _seed_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    notes: dict[str, Any],
) -> SimpleNamespace:
    workspace, peer = sample_data
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            internal_metadata={NOTES_KEY: notes},
        )
    )
    await db_session.commit()
    return SimpleNamespace(
        workspace_name=workspace.name, observer=peer.name, observed=peer.name
    )


async def _stored(db_session: AsyncSession, ctx: SimpleNamespace) -> dict[str, Any]:
    db_session.expire_all()
    metadata = await db_session.scalar(
        select(models.Collection.internal_metadata).where(
            models.Collection.workspace_name == ctx.workspace_name,
            models.Collection.observer == ctx.observer,
        )
    )
    assert metadata is not None
    return dict(metadata[NOTES_KEY])


@pytest.mark.asyncio
async def test_handler_persists_drops_expired_and_keeps_the_cursor(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    cursor = {"dreams": 3, "through_id": None}
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "cursor": cursor,
            "entries": [
                _entry("n1", "pending", created_dream=1).model_dump(mode="json"),
                _entry("n2", "map", created_dream=1).model_dump(mode="json"),
            ],
        },
    )
    handle = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=4
    )
    result = await handle(
        ctx, {"operations": [{"op": "add", "kind": "open", "text": "check employer"}]}
    )
    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "notes_added": 1,
        "notes_replaced": 0,
        "notes_removed": 0,
        "pending_resolved": 0,
    }

    stored = await _stored(db_session, ctx)
    # n1 (pending, created in dream 1) expired by dream 4; the cursor survives.
    assert [e.id for e in parse_entries(stored)] == ["n2", "n3"]
    assert stored["cursor"] == cursor


@pytest.mark.asyncio
async def test_handler_rejects_writes_over_the_cap(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "NOTES_MAX_TOKENS", 100)
    ctx = await _seed_collection(db_session, sample_data, {})
    handle = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=1
    )
    result = await handle(
        ctx,
        {"operations": [{"op": "add", "kind": "map", "text": "word " * 200}]},
    )
    assert isinstance(result, str) and result.startswith("ERROR:")
    assert "over the cap of 50" in result
    assert parse_entries(await _stored(db_session, ctx)) == []


def _sized(entry_id: str, specialist: str, words: int, minute: int) -> dict[str, Any]:
    return NoteEntry(
        id=entry_id,
        kind="open",
        specialist=specialist,
        text="word " * words,
        created_dream=1,
        updated_at=NOW + datetime.timedelta(minutes=minute),
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_cap_is_shared_per_specialist_and_lets_edits_shrink(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "NOTES_MAX_TOKENS", 100)
    # Deduction is over its share, as after the cap is lowered.
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "entries": [
                _sized("n1", "deduction", 40, 0),
                _sized("n2", "deduction", 40, 1),
            ]
        },
    )
    deduction = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=1
    )
    induction = make_update_notes_handler(
        specialist="induction", run_id="r1", dream_number=1
    )

    grow = await deduction(
        ctx, {"operations": [{"op": "add", "kind": "open", "text": "one more"}]}
    )
    assert isinstance(grow, str) and grow.startswith("ERROR:")
    # Induction still has its own share.
    added = await induction(
        ctx, {"operations": [{"op": "add", "kind": "open", "text": "a pattern"}]}
    )
    assert isinstance(added, ToolResult)
    # Shrinking is allowed even while still over the share.
    shrink = await deduction(
        ctx, {"operations": [{"op": "replace", "entry_id": "n1", "text": "short"}]}
    )
    assert isinstance(shrink, ToolResult)
    assert [e.id for e in parse_entries(await _stored(db_session, ctx))] == [
        "n1",
        "n2",
        "n3",
    ]


@pytest.mark.asyncio
async def test_shadow_mode_evicts_the_oldest_own_entries(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "NOTES_MAX_TOKENS", 100)
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", "shadow")
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "entries": [
                _sized("n1", "deduction", 12, 0),
                _sized("n2", "induction", 40, 1),
                _sized("n3", "deduction", 12, 2),
            ]
        },
    )
    handle = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=1
    )
    result = await handle(
        ctx, {"operations": [{"op": "add", "kind": "open", "text": "word " * 12}]}
    )
    assert isinstance(result, ToolResult)
    # n1 is deduction's oldest; induction's entry is never evicted.
    assert [e.id for e in parse_entries(await _stored(db_session, ctx))] == [
        "n2",
        "n3",
        "n4",
    ]


@pytest.mark.asyncio
async def test_handler_reports_malformed_calls(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    ctx = await _seed_collection(db_session, sample_data, {})
    handle = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=1
    )
    result = await handle(ctx, {"operations": [{"op": "remove", "entry_id": "n5"}]})
    assert isinstance(result, str)
    assert result.startswith("ERROR: operation 1: no entry 'n5'")


async def _run(state: DreamState | None) -> tuple[Any, Any]:
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
            state=state,
        )
    return llm_call.call_args.kwargs, make_executor.call_args.kwargs["handler_resolver"]


@pytest.fixture(autouse=True)
def _no_metrics(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["shadow", "on"])
async def test_notes_mode_offers_the_tool(mode: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", mode)
    kwargs, resolver = await _run(DreamState(cursor=NotesCursor(dreams=2)))
    assert "update_notes" in [t["name"] for t in kwargs["tools"]]
    assert "## WORKING NOTES" in kwargs["messages"][0]["content"]
    assert resolver is not None and resolver("update_notes") is not None


@pytest.mark.asyncio
async def test_notes_need_the_loaded_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", "on")
    kwargs, resolver = await _run(None)
    assert "update_notes" not in [t["name"] for t in kwargs["tools"]]
    assert resolver is None


@pytest.mark.asyncio
async def test_notes_off_by_default():
    kwargs, _ = await _run(DreamState(cursor=NotesCursor()))
    assert "update_notes" not in [t["name"] for t in kwargs["tools"]]
    assert "WORKING NOTES" not in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_loading_prunes_expired_and_dangling_entries(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    from src.dreamer.working_memory import load_notes_for_pass

    workspace, peer = sample_data
    live_id = "live".ljust(21, "0")
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "entries": [
                _entry("n1", "pending", created_dream=1).model_dump(mode="json"),
                {
                    **_entry("n2", "open").model_dump(mode="json"),
                    "ids": ["gone".ljust(21, "0")],
                },
                {**_entry("n3", "map").model_dump(mode="json"), "ids": [live_id]},
                _entry("n4", "pending", created_dream=3).model_dump(mode="json"),
            ]
        },
    )
    db_session.add(
        models.Document(
            id=live_id,
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            content="lives in Lisbon",
            level="explicit",
            embedding=[0.1] * 1536,
        )
    )
    await db_session.commit()

    loaded = await load_notes_for_pass(
        ctx.workspace_name, observer=peer.name, observed=peer.name, dream_number=4
    )
    assert [e.id for e in loaded.entries] == ["n3", "n4"]
    assert (loaded.pending_expired, loaded.dropped_for_missing_ids) == (1, 1)
    assert loaded.pending == 1 and loaded.tokens > 0
    # The pruned list is written back.
    assert [e.id for e in parse_entries(await _stored(db_session, ctx))] == [
        "n3",
        "n4",
    ]


def test_notes_section_puts_pending_first_and_asks_to_resolve():
    from src.dreamer.working_memory import format_notes

    entries = [_entry("n1", "map"), _entry("n2", "pending")]
    text = format_notes(entries, specialist="deduction")
    assert text.index("[n2] pending (deduction)") < text.index("[n1] map (deduction)")
    assert "hints to check, not facts" in text
    assert "Resolve every pending entry marked (deduction)" in text
    # Another specialist's pending entries are shown but not its to resolve.
    assert "Resolve every pending" not in format_notes(entries, specialist="induction")
    assert "Resolve every pending" not in format_notes(
        [_entry("n1", "map")], specialist="deduction"
    )


@pytest.mark.asyncio
async def test_dialectic_reads_only_map_entries(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    from src.dreamer.working_memory import load_map_entries

    _, peer = sample_data
    dead_map = {
        **_entry("n9", "map").model_dump(mode="json"),
        "ids": ["gone".ljust(21, "0")],
    }
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "entries": [
                *(
                    _entry(f"n{i}", kind).model_dump(mode="json")
                    for i, kind in enumerate(["pending", "open", "covered", "map"])
                ),
                # Every conclusion it cited is gone, and no dream has pruned it.
                dead_map,
            ]
        },
    )
    entries = await load_map_entries(
        ctx.workspace_name, observer=peer.name, observed=peer.name
    )
    assert [(e.id, e.kind) for e in entries] == [("n3", "map")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "loaded"), [("on", True), ("shadow", False)])
async def test_notes_are_loaded_only_when_on(
    mode: str, loaded: bool, monkeypatch: pytest.MonkeyPatch
):
    from src.dreamer.working_memory import LoadedNotes

    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", mode)
    load = AsyncMock(
        return_value=LoadedNotes(entries=[_entry("n1", "pending")], pending_expired=2)
    )
    with (
        patch("src.dreamer.specialists.load_notes_for_pass", new=load),
        patch("src.dreamer.specialists.emit") as emit,
    ):
        kwargs, _ = await _run(DreamState(cursor=NotesCursor(dreams=6)))
    user = kwargs["messages"][1]["content"]
    assert ("## YOUR NOTES FROM EARLIER DREAMS" in user) is loaded
    assert load.await_count == (1 if loaded else 0)
    if loaded:
        assert load.await_args is not None
        assert load.await_args.kwargs["dream_number"] == 7
        event = emit.call_args.args[0]
        assert (event.pending_loaded, event.pending_expired) == (1, 2)


@pytest.mark.asyncio
async def test_a_failed_notes_load_runs_the_pass_without_them(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "NOTES_MODE", "on")
    load = AsyncMock(side_effect=RuntimeError("db down"))
    with patch("src.dreamer.specialists.load_notes_for_pass", new=load):
        kwargs, _ = await _run(DreamState(cursor=NotesCursor(dreams=1)))
    assert load.await_count == 1
    assert "## YOUR NOTES FROM EARLIER DREAMS" not in kwargs["messages"][1]["content"]
    # The tool is still offered, so the pass can write notes.
    assert "update_notes" in [t["name"] for t in kwargs["tools"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_after_read", [False, True])
async def test_notes_reset_mid_dream_refuses_writes(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    reset_after_read: bool,
):
    """A reset (scope removal) after the dream started, or between the handler's
    read and its write, must not be written over with the old notes."""
    from src.dreamer.working_memory import EPOCH_KEY

    ctx = await _seed_collection(
        db_session,
        sample_data,
        {"entries": [_entry("n1", "map").model_dump(mode="json")]},
    )
    handle = make_update_notes_handler(
        specialist="deduction", run_id="r1", dream_number=1, epoch=None
    )
    operations = {"operations": [{"op": "add", "kind": "open", "text": "x"}]}
    if reset_after_read:
        # The conditional write finds the epoch moved after the read.
        store = AsyncMock(return_value=False)
        with patch("src.dreamer.working_memory._store_entries", new=store):
            result = await handle(ctx, operations)
        assert store.await_count == 1
    else:
        await crud.merge_collection_internal_metadata_value(
            db_session,
            ctx.workspace_name,
            ctx.observer,
            ctx.observed,
            NOTES_KEY,
            {"entries": [], EPOCH_KEY: "e2"},
        )
        result = await handle(ctx, operations)

    assert isinstance(result, str) and "were reset during this dream" in result
    if not reset_after_read:
        assert parse_entries(await _stored(db_session, ctx)) == []


@pytest.mark.asyncio
async def test_store_is_refused_once_the_epoch_moved(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    from src.dreamer.working_memory import (
        EPOCH_KEY,
        _store_entries,  # pyright: ignore[reportPrivateUsage]
    )

    ctx = await _seed_collection(
        db_session, sample_data, {"entries": [], EPOCH_KEY: "e2"}
    )
    old = [_entry("n1", "map")]
    assert not await _store_entries(
        ctx.workspace_name, ctx.observer, ctx.observed, old, epoch=None
    )
    assert parse_entries(await _stored(db_session, ctx)) == []
    assert await _store_entries(
        ctx.workspace_name, ctx.observer, ctx.observed, old, epoch="e2"
    )
    assert [e.id for e in parse_entries(await _stored(db_session, ctx))] == ["n1"]


def test_replace_updates_ids_only_when_given():
    entries = [_entry("n1", "open").model_copy(update={"ids": ["a1"]})]
    kept, _ = _apply(entries, [{"op": "replace", "entry_id": "n1", "text": "t"}])
    assert kept[0].ids == ["a1"]
    swapped, _ = _apply(
        entries, [{"op": "replace", "entry_id": "n1", "text": "t", "ids": ["id:b2"]}]
    )
    assert swapped[0].ids == ["b2"]


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "entry_id": "n1", "text": "mine now"},
        {"op": "remove", "entry_id": "n1"},
    ],
)
def test_specialists_cannot_edit_each_others_entries(operation: dict[str, Any]):
    with pytest.raises(ValueError, match="belongs to deduction"):
        _apply([_entry("n1", "covered")], [operation], specialist="induction")


@pytest.mark.asyncio
async def test_loading_strips_dead_ids_but_keeps_the_entry(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    from src.dreamer.working_memory import load_notes_for_pass

    workspace, peer = sample_data
    live_id, gone_id = "live".ljust(21, "0"), "gone".ljust(21, "0")
    ctx = await _seed_collection(
        db_session,
        sample_data,
        {
            "entries": [
                {
                    **_entry("n1", "map").model_dump(mode="json"),
                    "ids": [live_id, gone_id],
                }
            ]
        },
    )
    db_session.add(
        models.Document(
            id=live_id,
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            content="works in robotics",
            level="explicit",
            embedding=[0.1] * 1536,
        )
    )
    await db_session.commit()

    loaded = await load_notes_for_pass(
        ctx.workspace_name, observer=peer.name, observed=peer.name, dream_number=2
    )
    assert [(e.id, e.ids) for e in loaded.entries] == [("n1", [live_id])]
    assert (loaded.ids_stripped, loaded.dropped_for_missing_ids) == (1, 0)
    assert parse_entries(await _stored(db_session, ctx))[0].ids == [live_id]
