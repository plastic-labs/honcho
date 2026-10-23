import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.config import settings
from src.dreamer.working_memory import (
    MAX_CURSOR_HOLDS,
    NOTES_KEY,
    DreamState,
    NotesCursor,
    WorkItem,
    WorkList,
    advance_cursor,
    clear_position,
    format_work_list,
    load_dream_state,
    plan_cursor_move,
)


async def load_work_list(workspace: str, *, observer: str, observed: str):
    return await load_dream_state(
        workspace, observer=observer, observed=observed, include_work_list=True
    )


T0 = datetime.datetime(2026, 10, 1, 12, tzinfo=datetime.UTC)


def _id(name: str) -> str:
    """Document ids must be 21 characters; padding keeps names sortable."""
    return name.ljust(21, "0")


async def _seed(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    rows: list[tuple[str, str, datetime.datetime]],
    internal_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    workspace, peer = sample_data
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=peer.name,
            observed=peer.name,
            internal_metadata=internal_metadata or {},
        )
    )
    await db_session.flush()
    for doc_id, level, created_at in rows:
        db_session.add(
            models.Document(
                id=_id(doc_id),
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                content=f"content {doc_id}",
                level=level,
                embedding=[0.1] * 1536,
                created_at=created_at,
            )
        )
    await db_session.commit()
    return workspace.name, peer.name


async def _stored_notes(db_session: AsyncSession, workspace: str, peer: str) -> Any:
    db_session.expire_all()
    return await db_session.scalar(
        select(models.Collection.internal_metadata).where(
            models.Collection.workspace_name == workspace,
            models.Collection.observer == peer,
            models.Collection.observed == peer,
        )
    )


@pytest.mark.asyncio
async def test_backlog_is_worked_through_over_consecutive_dreams(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "BRIEF_MAX_NEW_CONCLUSIONS", 2)
    # b and c share a timestamp: paging must neither skip nor repeat either.
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [
            ("a", "explicit", T0),
            ("b", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("c", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("d", "deductive", T0 + datetime.timedelta(minutes=2)),
        ],
        internal_metadata={"dream": {"last_dream_at": "2026-09-01T00:00:00+00:00"}},
    )

    first_state = await load_work_list(workspace, observer=peer, observed=peer)
    first = first_state.work_list
    assert first is not None
    assert [i.id for i in first.items] == [_id("a"), _id("b")]
    assert first.remaining == 1
    await advance_cursor(
        workspace,
        observer=peer,
        observed=peer,
        state=first_state,
        run_id="r1",
        any_specialist_succeeded=True,
        all_specialists_succeeded=True,
    )

    second_state = await load_work_list(workspace, observer=peer, observed=peer)
    second = second_state.work_list
    assert second is not None
    assert [i.id for i in second.items] == [_id("c")]
    assert second.remaining == 0
    await advance_cursor(
        workspace,
        observer=peer,
        observed=peer,
        state=second_state,
        run_id="r2",
        any_specialist_succeeded=True,
        all_specialists_succeeded=True,
    )

    metadata = await _stored_notes(db_session, workspace, peer)
    cursor = NotesCursor.model_validate(metadata[NOTES_KEY]["cursor"])
    assert (cursor.through_id, cursor.dreams, cursor.last_run_id) == (_id("c"), 2, "r2")
    # The scheduler's key is untouched.
    assert metadata["dream"] == {"last_dream_at": "2026-09-01T00:00:00+00:00"}

    third = (await load_work_list(workspace, observer=peer, observed=peer)).work_list
    assert third is not None and third.items == [] and third.remaining == 0


@pytest.mark.asyncio
async def test_page_stops_at_the_token_cap_but_shows_one(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.DREAM, "BRIEF_MAX_NEW_CONCLUSION_TOKENS", 1)
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [
            ("a", "explicit", T0),
            ("b", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("c", "explicit", T0 + datetime.timedelta(minutes=2)),
        ],
    )
    work_list = (
        await load_work_list(workspace, observer=peer, observed=peer)
    ).work_list
    assert work_list is not None
    # The first is over the cap on its own, and is still shown.
    assert [i.id for i in work_list.items] == [_id("a")]
    assert work_list.remaining == 2


@pytest.mark.asyncio
async def test_remaining_never_goes_negative():
    """A commit between the count and the page can make the page the longer."""
    docs = [
        SimpleNamespace(id=f"x{i}", content="c", created_at=T0, internal_metadata={})
        for i in range(2)
    ]
    with (
        patch("src.dreamer.working_memory.tracked_db") as tracked_db,
        patch(
            "src.dreamer.working_memory.crud.get_collection_internal_metadata_value",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.dreamer.working_memory.crud.query_explicit_documents_after",
            new=AsyncMock(return_value=(docs, 1)),
        ),
    ):
        tracked_db.return_value.__aenter__.return_value = AsyncMock()
        state = await load_work_list("w", observer="p", observed="p")
    assert state.work_list is not None
    assert len(state.work_list.items) == 2 and state.work_list.remaining == 0


async def _set_last_dream_at(
    db_session: AsyncSession, workspace: str, peer: str, when: datetime.datetime
) -> None:
    """What `process_dream` does to the scheduler's key at the end of every dream."""
    await crud.update_collection_internal_metadata(
        db_session,
        workspace,
        peer,
        peer,
        update_data={"dream": {"last_dream_at": when.isoformat()}},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("any_succeeded", [True, False])
async def test_failed_specialist_is_shown_the_same_list_again(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    any_succeeded: bool,
):
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [("a", "explicit", T0), ("b", "explicit", T0 + datetime.timedelta(minutes=1))],
        internal_metadata={"dream": {"last_dream_at": "2026-09-01T00:00:00+00:00"}},
    )
    state = await load_work_list(workspace, observer=peer, observed=peer)
    await advance_cursor(
        workspace,
        observer=peer,
        observed=peer,
        state=state,
        run_id="r1",
        any_specialist_succeeded=any_succeeded,
        all_specialists_succeeded=False,
    )
    # The first dream had no position to hold; the scheduler's bound moving
    # past the list must not lose it.
    await _set_last_dream_at(
        db_session, workspace, peer, T0 + datetime.timedelta(hours=1)
    )

    again = await load_work_list(workspace, observer=peer, observed=peer)
    assert again.work_list is not None
    assert [i.id for i in again.work_list.items] == [_id("a"), _id("b")]
    # A dream where a specialist ran still counts, so note expiry keeps moving.
    assert again.cursor.dreams == (1 if any_succeeded else 0)
    assert again.cursor.holds == (1 if any_succeeded else 0)


@pytest.mark.asyncio
async def test_position_moves_after_repeated_holds(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [("a", "explicit", T0), ("b", "explicit", T0 + datetime.timedelta(minutes=1))],
    )
    for dream in range(1, MAX_CURSOR_HOLDS + 1):
        state = await load_work_list(workspace, observer=peer, observed=peer)
        assert state.work_list is not None
        assert [i.id for i in state.work_list.items] == [_id("a"), _id("b")]
        await advance_cursor(
            workspace,
            observer=peer,
            observed=peer,
            state=state,
            run_id=f"r{dream}",
            any_specialist_succeeded=True,
            all_specialists_succeeded=False,
        )

    after = await load_work_list(workspace, observer=peer, observed=peer)
    assert after.work_list is not None and after.work_list.items == []
    assert (after.cursor.through_id, after.cursor.holds) == (_id("b"), 0)


def _state(items: int, remaining: int, *, holds: int = 0) -> DreamState:
    return DreamState(
        cursor=NotesCursor(holds=holds),
        work_list=WorkList(
            items=[
                WorkItem(id=f"x{i}", content="c", created_at=T0) for i in range(items)
            ],
            remaining=remaining,
        ),
    )


@pytest.mark.parametrize(
    ("all_succeeded", "holds", "backlog"),
    [(True, 0, 5), (False, 0, 7), (False, MAX_CURSOR_HOLDS - 1, 5)],
)
def test_backlog_counts_what_the_next_dream_must_show(
    all_succeeded: bool, holds: int, backlog: int
):
    _, due = plan_cursor_move(
        _state(2, 5, holds=holds),
        any_specialist_succeeded=True,
        all_specialists_succeeded=all_succeeded,
    )
    assert due == backlog


def test_clear_position_keeps_entries_and_count():
    stored = {
        "entries": [{"id": "n1"}],
        "cursor": {
            "through_created_at": T0.isoformat(),
            "through_id": "x",
            "dreams": 3,
        },
    }
    cleared = clear_position(stored)
    assert cleared is not None
    assert cleared["entries"] == [{"id": "n1"}]
    cursor = NotesCursor.model_validate(cleared["cursor"])
    assert (cursor.position(), cursor.dreams) == (None, 3)
    assert clear_position(cleared) is None
    assert clear_position(None) is None


@pytest.mark.asyncio
async def test_brief_off_clears_a_stale_position(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    last_dream_at = T0 - datetime.timedelta(days=1)
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [
            ("old", "explicit", T0 - datetime.timedelta(days=2)),
            ("new", "explicit", T0),
        ],
        internal_metadata={
            "dream": {"last_dream_at": last_dream_at.isoformat()},
            NOTES_KEY: {
                "cursor": {
                    "through_created_at": (T0 - datetime.timedelta(days=3)).isoformat(),
                    "through_id": _id("older"),
                    "dreams": 4,
                }
            },
        },
    )
    state = await load_dream_state(
        workspace, observer=peer, observed=peer, include_work_list=False
    )
    await advance_cursor(
        workspace,
        observer=peer,
        observed=peer,
        state=state,
        run_id="r1",
        any_specialist_succeeded=True,
        all_specialists_succeeded=True,
    )

    # Turning the brief back on starts from the last dream, not the old position.
    back_on = await load_work_list(workspace, observer=peer, observed=peer)
    assert back_on.cursor.position() is None
    assert back_on.cursor.dreams == 5
    assert back_on.work_list is not None
    assert [i.id for i in back_on.work_list.items] == [_id("new")]


@pytest.mark.asyncio
async def test_first_list_starts_at_the_last_dream(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [
            ("old", "explicit", T0 - datetime.timedelta(days=2)),
            ("new", "explicit", T0),
        ],
        internal_metadata={
            "dream": {"last_dream_at": (T0 - datetime.timedelta(days=1)).isoformat()}
        },
    )
    work_list = (
        await load_work_list(workspace, observer=peer, observed=peer)
    ).work_list
    assert work_list is not None
    assert [i.id for i in work_list.items] == [_id("new")]


@pytest.mark.asyncio
async def test_cursor_merge_keeps_other_note_keys(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [],
        internal_metadata={NOTES_KEY: {"entries": [{"id": "n1"}]}},
    )
    await advance_cursor(
        workspace,
        observer=peer,
        observed=peer,
        state=None,
        run_id="r1",
        any_specialist_succeeded=True,
        all_specialists_succeeded=True,
    )
    notes = (await _stored_notes(db_session, workspace, peer))[NOTES_KEY]
    assert notes["entries"] == [{"id": "n1"}]
    assert notes["cursor"]["dreams"] == 1
    assert notes["cursor"]["through_id"] is None


def test_format_lists_items_and_the_remainder():
    text = format_work_list(
        WorkList(
            items=[WorkItem(id="a1", content="likes tea", created_at=T0)],
            remaining=3,
        )
    )
    assert "- [id:a1] [2026-10-01] likes tea" in text
    assert "3 more are not shown" in text


def test_format_shows_when_the_message_was_sent():
    text = format_work_list(
        WorkList(
            items=[
                WorkItem(
                    id="a1",
                    content="likes tea",
                    created_at=T0,
                    stated_at=datetime.datetime(2024, 3, 5, tzinfo=datetime.UTC),
                )
            ],
            remaining=0,
        )
    )
    assert "- [id:a1] [2024-03-05] likes tea" in text


def test_format_empty_list():
    text = format_work_list(WorkList(items=[], remaining=0))
    assert "No explicit conclusions have been created" in text


@pytest.mark.asyncio
async def test_query_ignores_deleted_and_other_levels(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = await _seed(
        db_session,
        sample_data,
        [("x", "explicit", T0), ("y", "inductive", T0)],
    )
    await db_session.execute(
        models.Document.__table__.update()
        .where(models.Document.id == _id("x"))
        .values(deleted_at=T0)
    )
    await db_session.commit()
    docs, total = await crud.query_explicit_documents_after(
        db_session,
        workspace,
        observer=peer,
        observed=peer,
        after=None,
        since=None,
        limit=10,
    )
    assert list(docs) == [] and total == 0
