"""State the dreamer carries from one dream to the next.

Stored per observer/observed pair in `Collection.internal_metadata["dream_notes"]`.
It has its own top-level key, separate from the scheduler's `"dream"` key,
because `update_collection_internal_metadata` replaces top-level values whole:
the scheduler's end-of-dream write would otherwise overwrite it.

The cursor is written by code at the end of each dream, never by the model,
so it stays correct when a dream writes nothing.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from src import crud
from src.config import settings
from src.dependencies import tracked_db

NOTES_KEY = "dream_notes"


class NotesCursor(BaseModel):
    """How far the since-last-dream work list has been worked through."""

    # Keyset position of the last explicit conclusion shown to a dream.
    through_created_at: datetime.datetime | None = None
    through_id: str | None = None
    last_run_id: str | None = None
    # Omni dreams completed for this pair since the cursor was introduced.
    dreams: int = 0

    def position(self) -> tuple[datetime.datetime, str] | None:
        if self.through_created_at is None or self.through_id is None:
            return None
        return self.through_created_at, self.through_id


def parse_cursor(stored: Any) -> NotesCursor:
    """Cursor from the stored `dream_notes` value; a fresh one if absent or bad."""
    if not isinstance(stored, dict):
        return NotesCursor()
    try:
        return NotesCursor.model_validate(
            cast(dict[str, Any], stored).get("cursor") or {}
        )
    except ValidationError:
        return NotesCursor()


@dataclass(frozen=True)
class WorkItem:
    id: str
    content: str
    created_at: datetime.datetime


@dataclass(frozen=True)
class WorkList:
    """Explicit conclusions created since the last dream, oldest first."""

    items: list[WorkItem]
    # Matching conclusions not shown because of the cap.
    remaining: int
    cursor: NotesCursor

    def through(self) -> tuple[datetime.datetime, str] | None:
        """Position the cursor advances to: the last conclusion shown."""
        if not self.items:
            return None
        last = self.items[-1]
        return last.created_at, last.id


def _parse_iso(value: Any) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


async def load_work_list(
    workspace_name: str, *, observer: str, observed: str
) -> WorkList:
    """Load the cursor and the explicit conclusions past it.

    Before the first cursor exists, the scheduler's `last_dream_at` bounds the
    list instead; a pair that has never dreamed starts from the beginning.
    """
    limit = settings.DREAM.BRIEF_MAX_NEW_CONCLUSIONS
    async with tracked_db("dream.work_list", read_only=True) as db:
        stored_notes = await crud.get_collection_internal_metadata_value(
            db, workspace_name, observer, observed, NOTES_KEY
        )
        cursor = parse_cursor(stored_notes)
        since: datetime.datetime | None = None
        if cursor.position() is None:
            scheduler_state = await crud.get_collection_internal_metadata_value(
                db, workspace_name, observer, observed, "dream"
            )
            if isinstance(scheduler_state, dict):
                since = _parse_iso(
                    cast(dict[str, Any], scheduler_state).get("last_dream_at")
                )
        documents, total = await crud.query_explicit_documents_after(
            db,
            workspace_name,
            observer=observer,
            observed=observed,
            after=cursor.position(),
            since=since,
            limit=limit,
        )
        items = [
            WorkItem(id=doc.id, content=doc.content, created_at=doc.created_at)
            for doc in documents
        ]
    return WorkList(items=items, remaining=total - len(items), cursor=cursor)


def format_work_list(work_list: WorkList) -> str:
    """User-prompt section listing the new conclusions, one line each."""
    if not work_list.items:
        return (
            "## NEW SINCE THE LAST DREAM\n\n"
            + "No explicit conclusions have been created since the last dream."
        )
    lines = [
        f"- [id:{item.id}] [{item.created_at.date().isoformat()}] {item.content}"
        for item in work_list.items
    ]
    more = ""
    if work_list.remaining > 0:
        more = (
            f"\n\n{work_list.remaining} more are not shown; "
            + "they will come in the next dream."
        )
    return (
        "## NEW SINCE THE LAST DREAM\n\n"
        + "Explicit conclusions created since the last dream, oldest first. "
        + "This is what changed: start from these rather than re-surveying "
        + "older memory, and search older memory only where a new conclusion "
        + "bears on it.\n\n"
        + "\n".join(lines)
        + more
    )


async def advance_cursor(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    work_list: WorkList | None,
    run_id: str,
) -> None:
    """Record a completed dream: bump the dream count and move past what it was shown.

    The position moves only through the conclusions the dream was shown, so a
    backlog larger than the cap is worked through over consecutive dreams.
    """
    cursor = work_list.cursor if work_list is not None else NotesCursor()
    if work_list is None:
        # No list this dream (brief off): re-read so the count stays accurate.
        async with tracked_db("dream.cursor_read", read_only=True) as db:
            cursor = parse_cursor(
                await crud.get_collection_internal_metadata_value(
                    db, workspace_name, observer, observed, NOTES_KEY
                )
            )
    through = work_list.through() if work_list is not None else None
    updated = cursor.model_copy(
        update={
            "dreams": cursor.dreams + 1,
            "last_run_id": run_id,
            **(
                {"through_created_at": through[0], "through_id": through[1]}
                if through is not None
                else {}
            ),
        }
    )
    async with tracked_db("dream.cursor_write") as db:
        await crud.merge_collection_internal_metadata_value(
            db,
            workspace_name,
            observer,
            observed,
            NOTES_KEY,
            {"version": 1, "cursor": updated.model_dump(mode="json")},
        )
