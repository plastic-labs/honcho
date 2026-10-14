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
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

from pydantic import BaseModel, Field, ValidationError

from src import crud
from src.config import settings
from src.dependencies import tracked_db
from src.utils.tokens import estimate_tokens
from src.utils.types import ToolResult

logger = logging.getLogger(__name__)

NOTES_KEY = "dream_notes"

NoteKind = Literal["pending", "open", "covered", "map"]
NOTE_KINDS: tuple[str, ...] = get_args(NoteKind)

# Dreams an entry survives, counted from the dream that wrote it. Enforced by
# code, never left to the model.
PENDING_TTL_DREAMS = 3
COVERED_TTL_DREAMS = 5


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


class NoteEntry(BaseModel):
    """One line of the dreamer's working notes for a pair."""

    id: str
    kind: NoteKind
    # Specialist that last wrote the entry.
    specialist: str
    text: str
    # Conclusion ids the entry refers to, without the `id:` prefix.
    ids: list[str] = Field(default_factory=list)
    # Ordinal of the dream that created the entry (cursor.dreams + 1 then).
    created_dream: int
    updated_at: datetime.datetime
    run_id: str | None = None

    def render(self) -> str:
        cited = f" (refs: {', '.join(f'id:{i}' for i in self.ids)})" if self.ids else ""
        return f"- [{self.id}] {self.kind}: {self.text}{cited}"


def entry_expired(entry: NoteEntry, dream_number: int) -> bool:
    """Whether an entry has outlived its kind's lifetime as of `dream_number`."""
    age = dream_number - entry.created_dream
    if entry.kind == "pending":
        return age >= PENDING_TTL_DREAMS
    if entry.kind == "covered":
        return age >= COVERED_TTL_DREAMS
    return False


def notes_tokens(entries: list[NoteEntry]) -> int:
    """Size of the notes as the model sees them, for the cap."""
    return estimate_tokens([entry.render() for entry in entries])


def parse_entries(stored: Any) -> list[NoteEntry]:
    """Entries from the stored `dream_notes` value, skipping any that don't parse."""
    if not isinstance(stored, dict):
        return []
    raw = cast(dict[str, Any], stored).get("entries")
    if not isinstance(raw, list):
        return []
    entries: list[NoteEntry] = []
    for item in cast(list[Any], raw):
        try:
            entries.append(NoteEntry.model_validate(item))
        except ValidationError:
            logger.warning("Dropping unparseable dream note entry")
    return entries


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

    def through(self) -> tuple[datetime.datetime, str] | None:
        """Position the cursor advances to: the last conclusion shown."""
        if not self.items:
            return None
        last = self.items[-1]
        return last.created_at, last.id


@dataclass(frozen=True)
class DreamState:
    """What a dream starts from: the cursor, the notes and the work list."""

    cursor: NotesCursor
    entries: list[NoteEntry] = field(default_factory=list)
    # Present only when the brief is enabled.
    work_list: WorkList | None = None

    @property
    def dream_number(self) -> int:
        """Ordinal of the dream about to run."""
        return self.cursor.dreams + 1


def _parse_iso(value: Any) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


async def load_dream_state(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    include_work_list: bool,
) -> DreamState:
    """Load the stored cursor and notes, and optionally the work list past the cursor.

    Before the first cursor exists, the scheduler's `last_dream_at` bounds the
    work list instead; a pair that has never dreamed starts from the beginning.
    """
    async with tracked_db("dream.working_memory", read_only=True) as db:
        stored_notes = await crud.get_collection_internal_metadata_value(
            db, workspace_name, observer, observed, NOTES_KEY
        )
        cursor = parse_cursor(stored_notes)
        entries = parse_entries(stored_notes)
        if not include_work_list:
            return DreamState(cursor=cursor, entries=entries)

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
            limit=settings.DREAM.BRIEF_MAX_NEW_CONCLUSIONS,
        )
        items = [
            WorkItem(id=doc.id, content=doc.content, created_at=doc.created_at)
            for doc in documents
        ]
    return DreamState(
        cursor=cursor,
        entries=entries,
        work_list=WorkList(items=items, remaining=total - len(items)),
    )


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
    state: DreamState | None,
    run_id: str,
) -> None:
    """Record a completed dream: bump the dream count and move past what it was shown.

    The position moves only through the conclusions the dream was shown, so a
    backlog larger than the cap is worked through over consecutive dreams.
    """
    if state is None:
        # The state failed to load before the dream; re-read so the count
        # stays accurate.
        state = await load_dream_state(
            workspace_name,
            observer=observer,
            observed=observed,
            include_work_list=False,
        )
    cursor = state.cursor
    through = state.work_list.through() if state.work_list is not None else None
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


# ---------------------------------------------------------------------------
# Working notes
# ---------------------------------------------------------------------------

UPDATE_NOTES_TOOL: dict[str, Any] = {
    "name": "update_notes",
    "description": (
        "Edit your working notes for this peer, which carry over to your next "
        "dream. Apply a list of operations, each changing one entry: `add` a new "
        "entry, `replace` an entry's text, or `remove` an entry. Notes are capped; "
        "a call that would exceed the cap is rejected, so remove or shorten "
        "entries first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "kind": {
                            "type": "string",
                            "enum": list(NOTE_KINDS),
                            "description": "Entry kind. Required for add.",
                        },
                        "text": {
                            "type": "string",
                            "description": "A sentence or two. Required for add and replace.",
                        },
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Conclusion ids the entry refers to (optional, add only).",
                        },
                        "entry_id": {
                            "type": "string",
                            "description": "Entry to change, e.g. n3. Required for replace and remove.",
                        },
                    },
                    "required": ["op"],
                },
            },
        },
        "required": ["operations"],
    },
}


def notes_system_section(max_tokens: int) -> str:
    """System-prompt section explaining the notes and their kinds."""
    return f"""## WORKING NOTES

You keep short notes for this peer that carry over to your next dream. Edit them with `update_notes`, one entry at a time:
- `pending`: a write you decided on but could not make. The next dream must make it or discard it. Dropped after {PENDING_TTL_DREAMS} dreams if unresolved.
- `open`: something to check next time, such as a conflict waiting on more evidence.
- `covered`: ground you reviewed and found nothing to update, so the next dream can skip it. Expires after {COVERED_TTL_DREAMS} dreams.
- `map`: how this peer's memory is organized, such as its main topics.

Keep each entry to a sentence or two and cite conclusion ids where they apply. Notes are capped at {max_tokens} tokens. Facts about the peer belong in conclusions, never only in notes."""


def _next_entry_id(entries: list[NoteEntry]) -> int:
    numbers = [
        int(entry.id[1:])
        for entry in entries
        if entry.id.startswith("n") and entry.id[1:].isdigit()
    ]
    return max(numbers, default=0) + 1


def _clean_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for value in cast(list[Any], raw):
        if isinstance(value, str) and value.strip():
            cleaned.append(value.strip().removeprefix("id:"))
    return cleaned


@dataclass
class _NotesEdit:
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def apply_note_operations(
    entries: list[NoteEntry],
    operations: Any,
    *,
    specialist: str,
    run_id: str | None,
    dream_number: int,
    now: datetime.datetime,
) -> tuple[list[NoteEntry], _NotesEdit]:
    """Apply `update_notes` operations to a copy of `entries`.

    Raises:
        ValueError: if any operation is malformed. Nothing is applied then.
    """
    if not isinstance(operations, list) or not operations:
        raise ValueError("operations must be a non-empty list")
    updated = [entry.model_copy() for entry in entries]
    by_id = {entry.id: entry for entry in updated}
    edit = _NotesEdit()
    next_id = _next_entry_id(updated)
    for index, raw in enumerate(cast(list[Any], operations), 1):
        if not isinstance(raw, dict):
            raise ValueError(f"operation {index} is not an object")
        op_input = cast(dict[str, Any], raw)
        op = op_input.get("op")
        text = op_input.get("text")
        entry_id = op_input.get("entry_id")
        if op == "add":
            kind = op_input.get("kind")
            if kind not in NOTE_KINDS:
                raise ValueError(
                    f"operation {index}: kind must be one of {', '.join(NOTE_KINDS)}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"operation {index}: add needs text")
            entry = NoteEntry(
                id=f"n{next_id}",
                kind=cast(NoteKind, kind),
                specialist=specialist,
                text=text.strip(),
                ids=_clean_ids(op_input.get("ids")),
                created_dream=dream_number,
                updated_at=now,
                run_id=run_id,
            )
            next_id += 1
            updated.append(entry)
            by_id[entry.id] = entry
            edit.added.append(entry.id)
        elif op == "replace":
            if not isinstance(entry_id, str) or entry_id not in by_id:
                raise ValueError(f"operation {index}: no entry {entry_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"operation {index}: replace needs text")
            target = by_id[entry_id]
            target.text = text.strip()
            target.specialist = specialist
            target.updated_at = now
            target.run_id = run_id
            edit.replaced.append(entry_id)
        elif op == "remove":
            if not isinstance(entry_id, str) or entry_id not in by_id:
                raise ValueError(f"operation {index}: no entry {entry_id!r}")
            updated.remove(by_id.pop(entry_id))
            edit.removed.append(entry_id)
        else:
            raise ValueError(f"operation {index}: op must be add, replace or remove")
    return updated, edit


async def _store_entries(
    workspace_name: str, observer: str, observed: str, entries: list[NoteEntry]
) -> None:
    async with tracked_db("dream.notes_write") as db:
        await crud.merge_collection_internal_metadata_value(
            db,
            workspace_name,
            observer,
            observed,
            NOTES_KEY,
            {
                "version": 1,
                "entries": [entry.model_dump(mode="json") for entry in entries],
            },
        )


NotesHandler = Callable[[Any, dict[str, Any]], Coroutine[Any, Any, ToolResult | str]]


def make_update_notes_handler(
    *, specialist: str, run_id: str | None, dream_number: int
) -> NotesHandler:
    """Tool handler for `update_notes`, bound to one specialist pass.

    Each call reads the stored notes, drops expired entries, applies the
    operations and writes the result back, so the notes survive a pass that
    fails later. A call that is malformed or would exceed the cap changes
    nothing.
    """

    async def handle(ctx: Any, tool_input: dict[str, Any]) -> ToolResult | str:
        async with tracked_db("dream.notes_read", read_only=True) as db:
            stored = await crud.get_collection_internal_metadata_value(
                db, ctx.workspace_name, ctx.observer, ctx.observed, NOTES_KEY
            )
        current = [
            entry
            for entry in parse_entries(stored)
            if not entry_expired(entry, dream_number)
        ]
        try:
            updated, edit = apply_note_operations(
                current,
                tool_input.get("operations"),
                specialist=specialist,
                run_id=run_id,
                dream_number=dream_number,
                now=datetime.datetime.now(datetime.UTC),
            )
        except ValueError as e:
            return f"ERROR: {e}. No notes were changed."

        cap = settings.DREAM.NOTES_MAX_TOKENS
        size = notes_tokens(updated)
        if size > cap:
            return (
                f"ERROR: the notes would be {size} tokens, over the cap of {cap}. "
                + "Remove or shorten entries first. No notes were changed."
            )

        await _store_entries(ctx.workspace_name, ctx.observer, ctx.observed, updated)
        changes = [
            f"{label} {', '.join(ids)}"
            for label, ids in (
                ("added", edit.added),
                ("replaced", edit.replaced),
                ("removed", edit.removed),
            )
            if ids
        ]
        return ToolResult(
            content=(
                f"Notes updated ({'; '.join(changes)}). "
                + f"They now use {size} of {cap} tokens."
            ),
            metadata={
                "notes_added": len(edit.added),
                "notes_replaced": len(edit.replaced),
                "notes_removed": len(edit.removed),
            },
        )

    return handle
