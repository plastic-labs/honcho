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
# Set to a fresh value whenever the notes are reset from outside a dream (scope
# removal). Note writes made during a dream go through only while it still
# matches what the dream started with, so a reset can't be written over by an
# edit read before it, or by a dream whose prompt still holds the old notes.
EPOCH_KEY = "epoch"

NoteKind = Literal["pending", "open", "covered", "map"]
NOTE_KINDS: tuple[str, ...] = get_args(NoteKind)

# Dreams an entry survives, counted from the dream that wrote it. Enforced by
# code, never left to the model.
PENDING_TTL_DREAMS = 3
COVERED_TTL_DREAMS = 5

# Dreams in a row the position may hold for a failed specialist before it moves
# anyway, so a specialist that always fails for a pair can't pin the work list.
MAX_CURSOR_HOLDS = 3


class NotesCursor(BaseModel):
    """How far the since-last-dream work list has been worked through."""

    # Keyset position of the last explicit conclusion shown to a dream. An
    # empty `through_id` marks a start pinned before the first conclusion shown.
    through_created_at: datetime.datetime | None = None
    through_id: str | None = None
    last_run_id: str | None = None
    # Omni dreams completed for this pair since the cursor was introduced.
    dreams: int = 0
    # Dreams in a row the position held because a specialist failed.
    holds: int = 0

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
        return f"- [{self.id}] {self.kind} ({self.specialist}): {self.text}{cited}"


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


def notes_share_tokens() -> int:
    """Cap on one specialist's own entries.

    Deduction and induction each get half of `NOTES_MAX_TOKENS`. Each can
    remove only its own entries, so with one shared cap the first to fill it
    would lock the other out.
    """
    return settings.DREAM.NOTES_MAX_TOKENS // 2


def _own(entries: list[NoteEntry], specialist: str) -> list[NoteEntry]:
    return [entry for entry in entries if entry.specialist == specialist]


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


def parse_epoch(stored: Any) -> str | None:
    """The stored notes' epoch; None if they were never reset."""
    if not isinstance(stored, dict):
        return None
    epoch = cast(dict[str, Any], stored).get(EPOCH_KEY)
    return epoch if isinstance(epoch, str) else None


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
    # Insertion time: orders the list and positions the cursor.
    created_at: datetime.datetime
    # When the source message was sent: the date search results show.
    stated_at: datetime.datetime | None = None

    @property
    def shown_date(self) -> str:
        return (self.stated_at or self.created_at).date().isoformat()


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
    # The notes' epoch when the dream started; see `EPOCH_KEY`.
    notes_epoch: str | None = None
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
        epoch = parse_epoch(stored_notes)
        if not include_work_list:
            return DreamState(cursor=cursor, entries=entries, notes_epoch=epoch)

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
        items: list[WorkItem] = []
        tokens = 0
        for doc in documents:
            # The page stops at the token cap too; the rest count as remaining.
            tokens += estimate_tokens(doc.content)
            if items and tokens > settings.DREAM.BRIEF_MAX_NEW_CONCLUSION_TOKENS:
                break
            items.append(
                WorkItem(
                    id=doc.id,
                    content=doc.content,
                    created_at=doc.created_at,
                    stated_at=_parse_iso(
                        doc.internal_metadata.get("message_created_at")
                    ),
                )
            )
    return DreamState(
        cursor=cursor,
        entries=entries,
        notes_epoch=epoch,
        # The count and the page are separate autocommit reads, so a commit
        # between them can leave the page longer than the count.
        work_list=WorkList(items=items, remaining=max(0, total - len(items))),
    )


def format_work_list(work_list: WorkList) -> str:
    """User-prompt section listing the new conclusions, one line each."""
    if not work_list.items:
        return (
            "## NEW SINCE THE LAST DREAM\n\n"
            + "No explicit conclusions have been created since the last dream."
        )
    lines = [
        f"- [id:{item.id}] [{item.shown_date}] {item.content}"
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


_CLEARED_POSITION: dict[str, Any] = {
    "through_created_at": None,
    "through_id": None,
    "holds": 0,
}


def plan_cursor_move(
    state: DreamState,
    *,
    any_specialist_succeeded: bool,
    all_specialists_succeeded: bool,
) -> tuple[NotesCursor, int]:
    """Where a dream leaves the cursor, and how much of the work list is still due.

    The position moves only through the conclusions the dream was shown, so a
    backlog larger than the cap is worked through over consecutive dreams. It
    holds when a specialist failed, so that specialist is shown them again, but
    only for `MAX_CURSOR_HOLDS` dreams in a row. The dream count moves only when
    a specialist ran.

    Returns:
        The updated cursor, and the work-list conclusions still due after this
        dream: the ones not shown, plus the shown ones when the position held.
    """
    cursor = state.cursor
    update: dict[str, Any] = (
        {"dreams": cursor.dreams + 1} if any_specialist_succeeded else {}
    )
    work_list = state.work_list
    if work_list is None:
        # The brief is off. Clear the position so that turning it back on
        # starts from the scheduler's last dream, not a stale position.
        return cursor.model_copy(update={**update, **_CLEARED_POSITION}), 0
    through = work_list.through()
    if through is None:
        return cursor.model_copy(update=update), work_list.remaining

    held_too_long = any_specialist_succeeded and cursor.holds + 1 >= MAX_CURSOR_HOLDS
    if all_specialists_succeeded or held_too_long:
        if not all_specialists_succeeded:
            logger.warning(
                "Moving the dream cursor past %d conclusions after %d failed dreams",
                len(work_list.items),
                MAX_CURSOR_HOLDS,
            )
        update |= {"through_created_at": through[0], "through_id": through[1]}
        update["holds"] = 0
        return cursor.model_copy(update=update), work_list.remaining

    if any_specialist_succeeded:
        update["holds"] = cursor.holds + 1
    if cursor.position() is None:
        # Without a position the list starts from the scheduler's
        # `last_dream_at`, which this dream is about to move past what it was
        # shown. Pin the start of the list instead, so it is shown again.
        update |= {
            "through_created_at": work_list.items[0].created_at,
            "through_id": "",
        }
    return (
        cursor.model_copy(update=update),
        work_list.remaining + len(work_list.items),
    )


def clear_position(stored: Any) -> dict[str, Any] | None:
    """The stored `dream_notes` value with its position cleared.

    For dreams that run with working memory off and so never call
    `advance_cursor`: turning it back on must not resume from a position those
    dreams have moved past. Returns None when there is no position to clear.
    """
    if not isinstance(stored, dict):
        return None
    cursor = parse_cursor(stored)
    if cursor.position() is None:
        return None
    cleared = cursor.model_copy(update=_CLEARED_POSITION)
    return {**cast(dict[str, Any], stored), "cursor": cleared.model_dump(mode="json")}


async def advance_cursor(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    state: DreamState | None,
    run_id: str,
    any_specialist_succeeded: bool,
    all_specialists_succeeded: bool,
) -> None:
    """Record a dream's effect on the cursor; see `plan_cursor_move`."""
    if state is None:
        # The state failed to load before the dream, so what it was shown is
        # unknown: hold the position, and re-read so the count stays accurate.
        state = await load_dream_state(
            workspace_name,
            observer=observer,
            observed=observed,
            include_work_list=False,
        )
        dreams = state.cursor.dreams + (1 if any_specialist_succeeded else 0)
        updated = state.cursor.model_copy(update={"dreams": dreams})
    else:
        updated, _ = plan_cursor_move(
            state,
            any_specialist_succeeded=any_specialist_succeeded,
            all_specialists_succeeded=all_specialists_succeeded,
        )
    updated = updated.model_copy(update={"last_run_id": run_id})
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
        "entry, `replace` an entry's text, or `remove` an entry. Your entries are "
        "capped; a call that would grow them past the cap is rejected, so remove "
        "or shorten entries first."
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
                            "description": "Conclusion ids the entry refers to (optional; on replace, given ids replace the old ones).",
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

You keep short notes for this peer that carry over to your next dream. Edit them with `update_notes`, one entry at a time. Each entry shows which specialist wrote it; you can see every entry, but you can only replace or remove your own.
- `pending`: a write you decided on but could not make. The next dream must make it or discard it. Dropped after {PENDING_TTL_DREAMS} dreams if unresolved.
- `open`: something to check next time, such as a conflict waiting on more evidence.
- `covered`: ground you reviewed and found nothing to update, so the next dream can skip it. Expires after {COVERED_TTL_DREAMS} dreams.
- `map`: how this peer's memory is organized, such as its main topics.

Keep each entry to a sentence or two and cite conclusion ids where they apply. Your own entries are capped at {max_tokens} tokens. Facts about the peer belong in conclusions, never only in notes."""


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
    pending_removed: int = 0


def _require_own(entry: NoteEntry, specialist: str, index: int) -> None:
    """Reject an edit to another specialist's entry.

    Deduction and induction share one set of notes but keep separate working
    state; one rewriting the other's entry would lose that state.
    """
    if entry.specialist != specialist:
        raise ValueError(
            f"operation {index}: entry {entry.id} belongs to {entry.specialist}; "
            + "you can only replace or remove your own entries"
        )


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
            _require_own(by_id[entry_id], specialist, index)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"operation {index}: replace needs text")
            target = by_id[entry_id]
            target.text = text.strip()
            if "ids" in op_input:
                target.ids = _clean_ids(op_input.get("ids"))
            target.updated_at = now
            target.run_id = run_id
            edit.replaced.append(entry_id)
        elif op == "remove":
            if not isinstance(entry_id, str) or entry_id not in by_id:
                raise ValueError(f"operation {index}: no entry {entry_id!r}")
            _require_own(by_id[entry_id], specialist, index)
            removed_entry = by_id.pop(entry_id)
            updated.remove(removed_entry)
            edit.removed.append(entry_id)
            if removed_entry.kind == "pending":
                edit.pending_removed += 1
        else:
            raise ValueError(f"operation {index}: op must be add, replace or remove")
    return updated, edit


async def _store_entries(
    workspace_name: str,
    observer: str,
    observed: str,
    entries: list[NoteEntry],
    *,
    epoch: str | None,
) -> bool:
    """Write the entries back, only while the notes' epoch is still `epoch`.

    Returns:
        False when the notes were reset since `epoch` was read; nothing is
        written then.
    """
    async with tracked_db("dream.notes_write") as db:
        return await crud.merge_collection_internal_metadata_value(
            db,
            workspace_name,
            observer,
            observed,
            NOTES_KEY,
            {
                "version": 1,
                "entries": [entry.model_dump(mode="json") for entry in entries],
            },
            if_unchanged=(EPOCH_KEY, epoch),
        )


@dataclass(frozen=True)
class LoadedNotes:
    """Notes as loaded for one pass, after pruning."""

    entries: list[NoteEntry]
    pending_expired: int = 0
    dropped_for_missing_ids: int = 0
    # Dead conclusion ids removed from entries that still cite a live one.
    ids_stripped: int = 0

    @property
    def pending(self) -> int:
        return sum(1 for entry in self.entries if entry.kind == "pending")

    @property
    def tokens(self) -> int:
        return notes_tokens(self.entries)


_KIND_ORDER = {kind: index for index, kind in enumerate(NOTE_KINDS)}


async def load_notes_for_pass(
    workspace_name: str,
    *,
    observer: str,
    observed: str,
    dream_number: int,
    epoch: str | None = None,
) -> LoadedNotes:
    """Read the notes, prune them, and store the pruned list back.

    Expired entries are dropped. Ids of conclusions that no longer exist
    (deleted or superseded) are stripped, and an entry is dropped only when
    every conclusion it cited is gone: one outdated premise shouldn't erase a
    note that still rests on others. Pruning is written back so stale entries
    stop counting against the cap, unless the notes were reset since the dream
    started (`epoch`).
    """
    async with tracked_db("dream.notes_load", read_only=True) as db:
        stored = await crud.get_collection_internal_metadata_value(
            db, workspace_name, observer, observed, NOTES_KEY
        )
        entries = parse_entries(stored)
        cited = sorted({cited_id for entry in entries for cited_id in entry.ids})
        live_ids = {
            doc.id
            for doc in await crud.fetch_documents_by_ids(
                db, workspace_name, observer, observed, cited
            )
        }

    kept: list[NoteEntry] = []
    pending_expired = 0
    dropped_for_missing_ids = 0
    ids_stripped = 0
    for entry in entries:
        if entry_expired(entry, dream_number):
            if entry.kind == "pending":
                pending_expired += 1
            continue
        live = [cited_id for cited_id in entry.ids if cited_id in live_ids]
        if entry.ids and not live:
            dropped_for_missing_ids += 1
            continue
        if len(live) != len(entry.ids):
            ids_stripped += len(entry.ids) - len(live)
            entry = entry.model_copy(update={"ids": live})
        kept.append(entry)

    if len(kept) != len(entries) or ids_stripped:
        await _store_entries(workspace_name, observer, observed, kept, epoch=epoch)
    return LoadedNotes(
        entries=kept,
        pending_expired=pending_expired,
        dropped_for_missing_ids=dropped_for_missing_ids,
        ids_stripped=ids_stripped,
    )


def format_notes(entries: list[NoteEntry], *, specialist: str) -> str:
    """User-prompt section with the dreamer's notes, pending entries first.

    Every specialist's entries are shown, but `specialist` is asked to resolve
    only its own pending entries, since it can't edit the others.
    """
    if not entries:
        return (
            "## YOUR NOTES FROM EARLIER DREAMS\n\n"
            + "You have no notes yet. Add them with `update_notes` as you work."
        )
    ordered = sorted(entries, key=lambda entry: _KIND_ORDER[entry.kind])
    resume = ""
    if any(
        entry.kind == "pending" and entry.specialist == specialist for entry in entries
    ):
        resume = (
            f" Resolve every pending entry marked ({specialist}) before new work: "
            + "make the write if it still holds, then remove the entry; otherwise "
            + "just remove it."
        )
    return (
        "## YOUR NOTES FROM EARLIER DREAMS\n\n"
        + "These are the dreamer's notes from earlier dreams: hints to check, not "
        + f"facts.{resume}\n\n"
        + "\n".join(entry.render() for entry in ordered)
    )


async def load_map_entries(
    workspace_name: str, *, observer: str, observed: str
) -> list[NoteEntry]:
    """The `map` entries for a pair, for the dialectic.

    Other kinds are unverified working hypotheses an answering agent could
    repeat as facts, so they are never returned. Pruning only runs when a dream
    loads the notes, so the same rule is applied here: an entry whose cited
    conclusions have all been deleted since is skipped.
    """
    async with tracked_db("dialectic.dream_map", read_only=True) as db:
        stored = await crud.get_collection_internal_metadata_value(
            db, workspace_name, observer, observed, NOTES_KEY
        )
        entries = [entry for entry in parse_entries(stored) if entry.kind == "map"]
        cited = sorted({cited_id for entry in entries for cited_id in entry.ids})
        live_ids = {
            doc.id
            for doc in await crud.fetch_documents_by_ids(
                db, workspace_name, observer, observed, cited
            )
        }
    return [
        entry
        for entry in entries
        if not entry.ids or any(cited_id in live_ids for cited_id in entry.ids)
    ]


def _evict_oldest(
    entries: list[NoteEntry], specialist: str, cap: int
) -> tuple[list[NoteEntry], int]:
    """Drop the specialist's least recently updated entries until its own fit.

    Returns the remaining entries and the size of the specialist's own.
    """
    kept = list(entries)
    own = sorted(_own(kept, specialist), key=lambda entry: entry.updated_at)
    size = notes_tokens(own)
    while own and size > cap:
        kept.remove(own.pop(0))
        size = notes_tokens(own)
    return kept, size


_NOTES_RESET_ERROR = (
    "ERROR: your notes were reset during this dream, so they can't be edited "
    + "until the next one. No notes were changed."
)

NotesHandler = Callable[[Any, dict[str, Any]], Coroutine[Any, Any, ToolResult | str]]


def make_update_notes_handler(
    *,
    specialist: str,
    run_id: str | None,
    dream_number: int,
    epoch: str | None = None,
) -> NotesHandler:
    """Tool handler for `update_notes`, bound to one specialist pass.

    Each call reads the stored notes, drops expired entries, applies the
    operations and writes the result back, so the notes survive a pass that
    fails later. A call that is malformed, or that grows the specialist's own
    entries past its share of the cap, changes nothing. In shadow mode the
    model never sees entry ids and so can't remove anything: the specialist's
    oldest entries are evicted to fit instead. Once the notes were reset since
    the dream started (`epoch`), every call is refused: the model may still
    hold the old notes and would write them back.
    """

    async def handle(ctx: Any, tool_input: dict[str, Any]) -> ToolResult | str:
        async with tracked_db("dream.notes_read", read_only=True) as db:
            stored = await crud.get_collection_internal_metadata_value(
                db, ctx.workspace_name, ctx.observer, ctx.observed, NOTES_KEY
            )
        if parse_epoch(stored) != epoch:
            return _NOTES_RESET_ERROR
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

        cap = notes_share_tokens()
        size = notes_tokens(_own(updated, specialist))
        if size > cap and settings.DREAM.NOTES_MODE == "shadow":
            updated, size = _evict_oldest(updated, specialist, cap)
        # An edit that shrinks the entries is let through even over the cap,
        # so they can still be trimmed after the cap is lowered.
        if size > cap and size > notes_tokens(_own(current, specialist)):
            return (
                f"ERROR: your entries would be {size} tokens, over the cap of "
                + f"{cap}. Remove or shorten entries first. No notes were changed."
            )

        if not await _store_entries(
            ctx.workspace_name, ctx.observer, ctx.observed, updated, epoch=epoch
        ):
            return _NOTES_RESET_ERROR
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
                + f"Your entries now use {size} of {cap} tokens."
            ),
            metadata={
                "notes_added": len(edit.added),
                "notes_replaced": len(edit.replaced),
                "notes_removed": len(edit.removed),
                "pending_resolved": edit.pending_removed,
            },
        )

    return handle
