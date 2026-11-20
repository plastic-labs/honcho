"""Hold back bot records no person interacted with, before they're ingested.

Bots post a lot that nobody answers: review walkthroughs, alerts, stats. Nothing
is derived about a bot, but its text is still read as context when the people
around it are derived and summarized, so unanswered bot text costs tokens and
adds noise. This keeps a bot's record when a person interacted with it, and
holds back the rest. The records themselves are unchanged, so a full export can
be kept and the decision made again later.

A person interacted with a bot record when:

* it's in a reply chain with a person's record (`metadata.reply_to` holds the
  `source_id` of the record replied to; a person replied to the bot, or the
  bot replied to a person, however long the chain), or
* a person @-mentions the bot in the same conversation (it's in their
  `participants`, and their text names it: `@name`, or Discord's `<@id>`): the
  bot's last record at or before the mention (what the person is answering) and
  its first record after it (the bot's answer) are kept.

A conversation is the record's container, narrowed by `metadata.thread` when it
has one (a pull request inside a repository). Records from different sources or
source instances never interact. Order is kept.
"""

from __future__ import annotations

import bisect
import datetime
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from .api_types import IngestParticipant, IngestRecord, IngestRecordDelete

BotRecords = Literal["interacted", "all", "none"]
"""Which bot records to keep: those a person interacted with, all, or none."""

R = TypeVar("R", bound=IngestRecord | IngestRecordDelete | Mapping[str, Any])


@dataclass(frozen=True)
class _Seen:
    """What the filter reads from one record, from its validated form."""

    instance: tuple[str, str]
    source_id: str | None
    reply_to: str | None
    conversation: tuple[str, str, str | None, str | None]
    at: datetime.datetime
    author: IngestParticipant | None
    participants: list[IngestParticipant]
    content: str

    @property
    def bot(self) -> bool:
        return self.author is not None and self.author.bot

    @property
    def person(self) -> bool:
        return self.author is not None and not self.author.bot


def _hint(metadata: Mapping[str, Any] | None, name: str) -> str | None:
    """A metadata hint used as a key: a string or a number, else ignored."""
    value = (metadata or {}).get(name)
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    return str(value)


def _seen(record: IngestRecord | Mapping[str, Any]) -> _Seen:
    model = (
        record
        if isinstance(record, IngestRecord)
        else IngestRecord.model_validate(record)
    )
    instance = (model.source, model.source_instance)
    container = model.container.id if model.container is not None else None
    at = model.created_at
    at = (at if at.tzinfo is not None else at.replace(tzinfo=datetime.UTC)).astimezone(
        datetime.UTC
    )
    return _Seen(
        instance=instance,
        source_id=model.source_id,
        reply_to=_hint(model.metadata, "reply_to"),
        conversation=(*instance, container, _hint(model.metadata, "thread")),
        at=at,
        author=model.author,
        participants=model.participants,
        content=model.content,
    )


def _bot_name(name: str) -> str:
    """A bot's name for matching: `dependabot[bot]` and `dependabot` are one."""
    folded = name.casefold()
    return folded.removesuffix("[bot]")


def _named_in(content: str, p: IngestParticipant) -> bool:
    """Whether a record's text @-mentions a participant, as opposed to naming
    it only because it replies to them: `@name`, or Discord's `<@id>`."""
    # The whole name: `@ci` and `@ci[bot]` name `ci[bot]`; `@city` doesn't.
    pattern = r"(?<![\w@])@" + re.escape(_bot_name(p.name)) + r"(?:\[bot\])?(?![\w-])"
    if re.search(pattern, content, re.IGNORECASE):
        return True
    return p.id is not None and (f"<@{p.id}>" in content or f"<@!{p.id}>" in content)


def filter_bot_records(
    records: Iterable[R], keep: BotRecords = "interacted"
) -> tuple[list[R], int]:
    """The records to ingest, and how many bot records were held back.

    A bot record is one whose author has `bot` set. With `keep="interacted"`,
    a bot record is kept only if a person interacted with it (see the module
    docs); `"all"` keeps every record, `"none"` drops every bot record. Records
    whose author isn't a bot are always kept. The whole list is read first,
    since a reply can come after the record it answers. Records are read as
    `ingest` would validate them, and returned unchanged.
    """
    items = list(records)
    if keep == "all":
        return items, 0
    # A delete has no author and nothing to read: it's always sent.
    upserts = [r for r in items if not _is_delete(r)]
    kept, held_back = _filter_upserts(upserts, keep)
    kept_ids = {id(r) for r in kept}
    return [r for r in items if _is_delete(r) or id(r) in kept_ids], held_back


def _is_delete(record: Any) -> bool:
    if isinstance(record, IngestRecordDelete):
        return True
    return isinstance(record, Mapping) and record.get("op") == "delete"  # pyright: ignore[reportUnknownMemberType]


def _filter_upserts(items: list[Any], keep: BotRecords) -> tuple[list[Any], int]:
    """filter_bot_records over records to ingest (no deletes)."""
    seen = [_seen(r) for r in items]
    if keep == "none":
        kept = [r for r, s in zip(items, seen, strict=True) if not s.bot]
        return kept, len(items) - len(kept)
    if not any(s.bot for s in seen):
        return items, 0

    # Reply chains as connected groups. Each record is joined to the ID it
    # replies to, whether or not that record is in the list (a thread's first
    # comment may have been dropped as empty), so a chain of any length, a
    # cycle, and replies to a missing root are each one group.
    parent: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    def find(node: tuple[Any, ...]) -> tuple[Any, ...]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def node(i: int) -> tuple[Any, ...]:
        s = seen[i]
        if s.source_id is not None:
            return ("id", *s.instance, s.source_id)
        return ("record", i)

    for i, s in enumerate(seen):
        here = find(node(i))
        if s.reply_to is not None:
            parent[here] = find(("id", *s.instance, s.reply_to))
    groups_with_a_person = {find(node(i)) for i, s in enumerate(seen) if s.person}
    keep_index = {
        i for i, s in enumerate(seen) if s.bot and find(node(i)) in groups_with_a_person
    }

    # Each bot's records per conversation, in time order, keyed by its ID when
    # it has one, else its name. Names are matched within a conversation only.
    def bot_key(p: IngestParticipant) -> str:
        return f"id:{p.id}" if p.id is not None else f"name:{_bot_name(p.name)}"

    timelines: dict[tuple[Any, ...], list[tuple[datetime.datetime, int]]] = defaultdict(
        list
    )
    keys_by_name: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for i, s in enumerate(seen):
        if s.bot and s.author is not None:
            key = bot_key(s.author)
            timelines[(s.conversation, key)].append((s.at, i))
            keys_by_name[(s.conversation, _bot_name(s.author.name))].add(key)
    times: dict[tuple[Any, ...], list[datetime.datetime]] = {}
    for key, timeline in timelines.items():
        timeline.sort(key=lambda entry: entry[0])
        times[key] = [at for at, _ in timeline]

    def mentioned(s: _Seen, p: IngestParticipant) -> set[str]:
        """The bot timelines a mention can mean, in this conversation: its ID,
        and any of its records that give only its name; or, without an ID,
        every bot by that name."""
        named = keys_by_name.get((s.conversation, _bot_name(p.name)), set())
        if p.id is None:
            return named
        return {bot_key(p)} | {k for k in named if k.startswith("name:")}

    for s in seen:
        if not s.person:
            continue
        for p in s.participants:
            if not p.bot or not _named_in(s.content, p):
                continue
            for key in mentioned(s, p):
                timeline = timelines.get((s.conversation, key))
                if not timeline:
                    continue
                after = bisect.bisect_right(times[(s.conversation, key)], s.at)
                if after > 0:
                    keep_index.add(timeline[after - 1][1])
                if after < len(timeline):
                    keep_index.add(timeline[after][1])

    kept = [r for i, r in enumerate(items) if not seen[i].bot or i in keep_index]
    return kept, len(items) - len(kept)
