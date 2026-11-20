"""`honcho ingest from <source> --follow`: what a following source sends to.

A source that follows its source live (catching up from a saved position, then
taking in new events until stopped) gets a ``FollowSink`` instead of the batch
sink. On top of ``container(name)`` and ``record(record)`` it offers:

* ``context(record)``: a record the source read again only as context for the
  bot filter (it was sent before the saved position); it's never sent.
* ``flush(live=...)``: send what's pending, and say which records are settled
  (sent and answered, whatever the outcome, or a bot record the filter dropped)
  and which are still held. Nothing is lost by a failed send: the records stay
  pending. A failure that may pass (Honcho unreachable, a server error, rate
  limited) is returned as ``Flushed.error``; any other is raised.
* ``position(instance)``: the position saved for this source instance, or None
  to start from the beginning. The sink remembers the revision it read.
* ``save_position(instance, position, ...)``: save it with that revision, so a
  second runner, or a reset that forgot the state, is noticed
  (``FollowConflict``). It's not a lock: two first runs both start. Returns
  False when the save failed in a way that may pass; the next one carries on.
* ``window``: how long a bot record waits for a person (``--bot-window``); a
  source reads that much history before its position as context.
* ``stop``: set when the user asks to stop (Ctrl-C, SIGTERM). The source
  finishes what it's doing, flushes and saves, then returns.

Bot records: with ``--bots interacted`` the sink keeps a rolling window of each
container's recent records (people's too) and runs the bot filter over it at
every flush. A person's record is sent at once and stays as context; a bot's is
sent once a person replies to it or mentions the bot, and dropped once the
window has passed: by source time (a newer record in its container is more than
the window newer), or, while following live, after waiting the window.
Experimental, like the rest of ingest.
"""

from __future__ import annotations

import datetime
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from honcho import (
    ConflictError,
    Honcho,
    RateLimitError,
    ServerError,
    filter_bot_records,
)
from honcho import ConnectionError as HonchoConnectionError
from honcho import TimeoutError as HonchoTimeoutError
from honcho.api_types import IngestRecord, IngestRecordDelete, parse_ingest_record
from pydantic import ValidationError

BATCH = 100
"""Records per request: the most the ingest route takes at once."""

_TRANSIENT = (HonchoConnectionError, HonchoTimeoutError, RateLimitError, ServerError)
"""Failures that may pass: the follower keeps what's pending and tries again."""


class FollowConflict(Exception):
    """Another runner saved this source's position, or a reset forgot it."""


class FollowRecordError(Exception):
    """A source handed over a record that isn't a valid ingest record."""

    def __init__(self, container: str, message: str) -> None:
        super().__init__(f"{container}: {message}")
        self.container: str = container


@dataclass
class Flushed:
    """What one flush settled and what it still holds, by record `source_id`."""

    settled: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    error: str | None = None
    """Why sending stopped, when it may work later; what wasn't sent is pending."""


@dataclass
class _Entry:
    record: IngestRecord
    received: datetime.datetime
    sent: bool = False


def _utc(moment: datetime.datetime) -> datetime.datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=datetime.UTC)
    return moment.astimezone(datetime.UTC)


def _key(record: IngestRecord | IngestRecordDelete) -> str:
    return record.source_id or ""


class FollowSink:
    """Pending records, the bot window, and the source's saved positions."""

    def __init__(
        self,
        client: Honcho,
        source: str,
        *,
        derive: str,
        bots: str,
        window: datetime.timedelta = datetime.timedelta(hours=1),
        now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
            datetime.UTC
        ),
    ) -> None:
        self.client: Honcho = client
        self.source: str = source
        self.derive: str = derive
        self.bots: str = bots
        self.window: datetime.timedelta = window
        self.now: Callable[[], datetime.datetime] = now
        self.stop: threading.Event = threading.Event()
        self.outcomes: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.held_back: int = 0
        self.sent: int = 0
        self._name: str = "(before any container)"
        self._deletes: list[IngestRecordDelete] = []
        self._context: defaultdict[str, list[_Entry]] = defaultdict(list)
        self._revisions: dict[str, str | None] = {}

    # The batch sink's two methods -------------------------------------------------

    def container(self, name: str) -> None:
        """The records that follow are from this container; for errors only."""
        self._name = name

    def record(self, record: Mapping[str, Any]) -> None:
        try:
            parsed = parse_ingest_record(record)
        except (ValueError, ValidationError) as e:
            raise FollowRecordError(self._name, str(e)) from None
        if isinstance(parsed, IngestRecordDelete):
            self._deletes.append(parsed)
            return
        container = parsed.container.id if parsed.container else ""
        self._context[container].append(_Entry(parsed, self.now()))

    def context(self, record: Mapping[str, Any]) -> None:
        """A record already sent, read again as context for the bot filter."""
        try:
            parsed = parse_ingest_record(record)
        except (ValueError, ValidationError) as e:
            raise FollowRecordError(self._name, str(e)) from None
        if isinstance(parsed, IngestRecordDelete):
            return
        container = parsed.container.id if parsed.container else ""
        self._context[container].append(_Entry(parsed, self.now(), sent=True))

    # Following -------------------------------------------------------------------------

    def flush(self, *, live: bool = True) -> Flushed:
        """Send what's ready; say what's settled and what's held.

        `live` says the source is following new events rather than reading
        history: then a held bot record also expires after waiting the window.
        A failure that may pass is returned in `error`; any other is raised.
        Either way, what wasn't answered stays pending.
        """
        flushed = Flushed()
        to_send: list[IngestRecord | IngestRecordDelete] = list(self._deletes)
        sending: list[_Entry] = []
        now = self.now()
        for entries in self._context.values():
            pending = [e for e in entries if not e.sent]
            if not pending:
                continue
            kept = self._kept(entries)
            newest = max(_utc(e.record.created_at) for e in entries)
            for entry in pending:
                key = _key(entry.record)
                if id(entry.record) in kept:
                    sending.append(entry)
                    continue
                expired = newest - _utc(entry.record.created_at) > self.window or (
                    live and now - entry.received > self.window
                )
                if expired or self.bots == "none":
                    entry.sent = True  # dropped: never sent, but settled
                    self.held_back += 1
                    flushed.settled.add(key)
                else:
                    flushed.held.add(key)
        to_send.extend(e.record for e in sending)

        answered = 0
        try:
            for start in range(0, len(to_send), BATCH):
                chunk = to_send[start : start + BATCH]
                result = self.client.ingest(chunk, derive=self.derive)  # pyright: ignore[reportArgumentType]
                for r in result.records:
                    self.outcomes[r.outcome] += 1
                    if r.reason:
                        self.reasons[r.reason] += 1
                answered += len(chunk)
                self.sent += len(chunk)
        except _TRANSIENT as e:
            flushed.error = f"{type(e).__name__}: {e}"
        finally:
            done = to_send[:answered]
            done_deletes = {id(r) for r in done if isinstance(r, IngestRecordDelete)}
            self._deletes = [d for d in self._deletes if id(d) not in done_deletes]
            done_ids = {id(r) for r in done}
            for entry in sending:
                if id(entry.record) in done_ids:
                    entry.sent = True
            flushed.settled.update(_key(r) for r in done)
            self._prune()
        return flushed

    def _kept(self, entries: list[_Entry]) -> set[int]:
        """The records the bot filter keeps, by identity."""
        records = [e.record for e in entries]
        if self.bots == "all":
            return {id(r) for r in records}
        kept, _ = filter_bot_records(
            records, keep="interacted" if self.bots == "interacted" else "none"
        )
        return {id(r) for r in kept}

    def _prune(self) -> None:
        """Keep each container's last window of records, by source time."""
        for container, entries in list(self._context.items()):
            newest = max(_utc(e.record.created_at) for e in entries)
            kept = [
                e
                for e in entries
                if not e.sent or newest - _utc(e.record.created_at) <= self.window
            ]
            if kept:
                self._context[container] = kept
            else:
                del self._context[container]

    def position(self, instance: str) -> dict[str, Any] | None:
        """The saved position for this source instance; None to start afresh."""
        state = self.client.ingest_sync_state(self.source, instance)
        self._revisions[instance] = state.revision if state else None
        return state.position if state else None

    def save_position(
        self,
        instance: str,
        position: dict[str, Any],
        *,
        last_record_at: datetime.datetime | None = None,
        error: str | None = None,
    ) -> bool:
        """Save the position (and health) with the revision read.

        Raises FollowConflict when the state changed since it was read. Returns
        False when the save failed in a way that may pass.
        """
        try:
            state = self.client.ingest_save_sync_state(
                self.source,
                instance,
                position,
                revision=self._revisions.get(instance),
                last_record_at=last_record_at,
                error=error,
            )
        except ConflictError:
            raise FollowConflict(
                f"The sync state of {self.source}/{instance} changed since it was read:"
                + " another runner is following it, or a reset forgot it. Stop the"
                + " other runner, then start again."
            ) from None
        except _TRANSIENT:
            return False
        self._revisions[instance] = state.revision
        return True


__all__ = ["BATCH", "FollowConflict", "FollowRecordError", "FollowSink", "Flushed"]
