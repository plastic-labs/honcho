"""`honcho ingest from discord --follow`: catch up, then follow a server live.

The follower keeps a cursor per channel and thread: the newest message ID such
that every message up to it was accepted by ingest or deliberately skipped (a
system notice, or a bot record nobody answered). The cursors are the position
it saves in Honcho, so a restart, on any machine, carries on from them.

* **Catch-up** reads each container's history after its cursor over REST, up to
  the newest message the container had when the read began; anything newer
  comes from the live events. It runs at the start, whenever the connection
  begins a fresh session (events in between were missed), and after events
  were dropped because the follower fell behind. It also reads the bot window
  (an hour) before the cursor again, as context for the bot filter only.
* **Live events** come from discord.py, which runs the connection (heartbeats,
  resuming, reconnecting) in a thread of its own. The follower takes the raw
  dispatch payloads: the same JSON the REST API returns, so a live message
  becomes a record exactly as an exported one does. Channels and threads are
  always listed over REST, never read from discord.py's caches.
* **What it promises:** after any downtime, new messages are recovered from
  history, except one created and deleted while the follower was down. Edits
  and deletes are best effort: taken while connected, and they change stored
  records only during the dry run. One made while the follower was down isn't
  seen: a cursor over message IDs can't see changes to older messages.

Experimental, like the rest of ingest.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from honcho_discord.client import DiscordClient, DiscordError
from honcho_discord.export import Selection, SelectionError, select
from honcho_discord.mapping import (
    EXPORTABLE_TYPES,
    MESSAGE_CHANNEL_TYPES,
    SOURCE,
    THREAD_TYPES,
    Container,
    Json,
    items,
    obj,
    to_record,
)

logger = logging.getLogger(__name__)

DISCORD_EPOCH_MS = 1_420_070_400_000
QUEUE_SIZE = 10_000
"""Live events waiting for the follower; past this they're dropped and caught up."""
FLUSH_RECORDS = 100
FLUSH_SECONDS = 5.0
MAX_BACKOFF = 60.0
MAX_PENDING = 1_000
"""While Honcho is failing, stop reading once this many records wait to be sent."""
RETRY_SECONDS = 60.0
"""How long before reading a channel again after Discord failed to answer."""

EVENTS = frozenset(
    {
        "READY",
        "RESUMED",
        "MESSAGE_CREATE",
        "MESSAGE_UPDATE",
        "MESSAGE_DELETE",
        "MESSAGE_DELETE_BULK",
        "THREAD_CREATE",
        "THREAD_UPDATE",
        "THREAD_LIST_SYNC",
        "CHANNEL_CREATE",
        "CHANNEL_UPDATE",
    }
)
"""The dispatches the follower reads; discord.py handles the rest itself."""


def snowflake_time(snowflake: int) -> datetime.datetime:
    """When a Discord ID was made."""
    return datetime.datetime.fromtimestamp(
        ((snowflake >> 22) + DISCORD_EPOCH_MS) / 1000, datetime.UTC
    )


def snowflake_before(snowflake: int, span: datetime.timedelta) -> int:
    """The first ID made `span` before this one (0 before Discord's epoch)."""
    ms = (snowflake >> 22) - int(span.total_seconds() * 1000)
    return max(0, ms) << 22


class Flushed(Protocol):
    settled: set[str]
    held: set[str]
    error: str | None


class FollowSink(Protocol):
    """What `honcho ingest from discord --follow` hands over: `honcho_cli.follow`."""

    stop: threading.Event
    window: datetime.timedelta

    def container(self, name: str) -> None: ...

    def record(self, record: Mapping[str, Any]) -> None: ...

    def context(self, record: Mapping[str, Any]) -> None: ...

    def flush(self, *, live: bool = True) -> Flushed: ...

    def position(self, instance: str) -> dict[str, Any] | None: ...

    def save_position(
        self,
        instance: str,
        position: dict[str, Any],
        *,
        last_record_at: datetime.datetime | None = None,
        error: str | None = None,
    ) -> bool: ...


FOLLOW_METHODS = ("context", "flush", "position", "save_position")


class Events(Protocol):
    """Live dispatches, `{"t": type, "d": data}`, in the order Discord sent them."""

    def start(self) -> None: ...

    def get(self, timeout: float) -> Json | None:
        """The next dispatch, or None when none came within `timeout` seconds."""
        ...

    def take_overflow(self) -> bool:
        """Whether dispatches were dropped since last asked."""
        ...

    def drain(self) -> Sequence[Json]:
        """Every dispatch waiting now, without waiting for more."""
        ...

    def failure(self) -> str | None:
        """Why the connection stopped for good (a bad token, say), or None."""
        ...

    def close(self) -> None: ...


class GatewayEvents:
    """Discord's live events, from discord.py running in a thread of its own.

    The thread only queues dispatches; it never blocks on the follower. When the
    queue is full a dispatch is dropped and the overflow flag set, so the
    follower catches up from history instead.
    """

    def __init__(self, token: str, *, size: int = QUEUE_SIZE) -> None:
        self._token: str = token
        self._queue: queue.Queue[Json] = queue.Queue(size)
        self._overflow: threading.Event = threading.Event()
        self._closing: threading.Event = threading.Event()
        self._failure: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._thread: threading.Thread = threading.Thread(
            target=self._run, name="discord-gateway", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def get(self, timeout: float) -> Json | None:
        try:
            return self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def take_overflow(self) -> bool:
        if not self._overflow.is_set():
            return False
        self._overflow.clear()
        return True

    def drain(self) -> list[Json]:
        drained: list[Json] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                return drained

    def failure(self) -> str | None:
        return self._failure

    def close(self) -> None:
        """Disconnect, waiting a few seconds at most (the thread is a daemon)."""
        self._closing.set()
        loop, client = self._loop, self._client
        if loop is not None and client is not None and not loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(client.close(), loop).result(3)
            except Exception:  # best effort: the process is stopping anyway
                logger.debug("The Discord connection didn't close cleanly")
        self._thread.join(2)

    def receive(self, message: Any) -> None:
        """One raw gateway message: queue it if it's a dispatch the follower reads."""
        payload: Any = message
        if isinstance(message, (bytes, str)):
            try:
                payload = json.loads(message)
            except ValueError:
                return
        if not isinstance(payload, Mapping):
            return
        data = obj(payload)
        if data.get("op") != 0 or data.get("t") not in EVENTS:
            return
        try:
            self._queue.put_nowait({"t": data["t"], "d": data.get("d")})
        except queue.Full:
            self._overflow.set()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as error:
            self._fail(f"the Discord connection failed ({type(error).__name__})")

    async def _main(self) -> None:
        import discord  # only a follower needs it; the export doesn't

        receive = self.receive

        class _Client(discord.Client):
            async def on_socket_raw_receive(self, msg: Any) -> None:
                receive(msg)

        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        client = _Client(
            intents=intents,
            enable_debug_events=True,
            max_messages=None,
            chunk_guilds_at_startup=False,
            member_cache_flags=discord.MemberCacheFlags.none(),
        )
        self._client = client
        self._loop = asyncio.get_running_loop()
        if self._closing.is_set():
            return
        try:
            async with client:
                await client.start(self._token, reconnect=True)
        except discord.LoginFailure:
            self._fail("Discord refused the bot token")
        except discord.PrivilegedIntentsRequired:
            self._fail(
                "the bot needs the Message Content intent: turn it on under Bot →"
                + " Privileged Gateway Intents in the Discord Developer Portal"
            )
        except discord.ConnectionClosed as error:
            self._fail(f"Discord closed the connection (code {error.code})")
        else:
            self._fail("the Discord connection ended")

    def _fail(self, reason: str) -> None:
        if not self._closing.is_set() and self._failure is None:
            self._failure = reason


@dataclass
class _Cursor:
    at: int = 0
    """Every message up to this ID is settled; 0 before the first."""
    pending: dict[int, bool] = field(default_factory=dict)
    """Messages handed to the sink since, and whether each is settled."""
    done: int = 0
    """The newest message taken off `pending` (settled, with all before it)."""


@dataclass
class FollowStats:
    records: int = 0
    edits: int = 0
    deletes: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    catch_ups: int = 0


class Follower:
    """Catch up a server's selected channels and threads, then follow them live."""

    def __init__(
        self,
        client: DiscordClient,
        guild_id: str,
        sink: FollowSink,
        events: Events,
        *,
        channel_ids: Collection[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client: DiscordClient = client
        self.guild: str = guild_id
        self.sink: FollowSink = sink
        self.events: Events = events
        self.channel_ids: Collection[str] | None = channel_ids
        self.clock: Callable[[], float] = clock
        self.stats: FollowStats = FollowStats()
        self.containers: dict[str, Container] = {}
        self.cursors: dict[str, _Cursor] = {}
        self._parents: set[str] = set()
        self._ignored: set[str] = set()
        self._members: dict[str, Json] = {}
        self._members_read: bool = False
        self._where: dict[int, str] = {}
        # Containers whose cursor mustn't move until a catch-up has read them
        # again, with the overflow epoch that put them there.
        self._gaps: dict[str, int] = {}
        self._epoch: int = 0
        self._ready: bool = False
        self._catch_up_wanted: bool = False
        self._catching_up: bool = False
        self._new: list[str] = []
        self._later: dict[str, float] = {}
        self._problems: dict[str, str] = {}
        self._incomplete: dict[str, str] = {}
        self._rediscover_at: float | None = None
        self._listed: bool = False
        self._stopping: str | None = None
        # Threads seen whose channel isn't followed, by ID.
        self._unparented: dict[str, Json] = {}
        # Channels a live event named that couldn't be looked up, and when to
        # try again.
        self._lookups: dict[str, float] = {}
        self._handed: int = 0
        self._failures: int = 0
        self._next_flush: float = clock() + FLUSH_SECONDS
        self._saved: tuple[dict[str, Any], str | None] | None = None

    # Running ---------------------------------------------------------------------

    def run(self) -> FollowStats:
        """Follow until the sink's `stop` is set; then send, save and return.

        Raises SystemExit with the reason when the connection stops for good (a
        bad token, a missing intent) or the selection can't be read at the start.
        """
        position = self.sink.position(self.guild) or {}
        for cid, mid in obj(position.get("messages")).items():
            try:
                self.cursors[str(cid)] = _Cursor(at=int(mid))
            except (TypeError, ValueError):
                logger.warning("Ignoring a malformed saved position for %s", cid)
        logger.info("Connecting to Discord…")
        self.events.start()
        try:
            failure = self._follow()
        except KeyboardInterrupt:
            raise  # a second Ctrl-C: stop at once (the connection's thread is a daemon)
        except BaseException:
            self.events.close()
            raise
        self.events.close()
        self._flush()
        self._save(failure)
        if failure is not None:
            raise SystemExit(f"Stopped following: {failure}")
        return self.stats

    def _follow(self) -> str | None:
        while not self.sink.stop.is_set():
            failure = self.events.failure() or self._stopping
            if failure is not None:
                return failure
            self._check_overflow()
            if self._ready and self._catch_up_wanted:
                self._catch_up_all()
                continue
            now = self.clock()
            if self._rediscover_at is not None and self._rediscover_at <= now:
                self._rediscover()
                continue
            due = [cid for cid, when in self._lookups.items() if when <= now]
            if due:
                # One a pass: live events and sending go on between them.
                cid = min(due, key=self._lookups.__getitem__)
                del self._lookups[cid]
                self._container(cid)
            for cid, when in list(self._later.items()):
                if when <= now:
                    del self._later[cid]
                    self._new.append(cid)
            if self._new:
                container = self.containers.get(self._new.pop(0))
                if container is not None:
                    self._catch_up(container)
                continue
            if not self._waiting_for_honcho():
                event = self.events.get(
                    min(1.0, max(0.05, self._next_flush - self.clock()))
                )
                if event is not None:
                    self._handle(event)
            self._keep_up()
        return None

    # Catching up -------------------------------------------------------------------

    def _catch_up_all(self) -> None:
        """Read every selected channel and thread from its cursor."""
        self._catch_up_wanted = False
        self._catching_up = True
        try:
            self._check_overflow()
            selection = self._select()
            if selection is None:
                return
            if not self._members_read:
                self._read_members()
            order = [c.id for c in selection.containers()]
            order += [cid for cid in self.containers if cid not in order]
            self._new.clear()
            for cid in order:
                if self.sink.stop.is_set():
                    return
                container = self.containers.get(cid)
                if container is not None:
                    self._catch_up(container)
            self.stats.catch_ups += 1
            if self._flush():
                self._save()
            logger.info(
                "Caught up on %d channels and threads; following live", len(order)
            )
        finally:
            self._catching_up = False

    def _list(self) -> Selection:
        """The selection now, leaving out selected channels deleted since the start.

        Raises SelectionError when none of them is left, or at the start.
        """
        try:
            return select(self.client, self.guild, self.channel_ids)
        except SelectionError:
            if self.channel_ids is None or not self._listed:
                raise
            exportable = {
                str(c["id"])
                for c in self.client.channels(self.guild)
                if c.get("type") in EXPORTABLE_TYPES
            }
            gone = sorted(set(self.channel_ids) - exportable)
            left = set(self.channel_ids) & exportable
            if not gone or not left:
                raise
            logger.warning("Channels %s are gone; following the rest", ", ".join(gone))
            self.channel_ids = left
            return select(self.client, self.guild, self.channel_ids)

    def _select(self) -> Selection | None:
        first = not self._listed
        try:
            selection = self._list()
        except SelectionError as error:
            if first:
                raise SystemExit(str(error)) from None
            # Every selected channel is gone: stop the usual way, sending and
            # saving what's pending first.
            self._stopping = str(error)
            return None
        except DiscordError as error:
            if first:
                raise SystemExit(
                    f"Couldn't list the server's channels: {error}"
                ) from None
            logger.warning(
                "Couldn't list the server's channels (%s); trying again", error
            )
            self._catch_up_wanted = True
            self.sink.stop.wait(RETRY_SECONDS)
            return None
        self._listed = True
        self._take_selection(selection)
        return selection

    def _take_selection(self, selection: Selection) -> None:
        """Follow what a listing found; list again later if it was incomplete."""
        self._incomplete = dict(selection.incomplete)
        if self._incomplete:
            self._rediscover_at = self.clock() + RETRY_SECONDS
        for channel in selection.channels:
            self._follow_parent(channel.id)
            if channel.type in MESSAGE_CHANNEL_TYPES:
                self._add(channel)
        for threads in selection.threads.values():
            for thread in threads:
                self._add(thread)

    def _rediscover(self) -> None:
        """List the channels and threads again after a listing came back incomplete.

        Containers found now are read from their start, as new threads are.
        """
        self._rediscover_at = None
        try:
            selection = self._list()
        except (SelectionError, DiscordError) as error:
            logger.warning("Couldn't list the server's threads again (%s)", error)
            self._rediscover_at = self.clock() + RETRY_SECONDS
            return
        self._take_selection(selection)

    def _read_members(self) -> None:
        self._members_read = True
        try:
            for member in self.client.members(self.guild):
                self._members.setdefault(str(member["user"]["id"]), member)
        except DiscordError as error:
            logger.warning(
                "Couldn't list members (%s); nicknames come from new messages", error
            )

    def _catch_up(self, container: Container) -> None:
        """Read one container's history after its cursor, up to its newest message."""
        cid = container.id
        self._check_overflow()
        epoch = self._epoch
        cursor = self._cursor(cid)
        before = cursor.at
        try:
            newest = self.client.newest_message_id(cid)
            if newest is not None:
                cutoff = int(newest)
                start = snowflake_before(before, self.sink.window) if before else 0
                self.sink.container(f"#{container.name}")
                for message in self.client.messages(cid, after=str(start)):
                    mid = int(message["id"])
                    if mid > cutoff:
                        break
                    record = self._record(message, container)
                    if mid <= before:
                        if record is not None:
                            self.sink.context(record)
                        continue
                    self._take(cid, mid, record)
                    self._keep_up()
                    if self.sink.stop.is_set():
                        return
        except DiscordError as error:
            if error.status in (403, 404):
                logger.warning(
                    "Can't read #%s (%s); not following it", container.name, error
                )
                del self.containers[cid]
                self._ignored.add(cid)
                self._problems.pop(cid, None)
                return
            logger.warning(
                "Couldn't read #%s (%s); trying again soon", container.name, error
            )
            self._problems[cid] = f"#{container.name}: {error}"
            self._later[cid] = self.clock() + RETRY_SECONDS
            # Live messages mustn't move the cursor past the unread history.
            self._gaps[cid] = self._epoch
            return
        self._problems.pop(cid, None)
        self._check_overflow()
        if self._gaps.get(cid, -1) <= epoch:
            self._gaps.pop(cid, None)
        self._advance(cid)

    def _check_overflow(self) -> None:
        if not self.events.take_overflow():
            return
        logger.warning("Fell behind Discord's live events; catching up from history")
        self._ready = True
        self._epoch += 1
        for cid in self.containers:
            self._gaps[cid] = self._epoch
        self._catch_up_wanted = True
        # Empty the queue, so what arrives during the catch-up has room. New
        # messages in it come back with the history (a message in a channel or
        # thread not followed yet still makes it followed, history and all);
        # deletes and new threads are taken now; edits (each needs a request)
        # are let go: best effort.
        for event in self.events.drain():
            if self.sink.stop.is_set():
                break
            kind, data = event.get("t"), obj(event.get("d"))
            if kind not in ("MESSAGE_CREATE", "MESSAGE_UPDATE"):
                self._handle(event)
            elif str(data.get("guild_id")) == self.guild:
                self._container(str(data.get("channel_id")))

    # Live events -------------------------------------------------------------------

    def _handle(self, event: Json) -> None:
        kind = event.get("t")
        data = obj(event.get("d"))
        if kind == "READY":
            if self._ready:
                logger.info("New Discord session: catching up on what it missed")
            self._ready = True
            self._catch_up_wanted = True
            return
        if kind == "RESUMED" or str(data.get("guild_id")) != self.guild:
            return
        if kind == "MESSAGE_CREATE":
            self._created(data)
        elif kind == "MESSAGE_UPDATE":
            self._edited(data)
        elif kind == "MESSAGE_DELETE":
            self._deleted(str(data.get("channel_id")), [data.get("id")])
        elif kind == "MESSAGE_DELETE_BULK":
            self._deleted(str(data.get("channel_id")), data.get("ids") or [])
        elif kind == "THREAD_LIST_SYNC":
            for thread in items(data.get("threads")):
                self._container_changed({**thread, "guild_id": self.guild})
        else:
            self._container_changed(data)

    def _created(self, message: Json) -> None:
        container = self._container(str(message.get("channel_id")))
        if container is None:
            return
        author, member = obj(message.get("author")), obj(message.get("member"))
        if author.get("id") and member:
            self._members[str(author["id"])] = {**member, "user": author}
        self._take(container.id, int(message["id"]), self._record(message, container))

    def _edited(self, message: Json) -> None:
        # An update without an edit time is Discord adding a link preview.
        if not message.get("edited_timestamp"):
            return
        container = self._container(str(message.get("channel_id")))
        if container is None:
            return
        try:
            current = self.client.message(container.id, str(message["id"]))
        except DiscordError as error:
            logger.warning(
                "Couldn't read an edited message in #%s: %s", container.name, error
            )
            return
        if current is None:  # deleted since: its delete follows
            return
        record = self._record(current, container)
        if record is not None:
            self.sink.container(f"#{container.name}")
            self.sink.record(record)
            self._handed += 1
            self.stats.edits += 1

    def _deleted(self, channel_id: str, ids: Iterable[Any]) -> None:
        if self._container(channel_id) is None:
            return
        for message_id in ids:
            self.sink.record(
                {
                    "op": "delete",
                    "source": SOURCE,
                    "source_instance": self.guild,
                    "source_id": str(message_id),
                }
            )
            self._handed += 1
            self.stats.deletes += 1

    def _container_changed(self, channel: Json) -> None:
        """A channel or thread was created, renamed or joined."""
        cid = str(channel.get("id"))
        if cid in self.containers:
            self.containers[cid] = Container.from_channel(channel)
            return
        if cid in self._parents or not self._wanted(channel):
            return
        self._adopt(channel)

    def _container(self, channel_id: str) -> Container | None:
        """The container a live event is in, if it's followed.

        A channel or thread not heard of before is looked up once: one created
        or unarchived under a selected channel is followed from then on.
        """
        known = self.containers.get(channel_id)
        if known is not None or channel_id in self._ignored:
            return known
        if channel_id in self._unparented:
            return None
        if self._lookups.get(channel_id, 0.0) > self.clock():
            return None  # it failed lately: the follow loop tries it again
        try:
            channel = self.client.channel(channel_id)
        except DiscordError as error:
            logger.warning("Couldn't look up channel %s: %s", channel_id, error)
            self._lookups[channel_id] = self.clock() + RETRY_SECONDS
            return None
        self._lookups.pop(channel_id, None)
        if channel is None:
            self._ignored.add(channel_id)
            return None
        if not self._wanted(channel):
            if channel.get("type") in THREAD_TYPES:
                # Its channel may only not be followed yet (a new channel, or
                # the first listing still to come): kept for when it is.
                self._unparented[channel_id] = channel
            else:
                self._ignored.add(channel_id)
            return None
        self._adopt(channel)
        return self.containers.get(channel_id)

    def _follow_parent(self, channel_id: str) -> None:
        """A channel's threads are followed: so are the ones seen before it was."""
        self._parents.add(channel_id)
        for thread_id, thread in list(self._unparented.items()):
            if str(thread.get("parent_id")) == channel_id:
                del self._unparented[thread_id]
                self._adopt(thread)

    def _wanted(self, channel: Json) -> bool:
        if str(channel.get("guild_id", self.guild)) != self.guild:
            return False
        kind = channel.get("type")
        if kind in THREAD_TYPES:
            return str(channel.get("parent_id")) in self._parents
        if kind in EXPORTABLE_TYPES:
            return self.channel_ids is None or str(channel["id"]) in self.channel_ids
        return False

    def _adopt(self, channel: Json) -> None:
        """Follow a channel or thread found live, reading its history first."""
        container = Container.from_channel(channel)
        if container.type in EXPORTABLE_TYPES and container.type not in THREAD_TYPES:
            self._follow_parent(container.id)
            if container.type not in MESSAGE_CHANNEL_TYPES:
                return
        self._add(container)

    def _add(self, container: Container) -> None:
        """Follow a container; one not followed before is queued for a catch-up."""
        known = container.id in self.containers
        self.containers[container.id] = container
        self._ignored.discard(container.id)
        if known:
            return
        self._new.append(container.id)
        # Until its first catch-up its cursor can't move: a live message in it
        # would otherwise skip the history before it.
        self._gaps.setdefault(container.id, self._epoch)

    # Records and cursors -------------------------------------------------------------

    def _record(self, message: Json, container: Container) -> dict[str, Any] | None:
        try:
            return to_record(message, container, self.guild, self._members)
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(
                "Skipping a malformed message in #%s: %r", container.name, error
            )
            return None

    def _take(self, cid: str, mid: int, record: dict[str, Any] | None) -> None:
        """Hand a new message's record to the sink, and track it for the cursor."""
        if record is None:
            self.stats.skipped["system_or_empty"] += 1
            self._track(cid, mid, settled=True)
            return
        self.sink.container(f"#{self.containers[cid].name}")
        self.sink.record(record)
        self._handed += 1
        self.stats.records += 1
        self._track(cid, mid)

    def _cursor(self, cid: str) -> _Cursor:
        return self.cursors.setdefault(cid, _Cursor())

    def _track(self, cid: str, mid: int, *, settled: bool = False) -> None:
        cursor = self._cursor(cid)
        if mid <= cursor.at:
            return  # sent again: ingest takes it as a duplicate
        cursor.pending[mid] = settled or cursor.pending.get(mid, False)
        self._where[mid] = cid

    def _advance(self, cid: str) -> None:
        cursor = self._cursor(cid)
        for mid in sorted(cursor.pending):
            if not cursor.pending[mid]:
                break
            del cursor.pending[mid]
            self._where.pop(mid, None)
            cursor.done = max(cursor.done, mid)
        if cid in self._gaps:
            return
        # What was settled during a gap can be newer than history a catch-up
        # has handed out since: stop just before the oldest message unsettled.
        limit = cursor.done
        if cursor.pending:
            limit = min(limit, min(cursor.pending) - 1)
        cursor.at = max(cursor.at, limit)

    # Sending and saving ----------------------------------------------------------------

    def _waiting_for_honcho(self) -> bool:
        return bool(self._failures) and self._handed >= MAX_PENDING

    def _keep_up(self) -> None:
        """Flush when due; while Honcho is failing and a lot waits, stop reading."""
        while not self.sink.stop.is_set():
            if self._flush_due() and self._flush():
                self._save()
            if not self._waiting_for_honcho():
                return
            self.sink.stop.wait(max(0.05, self._next_flush - self.clock()))

    def _flush_due(self) -> bool:
        if self.clock() >= self._next_flush:
            return True
        return not self._failures and self._handed >= FLUSH_RECORDS

    def _flush(self) -> bool:
        """Send what's pending and move the cursors; False when Honcho failed."""
        flushed = self.sink.flush(live=not self._catching_up)
        touched: set[str] = set()
        for key in flushed.settled - flushed.held:
            if not key.isdigit():
                continue
            cid = self._where.get(int(key))
            if cid is not None and int(key) in self.cursors[cid].pending:
                self.cursors[cid].pending[int(key)] = True
                touched.add(cid)
        for cid in touched:
            self._advance(cid)
        now = self.clock()
        if flushed.error is None:
            if self._failures:
                logger.info("Honcho is answering again")
            self._failures = 0
            self._handed = 0
            self._next_flush = now + FLUSH_SECONDS
            return True
        self._failures += 1
        delay = min(MAX_BACKOFF, FLUSH_SECONDS * 2 ** min(self._failures, 8))
        logger.warning(
            "Couldn't send to Honcho (%s); trying again in %.0fs", flushed.error, delay
        )
        self._next_flush = now + delay
        return False

    def _save(self, failure: str | None = None) -> None:
        position: dict[str, Any] = {
            "messages": {
                cid: str(cursor.at)
                for cid, cursor in sorted(self.cursors.items())
                if cursor.at
            }
        }
        problems = [*self._problems.values()]
        problems += [
            f"threads of {cid}: {why}" for cid, why in sorted(self._incomplete.items())
        ]
        error = failure or ("; ".join(problems)[:2000] or None)
        if self._saved == (position, error):
            return
        newest = max((cursor.at for cursor in self.cursors.values()), default=0)
        saved = self.sink.save_position(
            self.guild,
            position,
            last_record_at=snowflake_time(newest) if newest else None,
            error=error,
        )
        if saved:
            self._saved = (position, error)


__all__ = [
    "FOLLOW_METHODS",
    "Events",
    "Follower",
    "FollowSink",
    "FollowStats",
    "GatewayEvents",
    "snowflake_before",
    "snowflake_time",
]
