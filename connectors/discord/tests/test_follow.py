"""Following a server: catch-up from a position, live events, cursors, restarts."""

import datetime
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from honcho_discord.follow import (
    Follower,
    GatewayEvents,
    snowflake_before,
    snowflake_time,
)
from honcho_discord.source import SOURCE

from .fakes import BOT, GUILD, JESS, MAYA, FakeDiscord, client_for
from .fakes import message as msg


@dataclass
class Flushed:
    settled: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    error: str | None = None


class FakeSink:
    """The CLI's follow sink, with the sending faked: what was sent, held, saved."""

    def __init__(self, position: dict[str, Any] | None = None) -> None:
        self.stop: threading.Event = threading.Event()
        self.window: datetime.timedelta = datetime.timedelta(hours=1)
        self.state: dict[str, Any] | None = position
        self.pending: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.context_ids: list[str] = []
        self.saved: list[tuple[dict[str, Any], str | None]] = []
        # What had been sent when each position was saved.
        self.sent_when_saved: list[set[str]] = []
        self.held: set[str] = set()
        self.failing: int = 0

    def container(self, name: str) -> None:
        _ = name

    def record(self, record: Mapping[str, Any]) -> None:
        self.pending.append(dict(record))

    def context(self, record: Mapping[str, Any]) -> None:
        self.context_ids.append(str(record["source_id"]))

    def flush(self, *, live: bool = True) -> Flushed:
        _ = live
        if self.failing:
            self.failing -= 1
            return Flushed(error="ServerError: 503")
        ready = [r for r in self.pending if r["source_id"] not in self.held]
        self.pending = [r for r in self.pending if r["source_id"] in self.held]
        self.sent.extend(ready)
        return Flushed(
            settled={str(r["source_id"]) for r in ready},
            held={str(r["source_id"]) for r in self.pending},
        )

    def position(self, instance: str) -> dict[str, Any] | None:
        assert instance == GUILD
        return self.state

    def save_position(
        self,
        instance: str,
        position: dict[str, Any],
        *,
        last_record_at: datetime.datetime | None = None,
        error: str | None = None,
    ) -> bool:
        assert instance == GUILD
        _ = last_record_at
        self.state = position
        self.saved.append((position, error))
        self.sent_when_saved.append(set(self.ids()))
        return True

    def ids(self) -> list[str]:
        return [str(r["source_id"]) for r in self.sent]

    def cursors(self) -> dict[str, str]:
        assert self.state is not None
        return dict(self.state["messages"])


Step = dict[str, Any] | Callable[[], None] | str


class Script:
    """Live events from a list: dispatches, actions between them, `"overflow"`.

    When the list runs out, it asks the follower to stop.
    """

    def __init__(self, sink: FakeSink, steps: Iterable[Step]) -> None:
        self.sink: FakeSink = sink
        self.steps: list[Step] = list(steps)
        self.overflow: bool = False
        # What was waiting in the queue when it overflowed.
        self.queued: list[dict[str, Any]] = []
        self.fail: str | None = None
        self.closed: bool = False

    def start(self) -> None:
        pass

    def get(self, timeout: float) -> dict[str, Any] | None:
        _ = timeout
        while self.steps:
            step = self.steps.pop(0)
            if step == "overflow":
                self.overflow = True
                return None
            if callable(step):
                step()
                continue
            assert isinstance(step, dict)
            return step
        self.sink.stop.set()
        return None

    def take_overflow(self) -> bool:
        overflow, self.overflow = self.overflow, False
        return overflow

    def drain(self) -> list[dict[str, Any]]:
        drained, self.queued = self.queued, []
        return drained

    def failure(self) -> str | None:
        return self.fail

    def close(self) -> None:
        self.closed = True


READY: dict[str, Any] = {"t": "READY", "d": {"session_id": "s"}}


def ev(kind: str, **data: Any) -> dict[str, Any]:
    return {"t": kind, "d": {"guild_id": GUILD, **data}}


def created(message: dict[str, Any]) -> dict[str, Any]:
    return ev("MESSAGE_CREATE", **message)


class Clock:
    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now


def follow(
    fake: FakeDiscord,
    sink: FakeSink,
    steps: Iterable[Step],
    *,
    channel_ids: set[str] | None = None,
    clock: Clock | None = None,
) -> tuple[Follower, Script]:
    script = Script(sink, steps)
    with client_for(fake) as client:
        follower = Follower(
            client,
            GUILD,
            sink,
            script,
            channel_ids=channel_ids,
            clock=clock or Clock(),
        )
        follower.run()
    return follower, script


def test_first_run_reads_the_history_then_follows_live() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    live = msg(9, "200", MAYA, "after the catch-up")
    elsewhere = msg(51, "230", MAYA, "voice chat, not selected")
    follower, script = follow(
        fake,
        sink,
        [READY, created(live), created(elsewhere)],
        channel_ids={"200", "210"},
    )

    # dev's history (8 is a system notice), its thread, random; then live.
    assert sink.ids() == [*map(str, range(1, 8)), "20", "30", "9"]
    assert sink.cursors() == {"200": "9", "210": "30", "300": "20"}
    assert follower.stats.records == 10
    assert follower.stats.skipped == {"system_or_empty": 1}
    assert script.closed
    # A channel that isn't selected is looked up once, then ignored.
    assert fake.requests.count("/channels/230") == 1


def test_a_restart_carries_on_from_the_saved_position() -> None:
    fake = FakeDiscord()
    sink = FakeSink({"messages": {"200": "5", "300": "20", "210": "30"}})
    follow(fake, sink, [READY], channel_ids={"200", "210"})

    assert sink.ids() == ["6", "7"]
    # What's at or before the cursor is read again as context only.
    assert sink.context_ids == ["1", "2", "3", "4", "5", "20", "30"]
    assert sink.cursors() == {"200": "8", "210": "30", "300": "20"}


def test_catch_up_stops_at_the_newest_message_when_it_began() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    posted = msg(99, "200", JESS, "posted while catching up")

    def post_during_the_read(channel: str) -> None:
        if channel == "200" and posted not in fake.messages["200"]:
            fake.messages["200"].append(posted)

    fake.on_history = post_during_the_read
    follow(fake, sink, [READY, created(posted)], channel_ids={"200"})

    # Read once, from its live event, not also from history.
    assert sink.ids().count("99") == 1
    assert sink.ids()[-1] == "99"
    assert sink.cursors()["200"] == "99"


def test_edits_send_the_current_message_and_deletes_send_deletes() -> None:
    fake, sink = FakeDiscord(), FakeSink()

    def edit() -> None:
        message = fake.messages["200"][2]
        message["content"] = "dev 3, edited"
        message["edited_timestamp"] = "2026-11-04T11:00:00+00:00"

    follow(
        fake,
        sink,
        [
            READY,
            edit,
            ev("MESSAGE_UPDATE", id="3", channel_id="200", edited_timestamp="x"),
            # A link preview arriving isn't an edit.
            ev("MESSAGE_UPDATE", id="4", channel_id="200", embeds=[]),
            # Deleted before it could be read: its delete follows.
            ev("MESSAGE_UPDATE", id="404", channel_id="200", edited_timestamp="x"),
            ev("MESSAGE_DELETE", id="4", channel_id="200"),
            ev("MESSAGE_DELETE_BULK", ids=["5", "6"], channel_id="200"),
            ev("MESSAGE_DELETE", id="50", channel_id="230"),
        ],
        channel_ids={"200"},
    )

    live = sink.sent[8:]
    assert [(r.get("op"), r["source_id"]) for r in live] == [
        (None, "3"),
        ("delete", "4"),
        ("delete", "5"),
        ("delete", "6"),
    ]
    assert live[0]["content"] == "dev 3, edited"
    assert live[0]["revised_at"] == "2026-11-04T11:00:00+00:00"
    assert live[1] == {
        "op": "delete",
        "source": "discord",
        "source_instance": GUILD,
        "source_id": "4",
    }


def test_a_new_thread_is_read_from_its_start() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    thread = {"id": "320", "type": 11, "name": "new-idea", "parent_id": "200"}
    elsewhere = {"id": "321", "type": 11, "name": "off-topic", "parent_id": "210"}

    def open_thread() -> None:
        fake.active.extend([thread, elsewhere])
        fake.messages["320"] = [
            msg(41, "320", JESS, "first"),
            msg(42, "320", MAYA, "second"),
        ]

    follow(
        fake,
        sink,
        [
            READY,
            open_thread,
            ev("THREAD_CREATE", **thread),
            ev("THREAD_CREATE", **elsewhere),
            created(msg(43, "320", JESS, "third")),
            created(msg(44, "321", JESS, "not followed")),
        ],
        channel_ids={"200"},
    )

    assert sink.ids()[-3:] == ["41", "42", "43"]
    assert sink.cursors()["320"] == "43"
    assert "321" not in sink.cursors()


def test_a_message_in_an_unknown_channel_is_looked_up_once() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    thread = {"id": "330", "type": 12, "name": "private", "parent_id": "200"}

    def add_thread() -> None:
        fake.active.append(thread)
        fake.messages["330"] = [msg(60, "330", MAYA, "in a private thread")]

    follow(
        fake,
        sink,
        [
            READY,
            add_thread,
            created(msg(61, "330", JESS, "joined")),
            created(msg(70, "999", JESS, "nowhere")),
            created(msg(71, "999", JESS, "still nowhere")),
        ],
        channel_ids={"200"},
    )

    assert sink.ids()[-2:] == ["61", "60"]
    assert sink.cursors()["330"] == "61"
    assert fake.requests.count("/channels/999") == 1


def test_a_held_record_keeps_the_cursor_behind_it() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()
    sink.held = {"6"}

    def later() -> None:
        clock.now += 10

    def release() -> None:
        sink.held.clear()
        clock.now += 10

    follow(fake, sink, [READY, later, ev("RESUMED"), release], clock=clock)

    held_back = [
        position["messages"]["200"]
        for position, _ in sink.saved
        if "200" in position["messages"]
    ]
    assert held_back[0] == "5"
    assert sink.cursors()["200"] == "8"


def test_dropped_events_are_caught_up_from_history() -> None:
    fake, sink = FakeDiscord(), FakeSink()

    def dropped() -> None:
        fake.messages["200"].append(msg(10, "200", JESS, "its event was dropped"))

    follower, _ = follow(
        fake,
        sink,
        [READY, dropped, "overflow", created(msg(11, "200", MAYA, "after"))],
        channel_ids={"200"},
    )

    assert sink.ids()[-2:] == ["10", "11"]
    assert follower.stats.catch_ups == 2
    assert sink.cursors()["200"] == "11"


def test_a_container_that_overflowed_while_read_is_read_again() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    script = Script(sink, [READY])
    overflowed: list[str] = []

    def overflow_once(channel: str) -> None:
        if channel == "200" and not overflowed:
            overflowed.append(channel)
            script.overflow = True

    fake.on_history = overflow_once
    with client_for(fake) as client:
        follower = Follower(
            client, GUILD, sink, script, channel_ids={"200", "210"}, clock=Clock()
        )
        follower.run()

    # The first pass saved no cursor for dev: events were dropped while it was
    # read. The threads and channels read after that were fine.
    first = sink.saved[0][0]["messages"]
    assert "200" not in first
    assert first == {"210": "30", "300": "20"}
    assert follower.stats.catch_ups == 2
    assert sink.cursors()["200"] == "8"


def test_a_failed_catch_up_holds_the_cursor_until_it_is_read() -> None:
    fake = FakeDiscord()
    sink = FakeSink({"messages": {"200": "5", "300": "20"}})
    clock = Clock()
    # Dev's history can't be read at first: the client gives up after six tries.
    fake.failures["/channels/200/messages?newest"] = 6
    sink.held = {"6"}

    def later() -> None:
        clock.now += 61

    def release() -> None:
        sink.held.clear()
        clock.now += 61

    follow(
        fake,
        sink,
        [
            READY,
            created(msg(10, "200", JESS, "live")),
            later,
            ev("RESUMED"),  # the live message is settled; dev's history is read again
            later,
            ev("RESUMED"),  # a flush while message 6 is still held
            release,
        ],
        channel_ids={"200"},
        clock=clock,
    )

    # Neither the live message nor what was settled around the held one moves
    # the cursor past message 6 until it's sent.
    for (position, _), sent in zip(sink.saved, sink.sent_when_saved, strict=True):
        assert int(position["messages"]["200"]) < 6 or {"6", "7"} <= sent
    assert sink.cursors()["200"] == "10"
    assert sink.ids().count("6") == 1 and "7" in sink.ids()
    assert sink.saved[0][1] is not None and "#dev" in sink.saved[0][1]


def test_an_incomplete_thread_listing_is_tried_again() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()
    fake.failures["/channels/220/threads/archived/public"] = 6

    def later() -> None:
        clock.now += 61

    follow(fake, sink, [READY, later, ev("RESUMED")], channel_ids={"220"}, clock=clock)

    assert sink.ids() == ["40"]
    assert sink.saved[0][1] is not None and "threads of 220" in sink.saved[0][1]
    assert sink.saved[-1] == ({"messages": {"310": "40"}}, None)


def test_a_long_honcho_outage_backs_off_without_overflowing() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    sink.failing = 1
    with client_for(fake) as client:
        follower = Follower(client, GUILD, sink, Script(sink, []), clock=Clock())
        follower._failures = 5_000  # pyright: ignore[reportPrivateUsage]
        assert not follower._flush()  # pyright: ignore[reportPrivateUsage]


def test_an_overflow_empties_the_queue_keeping_deletes_and_new_threads() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    script = Script(sink, [READY])
    thread = {"id": "320", "type": 11, "name": "new", "parent_id": "200"}

    def overflow() -> None:
        fake.active.append(thread)
        fake.messages["320"] = [msg(41, "320", JESS, "in the new thread")]
        script.queued = [
            ev("MESSAGE_DELETE", id="4", channel_id="200"),
            created(msg(99, "200", JESS, "comes back with the history, or not")),
            ev("MESSAGE_UPDATE", id="3", channel_id="200", edited_timestamp="x"),
            ev("THREAD_CREATE", **thread),
        ]
        script.overflow = True

    script.steps.append(overflow)
    script.steps.append(ev("RESUMED"))
    with client_for(fake) as client:
        follower = Follower(
            client, GUILD, sink, script, channel_ids={"200"}, clock=Clock()
        )
        follower.run()

    assert script.queued == []
    after = sink.ids()[8:]
    assert after == ["4", "41"]
    assert sink.sent[8]["op"] == "delete"
    assert follower.stats.catch_ups == 2


def test_an_overflow_still_follows_a_thread_its_queued_messages_name() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    script = Script(sink, [READY])
    hidden = {"id": "330", "type": 12, "name": "private", "parent_id": "200"}

    def overflow() -> None:
        fake.unlisted.append(hidden)
        fake.messages["330"] = [msg(60, "330", MAYA, "in a thread no listing shows")]
        script.queued = [created(msg(60, "330", MAYA, "in a thread no listing shows"))]
        script.overflow = True

    script.steps.append(overflow)
    script.steps.append(ev("RESUMED"))
    with client_for(fake) as client:
        Follower(client, GUILD, sink, script, channel_ids={"200"}, clock=Clock()).run()

    assert sink.ids()[-1] == "60"
    assert sink.cursors()["330"] == "60"


def test_a_thread_seen_before_its_channel_is_followed_once_the_channel_is() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    hidden = {"id": "330", "type": 12, "name": "private", "parent_id": "200"}
    fake.unlisted.append(hidden)
    fake.messages["330"] = [msg(60, "330", MAYA, "in a thread no listing shows")]
    # Behind before the first listing: its channel isn't followed yet.
    script = Script(sink, [])
    script.queued = [created(msg(60, "330", MAYA, "in a thread no listing shows"))]
    script.overflow = True
    with client_for(fake) as client:
        Follower(client, GUILD, sink, script, channel_ids={"200"}, clock=Clock()).run()

    assert "60" in sink.ids()
    assert sink.cursors()["330"] == "60"
    assert fake.requests.count("/channels/330") == 1


def test_a_channel_it_couldnt_look_up_is_tried_once_a_minute() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()
    hidden = {"id": "330", "type": 12, "name": "private", "parent_id": "200"}
    fake.unlisted.append(hidden)
    fake.messages["330"] = [
        msg(60, "330", MAYA, "first"),
        msg(61, "330", JESS, "second"),
    ]
    fake.failures["/channels/330"] = 12  # two lookups' worth of retries

    def later() -> None:
        clock.now += 61

    lookups: list[int] = []

    def count() -> None:
        lookups.append(fake.requests.count("/channels/330"))

    follow(
        fake,
        sink,
        [
            READY,
            created(msg(60, "330", MAYA, "first")),
            created(msg(61, "330", JESS, "second")),
            count,
            later,
            ev("RESUMED"),
            later,
            ev("RESUMED"),
        ],
        channel_ids={"200"},
        clock=clock,
    )

    # Six failed tries for the first message and none for the second; six more
    # a minute later; then one that finds it and reads its history.
    assert lookups == [6]
    assert fake.requests.count("/channels/330") == 13
    assert sorted(sink.ids()[-2:]) == ["60", "61"]
    assert sink.cursors()["330"] == "61"


def test_lookups_tried_again_take_turns_with_live_events() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()
    for cid in ("330", "331"):
        fake.failures[f"/channels/{cid}"] = 12
    retried: list[int] = []

    def later() -> None:
        clock.now += 61

    def before_the_live_message() -> None:
        retried.append(fake.requests.count("/channels/331"))

    follow(
        fake,
        sink,
        [
            READY,
            created(msg(60, "330", MAYA, "unknown")),
            created(msg(61, "331", MAYA, "unknown too")),
            later,
            ev("RESUMED"),
            before_the_live_message,
            created(msg(12, "210", JESS, "live")),
        ],
        channel_ids={"210"},
        clock=clock,
    )

    # One channel was tried again, then the live message was handled before
    # the other's turn.
    assert retried == [6]
    assert sink.ids()[-1] == "12"


def test_losing_every_selected_channel_sends_and_saves_before_stopping() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()

    def deleted() -> None:
        fake.channels = [c for c in fake.channels if c["id"] != "210"]

    script = Script(
        sink, [READY, created(msg(31, "210", JESS, "last words")), deleted, READY]
    )
    with client_for(fake) as client:
        follower = Follower(
            client, GUILD, sink, script, channel_ids={"210"}, clock=clock
        )
        with pytest.raises(SystemExit, match="Not exportable"):
            follower.run()

    assert sink.ids() == ["30", "31"]
    position, error = sink.saved[-1]
    assert position == {"messages": {"210": "31"}}
    assert error is not None and "Not exportable" in error


def test_a_channel_found_again_is_read_before_its_cursor_moves() -> None:
    fake = FakeDiscord()
    sink = FakeSink({"messages": {"200": "8", "300": "20", "330": "60"}})
    clock = Clock()
    hidden = {"id": "330", "type": 12, "name": "private", "parent_id": "200"}
    fake.unlisted.append(hidden)
    fake.messages["330"] = [
        msg(60, "330", JESS, "seen"),
        msg(61, "330", MAYA, "posted while it was down"),
    ]

    def later() -> None:
        clock.now += 10

    follow(
        fake,
        sink,
        [READY, later, created(msg(62, "330", JESS, "live"))],
        channel_ids={"200"},
        clock=clock,
    )

    assert "61" in sink.ids()
    for (position, _), sent in zip(sink.saved, sink.sent_when_saved, strict=True):
        assert int(position["messages"].get("330", "0")) < 61 or "61" in sent
    assert sink.cursors()["330"] == "62"


def test_a_deleted_channel_leaves_the_rest_followed() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    later_thread = {
        "id": "340",
        "type": 11,
        "name": "while-down",
        "parent_id": "200",
        "thread_metadata": {"archive_timestamp": "2026-11-20T00:00:00+00:00"},
    }

    def while_disconnected() -> None:
        fake.channels = [c for c in fake.channels if c["id"] != "210"]
        fake.archived["200"] = [later_thread]
        fake.messages["340"] = [msg(70, "340", MAYA, "archived while it was down")]

    follower, _ = follow(
        fake,
        sink,
        [READY, while_disconnected, READY],
        channel_ids={"200", "210"},
    )

    assert "70" in sink.ids()
    assert follower.channel_ids == {"200"}


def test_a_fresh_session_catches_up_and_a_resumed_one_doesnt() -> None:
    fake, sink = FakeDiscord(), FakeSink()

    def while_disconnected() -> None:
        fake.messages["210"].append(msg(31, "210", JESS, "missed"))

    follower, _ = follow(
        fake,
        sink,
        [READY, ev("RESUMED"), while_disconnected, READY],
        channel_ids={"210"},
    )

    assert sink.ids() == ["30", "31"]
    assert follower.stats.catch_ups == 2


def test_a_lost_connection_stops_with_its_reason_saved() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    script = Script(sink, [READY])

    def lose_it() -> None:
        script.fail = "Discord refused the bot token"

    script.steps.append(lose_it)
    script.steps.append(ev("RESUMED"))
    with client_for(fake) as client:
        follower = Follower(client, GUILD, sink, script, channel_ids={"210"})
        with pytest.raises(SystemExit, match="refused the bot token"):
            follower.run()

    assert sink.saved[-1] == (
        {"messages": {"210": "30"}},
        "Discord refused the bot token",
    )


def test_records_wait_while_honcho_is_failing() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    clock = Clock()
    sink.failing = 2

    def wait() -> None:
        clock.now += 61

    follow(
        fake, sink, [READY, wait, ev("RESUMED"), wait], channel_ids={"210"}, clock=clock
    )

    assert sink.ids() == ["30"]
    assert sink.saved == [({"messages": {"210": "30"}}, None)]


def test_live_messages_use_the_nickname_they_carry() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    fake.members = []
    message = msg(12, "210", MAYA, "hi", member={"nick": "Maya R"})
    follow(fake, sink, [READY, created(message)], channel_ids={"210"})
    assert sink.sent[-1]["author"]["name"] == "Maya R"


def test_bot_records_are_handed_over_for_the_cli_to_filter() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    follow(
        fake, sink, [READY, created(msg(13, "210", BOT, "deploy"))], channel_ids={"210"}
    )
    assert sink.sent[-1]["author"]["bot"] is True


def test_a_selection_it_cant_read_stops_at_the_start() -> None:
    fake, sink = FakeDiscord(), FakeSink()
    with pytest.raises(SystemExit, match="Not exportable"):
        follow(fake, sink, [READY], channel_ids={"404"})


def test_follow_needs_the_cli_follow_sink(tmp_path: Any) -> None:
    token = tmp_path / "token"
    token.write_text("t0ken")
    base = ["--token-file", str(token), "--guild", GUILD, "--follow"]

    class BatchSink:
        def container(self, name: str) -> None:
            _ = name

        def record(self, record: Mapping[str, Any]) -> None:
            _ = record

    with pytest.raises(SystemExit, match="needs `honcho ingest from discord --follow`"):
        SOURCE.main(base, BatchSink())
    with pytest.raises(SystemExit, match="don't go together"):
        SOURCE.main([*base, "--limit-per-container", "3"], FakeSink())
    assert SOURCE.follows


def test_the_gateway_queues_only_the_dispatches_it_follows() -> None:
    events = GatewayEvents("t0ken", size=2)
    events.receive(json.dumps({"op": 0, "t": "MESSAGE_CREATE", "d": {"id": "1"}}))
    events.receive({"op": 0, "t": "TYPING_START", "d": {}})
    events.receive(json.dumps({"op": 11}))
    events.receive("not json")
    assert events.get(0) == {"t": "MESSAGE_CREATE", "d": {"id": "1"}}
    assert events.get(0) is None
    assert not events.take_overflow()

    for i in range(3):
        events.receive({"op": 0, "t": "MESSAGE_DELETE", "d": {"id": str(i)}})
    assert events.take_overflow()
    assert not events.take_overflow()


def test_snowflakes_tell_time() -> None:
    # Discord's documented example.
    assert snowflake_time(175928847299117063) == datetime.datetime(
        2016, 4, 30, 11, 18, 25, 796000, tzinfo=datetime.UTC
    )
    hour_before = snowflake_before(175928847299117063, datetime.timedelta(hours=1))
    assert snowflake_time(hour_before) == datetime.datetime(
        2016, 4, 30, 10, 18, 25, 796000, tzinfo=datetime.UTC
    )
    assert snowflake_before(5, datetime.timedelta(hours=1)) == 0
