"""`honcho ingest from <source> --follow`: the follow sink, and the command around it."""

from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock

import pytest
from honcho import ConflictError, ServerError, UnprocessableEntityError
from honcho.api_types import IngestSyncState

from honcho_cli.follow import FollowConflict, FollowSink
from honcho_cli.main import app

from .test_ingest import (  # noqa: F401  (fixtures)
    STATUS,
    _client,
    _FakeSource,
    _from,
    _ingest_response,
    cfg,
    runner,
)

T0 = datetime.datetime(2026, 11, 20, 10, 0, tzinfo=datetime.UTC)
BOT = {"id": "b1", "name": "ci", "bot": True}


def _rec(n: int, minute: int, *, bot: bool = False, **extra) -> dict:
    return {
        "source": "discord",
        "source_instance": "g1",
        "source_id": f"m{n}",
        "created_at": (T0 + datetime.timedelta(minutes=minute)).isoformat(),
        "container": {"id": "c1", "name": "dev"},
        "author": BOT if bot else {"id": f"u{n}", "name": f"Person {n}"},
        "content": f"message {n}",
        **extra,
    }


def _delete(n: int) -> dict:
    return {"op": "delete", "source": "discord", "source_instance": "g1", "source_id": f"m{n}"}


class _Clock:
    def __init__(self) -> None:
        self.now = T0 + datetime.timedelta(hours=1)

    def __call__(self) -> datetime.datetime:
        return self.now


def _sink(*, bots: str = "interacted", ingest=None, **client_methods):
    ingest = ingest or MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    client = _client(ingest=ingest, **client_methods)
    clock = _Clock()
    sink = FollowSink(client, "discord", derive="auto", bots=bots, now=clock)
    return sink, ingest, clock


def _sent(ingest: MagicMock) -> list[str]:
    return [r.source_id for (chunk,), _ in ingest.call_args_list for r in chunk]


def test_a_person_is_sent_at_once_and_a_bot_waits_for_an_answer():
    sink, ingest, _ = _sink()
    sink.record(_rec(0, 0))
    sink.record(_rec(1, 1, bot=True))

    first = sink.flush()
    assert _sent(ingest) == ["m0"]
    assert (first.settled, first.held, first.error) == ({"m0"}, {"m1"}, None)

    sink.record(_rec(2, 2, metadata={"reply_to": "m1"}))
    second = sink.flush()
    assert _sent(ingest) == ["m0", "m1", "m2"]
    assert (second.settled, second.held) == ({"m1", "m2"}, set())
    assert all(kwargs == {"derive": "auto"} for _, kwargs in ingest.call_args_list)


def test_an_unanswered_bot_record_is_dropped_once_the_window_has_passed():
    # By source time: a newer record in its container is more than the window newer.
    sink, ingest, _ = _sink()
    sink.record(_rec(1, 0, bot=True))
    assert sink.flush(live=False).held == {"m1"}
    sink.record(_rec(2, 61))
    dropped = sink.flush(live=False)
    assert _sent(ingest) == ["m2"]
    assert dropped.settled == {"m1", "m2"}
    assert sink.held_back == 1

    # While following live, also after waiting the window in wall time; reading
    # history, not.
    sink, ingest, clock = _sink()
    sink.record(_rec(1, 0, bot=True))
    clock.now += datetime.timedelta(minutes=61)
    assert sink.flush(live=False).held == {"m1"}
    assert sink.flush(live=True).settled == {"m1"}
    assert _sent(ingest) == []


def test_context_is_never_sent_but_counts_for_the_bot_filter():
    sink, ingest, _ = _sink()
    sink.context(_rec(0, 0))
    sink.record(_rec(1, 1, bot=True, metadata={"reply_to": "m0"}))
    flushed = sink.flush()
    assert _sent(ingest) == ["m1"]
    assert flushed.settled == {"m1"}


def test_deletes_pass_straight_through():
    sink, ingest, _ = _sink(bots="none")
    sink.record(_delete(7))
    sink.record(_rec(1, 0, bot=True))
    flushed = sink.flush()
    assert _sent(ingest) == ["m7"]
    assert flushed.settled == {"m7", "m1"}


def test_a_failure_that_may_pass_keeps_the_records_pending():
    calls = {"n": 0}

    def flaky(chunk, derive):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerError("unavailable", status=503)
        return _ingest_response(len(chunk))

    sink, ingest, _ = _sink(ingest=MagicMock(side_effect=flaky))
    sink.record(_rec(0, 0))
    sink.record(_delete(5))

    failed = sink.flush()
    assert failed.error is not None and "unavailable" in failed.error
    assert failed.settled == set()

    retried = sink.flush()
    assert retried.error is None
    assert retried.settled == {"m0", "m5"}
    assert sink.sent == 2


def test_a_bot_reply_that_qualified_is_still_sent_after_a_failed_send():
    calls = {"n": 0}

    def flaky(chunk, derive):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ServerError("unavailable", status=503)
        return _ingest_response(len(chunk))

    sink, ingest, _ = _sink(ingest=MagicMock(side_effect=flaky))
    sink.record(_rec(0, 0))
    sink.flush()
    # The bot answers the person; a newer record is past the window, so the
    # person's record is pruned when the send fails.
    sink.record(_rec(1, 1, bot=True, metadata={"reply_to": "m0"}))
    sink.record(_rec(2, 121))
    assert sink.flush().error is not None

    retried = sink.flush()
    assert retried.settled == {"m1", "m2"}
    (last_chunk,), _ = ingest.call_args
    assert [r.source_id for r in last_chunk] == ["m1", "m2"]
    assert sink.held_back == 0


def test_any_other_failure_is_raised_and_keeps_the_records_pending():
    sink, _, _ = _sink(ingest=MagicMock(side_effect=UnprocessableEntityError("bad")))
    sink.record(_rec(0, 0))
    with pytest.raises(UnprocessableEntityError):
        sink.flush()


def _state(revision: str) -> IngestSyncState:
    return IngestSyncState(
        source="discord",
        source_instance="g1",
        last_synced_at=T0,
        position={"messages": {"c1": "5"}},
        revision=revision,
    )


def test_a_position_is_saved_with_the_revision_it_was_read_at():
    save = MagicMock(side_effect=[_state("r2"), _state("r3")])
    sink, _, _ = _sink(
        ingest_sync_state=MagicMock(return_value=_state("r1")), ingest_save_sync_state=save
    )
    assert sink.position("g1") == {"messages": {"c1": "5"}}

    assert sink.save_position("g1", {"messages": {"c1": "6"}}, last_record_at=T0)
    assert sink.save_position("g1", {"messages": {"c1": "7"}}, error="#dev: 403")
    assert [c.kwargs["revision"] for c in save.call_args_list] == ["r1", "r2"]
    assert save.call_args_list[1].kwargs["error"] == "#dev: 403"


def test_a_first_save_has_no_revision_and_a_race_is_a_conflict():
    save = MagicMock(side_effect=ConflictError("changed"))
    sink, _, _ = _sink(
        ingest_sync_state=MagicMock(return_value=None), ingest_save_sync_state=save
    )
    assert sink.position("g1") is None
    with pytest.raises(FollowConflict, match="another runner"):
        sink.save_position("g1", {"messages": {}})
    assert save.call_args.kwargs["revision"] is None


def test_a_save_that_may_work_later_says_so():
    sink, _, _ = _sink(
        ingest_sync_state=MagicMock(return_value=None),
        ingest_save_sync_state=MagicMock(side_effect=ServerError("down", status=502)),
    )
    sink.position("g1")
    assert sink.save_position("g1", {"messages": {}}) is False


# --- ingest from --follow ----------------------------------------------------------------


class _FollowingSource(_FakeSource):
    follows = True

    def __init__(self, records: list[dict], *, save_error: Exception | None = None):
        super().__init__({"#dev": records})
        self.position: dict | None = None
        self.save_error = save_error

    def main(self, argv, sink):
        self.argv = argv
        assert "--follow" in argv
        self.position = sink.position("g1")
        sink.container("#dev")
        for record in self.containers["#dev"]:
            sink.record(record)
        flushed = sink.flush()
        sink.save_position("g1", {"messages": {"c1": max(flushed.settled)}})
        return 0


def _follow_client(*, save=None):
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    client = _client(
        ingest=ingest,
        ingest_status=MagicMock(return_value=STATUS),
        ingest_sync_state=MagicMock(return_value=_state("r1")),
        ingest_save_sync_state=save or MagicMock(return_value=_state("r2")),
    )
    return client, ingest


def test_follow_runs_the_source_with_a_follow_sink_deriving_auto(cfg, runner):
    source = _FollowingSource([_rec(0, 0), _rec(1, 1)])
    client, ingest = _follow_client()

    result = _from(runner, ["fake", "--guild", "1", "--follow", "-w", "ws1"], client, fake=source)

    assert result.exit_code == 0, result.stderr
    assert source.argv == ["--guild", "1", "--follow"]
    assert source.position == {"messages": {"c1": "5"}}
    assert _sent(ingest) == ["m0", "m1"]
    assert all(kwargs == {"derive": "auto"} for _, kwargs in ingest.call_args_list)
    client.ingest_save_sync_state.assert_called_once()
    summary = json.loads(result.stdout)
    assert (summary["records"], summary["created"], summary["derive"]) == (2, 2, "auto")


def test_follow_keeps_an_explicit_derive(cfg, runner):
    client, ingest = _follow_client()
    result = _from(
        runner,
        ["fake", "--follow", "--derive", "later", "-w", "ws1"],
        client,
        fake=_FollowingSource([_rec(0, 0)]),
    )
    assert result.exit_code == 0, result.stderr
    assert ingest.call_args.kwargs == {"derive": "later"}


def test_follow_is_refused_for_a_source_that_cant(cfg, runner):
    client, ingest = _follow_client()
    result = _from(runner, ["fake", "--follow", "-w", "ws1"], client, fake=_FakeSource({}))
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "CANT_FOLLOW"
    ingest.assert_not_called()


def test_follow_wants_a_bot_window_of_a_minute_or_more(cfg, runner):
    client, _ = _follow_client()
    result = _from(
        runner,
        ["fake", "--follow", "--bot-window", "0", "-w", "ws1"],
        client,
        fake=_FollowingSource([]),
    )
    assert result.exit_code == 1
    assert "--bot-window" in json.loads(result.stderr)["error"]["message"]


def test_follow_stops_with_a_clear_error_when_another_runner_saved(cfg, runner):
    client, _ = _follow_client(save=MagicMock(side_effect=ConflictError("changed")))
    result = _from(
        runner, ["fake", "--follow", "-w", "ws1"], client, fake=_FollowingSource([_rec(0, 0)])
    )
    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "SYNC_CONFLICT"
    assert "another runner" in error["message"]


def test_follow_names_the_container_of_an_invalid_record(cfg, runner):
    client, _ = _follow_client()
    result = _from(
        runner,
        ["fake", "--follow", "-w", "ws1"],
        client,
        fake=_FollowingSource([{"source": "discord"}]),
    )
    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "INVALID_RECORD"
    assert error["details"]["container"] == "#dev"


def test_follow_help_mentions_it(cfg, runner):
    result = runner.invoke(app, ["ingest", "from", "--help"])
    assert result.exit_code == 0
    assert "--follow" in result.stdout
    assert "--bot-window" in result.stdout
