"""Following repositories: polls since a watermark, commits since a head, restarts."""

import datetime
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from honcho_github.export import commits_between
from honcho_github.follow import GitHubFollower
from honcho_github.mapping import Repo
from honcho_github.source import SOURCE

from .fakes import SAM, FakeGitHub, client_for, item

REPOS = ["acme/web", "acme/api"]
T0 = datetime.datetime(2026, 11, 20, 12, 0, tzinfo=datetime.UTC)


class Steps(threading.Event):
    """The follower's stop: each wait between polls runs the next action; when
    they run out, it stops."""

    def __init__(self, actions: Iterable[Callable[[], None]]) -> None:
        super().__init__()
        self.actions: list[Callable[[], None]] = list(actions)

    def wait(self, timeout: float | None = None) -> bool:
        _ = timeout
        if self.actions:
            self.actions.pop(0)()
            return False
        self.set()
        return True


@dataclass
class Flushed:
    settled: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    error: str | None = None


class FakeSink:
    """The CLI's follow sink, with the sending faked."""

    def __init__(self, actions: Iterable[Callable[[], None]] = ()) -> None:
        self.steps: Steps = Steps(actions)
        self.stop: threading.Event = self.steps
        self.state: dict[str, Any] | None = None
        self.pending: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.saved: list[tuple[dict[str, Any], str | None]] = []
        self.held: set[str] = set()
        self.failing: int = 0

    def container(self, name: str) -> None:
        _ = name

    def record(self, record: Mapping[str, Any]) -> None:
        self.pending.append(dict(record))

    def flush(self, *, live: bool = True) -> Flushed:
        assert live
        if self.failing:
            self.failing -= 1
            return Flushed(error="ServerError: 503")
        ready = [r for r in self.pending if r["source_id"] not in self.held]
        self.pending = [r for r in self.pending if r["source_id"] in self.held]
        self.sent.extend(ready)
        return Flushed(
            settled={r["source_id"] for r in ready},
            held={r["source_id"] for r in self.pending},
        )

    def position(self, instance: str) -> dict[str, Any] | None:
        assert instance == "github-com"
        return self.state

    def save_position(
        self,
        instance: str,
        position: dict[str, Any],
        *,
        last_record_at: datetime.datetime | None = None,
        error: str | None = None,
    ) -> bool:
        assert instance == "github-com"
        _ = last_record_at
        self.state = position
        self.saved.append((position, error))
        return True

    def ids(self) -> list[str]:
        return [str(r["source_id"]) for r in self.sent]


def follow(
    fake: FakeGitHub,
    sink: FakeSink,
    *,
    repos: list[str] | None = None,
    times: Iterable[datetime.datetime] = (),
    since: str | None = None,
) -> GitHubFollower:
    clock = iter([*times, *(T0 + datetime.timedelta(hours=h) for h in range(10, 20))])
    with client_for(fake) as client:
        follower = GitHubFollower(
            client,
            repos or REPOS,
            [],
            sink,
            since=since,
            now=lambda: next(clock),
        )
        follower.run()
    return follower


def _commit(oid: str, at: str, message: str) -> dict[str, Any]:
    return {
        "id": f"C_{oid}",
        "oid": oid,
        "url": "u",
        "committedDate": at,
        "message": message,
        "author": {"name": "Sam", "user": {"login": "sam", "databaseId": 22}},
        "associatedPullRequests": {"nodes": []},
    }


def test_the_first_poll_reads_everything_and_saves_watermark_and_head() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    follower = follow(fake, sink, times=[T0])

    ids = sink.ids()
    assert {"I_1", "IC_1", "IC_2", "PR_12", "PR_13", "C_1", "C_old"} <= set(ids)
    assert len(ids) == len(set(ids))
    assert sink.state == {
        "repos": {
            "acme/api": {"updated_since": "2026-11-20T11:59:00Z", "head": None},
            "acme/web": {
                "updated_since": "2026-11-20T11:59:00Z",
                "head": "abcdef0123456789",
            },
        }
    }
    assert follower.stats.polls == 1


def test_a_later_poll_sends_only_what_changed() -> None:
    fake = FakeGitHub()

    def changes() -> None:
        issue = fake.issues[0]
        issue["updatedAt"] = "2026-11-20T12:30:00Z"
        # An edit, and a new comment, on the issue; a new commit on main.
        fake.issue_comments_rest[0] = {
            **fake.issue_comments_rest[0],
            "body": "Fixed by @newbie.",
            "lastEditedAt": "2026-11-20T12:20:00Z",
        }
        fake.issue_comments_rest.append(
            item("IC_3", "2026-11-20T12:25:00Z", SAM, "Confirmed.")
        )
        fake.commits.insert(
            0, _commit("feedface", "2026-11-20T12:10:00Z", "fix: the crash")
        )

    sink = FakeSink([changes])
    follow(fake, sink, times=[T0, T0 + datetime.timedelta(minutes=35)])

    # Oldest first: the edited comment, the commit, the new comment.
    first = len(sink.ids()) - 3
    assert sink.ids()[first:] == ["IC_2", "C_feedface", "IC_3"]
    edited = sink.sent[first]
    assert edited["revised_at"] == "2026-11-20T12:20:00Z"
    assert "Fixed by @newbie." in edited["content"]
    assert sink.state is not None
    assert sink.state["repos"]["acme/web"] == {
        "updated_since": "2026-11-20T12:34:00Z",
        "head": "feedface",
    }
    # The pull request wasn't updated since the watermark: its review thread
    # isn't read again.
    assert [v for _, v in fake.queries if v.get("id") == "T_1"] == [
        {"id": "T_1", "after": "t1"}
    ]


def test_a_restart_carries_on_from_the_saved_position() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    sink.state = {
        "repos": {
            "acme/web": {
                "updated_since": "2026-11-05T12:00:00Z",
                "head": "1111111111111111",
            },
        }
    }
    follow(fake, sink, repos=["acme/web"], times=[T0])
    # Updated since: the pull request PR_12 (and its thread), not the issue or
    # PR_13; commits back to the saved head.
    assert "I_1" not in sink.ids()
    assert "PR_13" not in sink.ids()
    assert "PR_12" in sink.ids()
    assert [i for i in sink.ids() if i.startswith("C_")] == ["C_1"]


def test_a_merged_branchs_older_commits_are_read() -> None:
    fake = FakeGitHub()
    sink = FakeSink()
    sink.state = {
        "repos": {
            "acme/web": {
                "updated_since": "2026-11-19T00:00:00Z",
                "head": "abcdef0123456789",
            }
        }
    }
    feature = _commit("f1f1f1", "2026-11-01T09:00:00Z", "feat: started long ago")
    merge = _commit("m0m0m0", "2026-11-20T11:00:00Z", "Merge branch 'feature'")
    merge["message"] = "feat: land the feature"
    # In history order the feature's commit comes after the old head: stopping
    # there would miss it. Compare lists what the merge brought.
    fake.commits = [merge, *fake.commits[:1], feature, *fake.commits[1:]]
    fake.compared[("abcdef0123456789", "m0m0m0")] = ["f1f1f1", "m0m0m0"]

    follow(fake, sink, repos=["acme/web"], times=[T0])

    commits = [i for i in sink.ids() if i.startswith("C_")]
    assert sorted(commits) == ["C_f1f1f1", "C_m0m0m0"]
    assert sink.state is not None
    assert sink.state["repos"]["acme/web"]["head"] == "m0m0m0"


def test_a_rewritten_head_reads_the_newest_commits_again() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    sink.state = {
        "repos": {"acme/web": {"updated_since": "2026-11-19T00:00:00Z", "head": "gone"}}
    }
    follow(fake, sink, repos=["acme/web"], times=[T0])
    assert sorted(i for i in sink.ids() if i.startswith("C_")) == ["C_1", "C_old"]


def test_compare_keeps_the_newest_commits_when_there_are_too_many() -> None:
    fake = FakeGitHub()
    fake.commits = [
        _commit(f"{n:06x}", f"2026-11-{1 + n // 24:02d}T{n % 24:02d}:00:00Z", f"c{n}")
        for n in reversed(range(250))
    ]
    with client_for(fake) as client:
        repo = Repo.from_node(fake.repos["web"])
        found = commits_between(client, repo, f"{0:06x}", f"{249:06x}", limit=150)
    assert found is not None
    commits, cut = found
    assert cut
    assert [c["oid"] for c in commits] == [
        f"{n:06x}" for n in reversed(range(100, 250))
    ]


def test_a_failed_send_mid_poll_waits_for_the_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("honcho_github.follow.FLUSH_RECORDS", 2)
    fake, sink = FakeGitHub(), FakeSink()
    sink.failing = 1
    flushes: list[int] = []
    real = sink.flush

    def counted(*, live: bool = True) -> Flushed:
        flushes.append(1)
        return real(live=live)

    sink.flush = counted  # type: ignore[method-assign]
    follow(fake, sink, repos=["acme/web"], times=[T0])
    # The failed one mid-poll, one after the poll, and the last one.
    assert len(flushes) == 3
    assert len(sink.ids()) == len(set(sink.ids()))


def test_a_long_outage_backs_off_without_overflowing() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    sink.failing = 1
    with client_for(fake) as client:
        follower = GitHubFollower(client, ["acme/web"], [], sink, now=lambda: T0)
        follower._failures = 5_000  # pyright: ignore[reportPrivateUsage]
        follower.run()
    assert sink.saved


def test_a_held_record_keeps_the_watermark_back_until_it_settles() -> None:
    fake = FakeGitHub()
    sink = FakeSink()

    def release() -> None:
        sink.held.clear()

    sink.steps.actions.append(release)
    sink.held = {"PC_1"}
    follow(
        fake,
        sink,
        repos=["acme/web"],
        times=[T0, T0 + datetime.timedelta(minutes=5)],
    )

    assert sink.saved[0][0] == {"repos": {}}
    assert sink.state is not None
    assert sink.state["repos"]["acme/web"]["updated_since"] == "2026-11-20T12:04:00Z"
    # Read again by the second poll, but sent once.
    assert sink.ids().count("PR_12") == 1
    assert sink.ids().count("PC_1") == 1


def test_records_wait_while_honcho_is_failing() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    sink.failing = 1
    follow(fake, sink, repos=["acme/web"], times=[T0])
    assert len(sink.saved) == 1
    assert len(sink.ids()) == len(set(sink.ids()))
    assert "acme/web" in sink.saved[0][0]["repos"]


def test_a_repository_it_cant_read_is_saved_as_the_error() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    fake.forbidden = {"api"}
    follow(fake, sink, times=[T0])
    position, error = sink.saved[-1]
    assert "acme/api" not in position["repos"]
    assert "acme/web" in position["repos"]
    assert error is not None and error.startswith("acme/api: ")


def test_since_sets_where_a_first_poll_starts() -> None:
    fake, sink = FakeGitHub(), FakeSink()
    follow(fake, sink, repos=["acme/web"], times=[T0], since="2026-11-05T12:00:00Z")
    assert "I_1" not in sink.ids()
    assert [i for i in sink.ids() if i.startswith("C_")] == ["C_1"]


def test_a_bad_selection_stops_at_the_start() -> None:
    with pytest.raises(SystemExit, match="doesn't exist"):
        follow(FakeGitHub(), FakeSink(), repos=["acme/nope"], times=[T0])


def test_follow_needs_the_cli_follow_sink_and_a_sane_interval(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("t0ken")
    base = ["--token-file", str(token), "--repo", "acme/web", "--follow"]

    class BatchSink:
        def container(self, name: str) -> None:
            _ = name

        def record(self, record: Mapping[str, Any]) -> None:
            _ = record

    with pytest.raises(SystemExit, match="needs `honcho ingest from github --follow`"):
        SOURCE.main(base, BatchSink())
    with pytest.raises(SystemExit, match="at least 30 seconds"):
        SOURCE.main([*base, "--interval", "5"], FakeSink())
    assert SOURCE.follows
