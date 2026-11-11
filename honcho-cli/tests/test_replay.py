"""`honcho workspace replay`: session selection, message replay, resume, and credentials."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import honcho
import httpx
import pytest
from honcho.api_types import SessionPeerConfig
from honcho_cli import replay as replay_lib
from honcho_cli.common import get_endpoint_client
from honcho_cli.main import app
from typer.testing import CliRunner

T0 = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)


def _message(i: int, peer_id: str = "alice", *, day: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        peer_id=peer_id,
        content=f"message {i}",
        created_at=T0 + timedelta(days=day, minutes=i),
        metadata={"i": i} if i == 0 else {},
    )


class FakePage(list):
    """The slice of the SDK's ``SyncPage`` that replay reads."""

    def __init__(self, items, total: int | None = None) -> None:
        super().__init__(items)
        self.items = list(items)
        self._raw_items = list(items)
        self.total = len(self.items) if total is None else total

    def get_next_page(self) -> None:
        return None


class FakeSession:
    def __init__(self, sid: str, messages=(), *, members=("alice",), created_at=T0, **attrs) -> None:
        self.id = sid
        self.stored = list(messages)
        self.members = list(members)
        self.created_at = created_at
        self.metadata = attrs.get("metadata", {})
        self.configuration = attrs.get("configuration")
        self.scopes = attrs.get("scopes")
        self.calls: list[tuple] = []

    def messages(self, *, size: int = 50, reverse: bool = False) -> FakePage:
        ordered = self.stored[::-1] if reverse else self.stored
        return FakePage(ordered[:size] if reverse else ordered, total=len(self.stored))

    def get_peer_configuration(self, peer_id: str) -> SessionPeerConfig:
        return SessionPeerConfig(observe_others=peer_id == "alice")

    def add_messages(self, batch) -> None:
        self.calls.append(("add_messages", len(batch)))
        self.stored.extend(
            SimpleNamespace(
                peer_id=m.peer_id, content=m.content, created_at=m.created_at, metadata=m.metadata or {}
            )
            for m in batch
        )

    def remove_peers(self, peers) -> None:
        self.calls.append(("remove_peers", peers))


class FakeHoncho:
    """An in-memory workspace exposing the SDK calls replay makes, with a write log."""

    def __init__(self, workspace_id: str, sessions=(), peers=(), conclusions: int = 0) -> None:
        self.workspace_id = workspace_id
        self.conclusions = SimpleNamespace(list=lambda size=50: FakePage([], total=conclusions))
        self.base_url = f"http://{workspace_id}.test"
        self.by_id = {s.id: s for s in sessions}
        self.peer_list = list(peers)
        self.writes: list[tuple] = []
        self._http = SimpleNamespace(get=self._get)
        self._exists = bool(self.by_id or self.peer_list)
        self._workspace_ensured = False

    def _ensure(self) -> None:
        """The SDK's get-or-create on the workspace, which runs before every read."""
        if not self._workspace_ensured and not self._exists:
            self.writes.append(("workspace", self.workspace_id))
            self._exists = True
        self._workspace_ensured = True

    def open_session(self, sid: str) -> FakeSession:
        self._ensure()
        return self.by_id.get(sid) or FakeSession(sid)

    def get_metadata(self) -> dict:
        self._ensure()
        return {}

    def _get(self, path: str, query=None) -> dict:
        session = self.by_id[path.split("/sessions/")[1].split("/")[0]]
        return {"items": [{"id": peer_id} for peer_id in session.members], "pages": 1}

    def workspaces(self, filters=None, *, size: int = 50) -> FakePage:
        return FakePage([SimpleNamespace(id=self.workspace_id, metadata={"team": "growth"}, configuration=None)])

    def sessions(self, *, size: int = 50, include_scopes: bool = False) -> FakePage:
        self._ensure()
        return FakePage(list(self.by_id.values())[:size], total=len(self.by_id))

    def peers(self, filters=None, *, size: int = 50) -> FakePage:
        self._ensure()
        if filters:
            wanted = filters["id"]["in"]
            assert len(wanted) <= 100
            return FakePage([peer for peer in self.peer_list if peer.id in wanted])
        return FakePage(self.peer_list)

    def scopes(self, *, size: int = 50) -> FakePage:
        return FakePage([])

    def peer(self, peer_id: str, **kwargs) -> None:
        self._ensure()
        self.writes.append(("peer", peer_id, kwargs))

    def session(self, sid: str, **kwargs) -> FakeSession:
        self._ensure()
        self.writes.append(("session", sid, kwargs))
        return self.by_id.setdefault(sid, FakeSession(sid))

    def set_metadata(self, metadata) -> None:
        self.writes.append(("set_metadata", metadata))

    def set_configuration(self, configuration) -> None:
        self.writes.append(("set_configuration", configuration))


@pytest.fixture(autouse=True)
def _read_only_session():
    """Route replay's read-only ``Session(id, client)`` to the fake workspace."""
    with patch.object(replay_lib, "Session", side_effect=lambda sid, client: client.open_session(sid)):
        yield


def _writes(fake: FakeHoncho, kind: str) -> list[tuple]:
    return [w for w in fake.writes if w[0] == kind]


def _plans(*sessions: FakeSession) -> list[replay_lib.SessionPlan]:
    plans = [replay_lib.SessionPlan(s) for s in sessions]
    replay_lib.probe_sessions(plans)
    return plans


class TestSelectMostRecent:
    def test_ranks_by_newest_message_not_by_session_age(self):
        """The oldest session wins a slot when its latest message is the newest."""
        oldest_but_active = FakeSession("a", [_message(0, day=30)], created_at=T0 - timedelta(days=90))
        stale = FakeSession("b", [_message(0, day=1)], created_at=T0)
        recent = FakeSession("c", [_message(0, day=20)], created_at=T0 + timedelta(days=1))

        selected = replay_lib.select_most_recent(_plans(oldest_but_active, stale, recent), 2)

        # Kept in source order (oldest session first), so they replay in that order.
        assert [p.session.id for p in selected] == ["a", "c"]

    def test_session_without_messages_ranks_below_any_session_with_one(self):
        empty_but_new = FakeSession("empty", [], created_at=T0 + timedelta(days=365))
        old_with_message = FakeSession("old", [_message(0, day=-300)])

        selected = replay_lib.select_most_recent(_plans(empty_but_new, old_with_message), 1)

        assert [p.session.id for p in selected] == ["old"]


class TestReplaySession:
    def test_sends_every_message_in_order_with_its_original_timestamp(self):
        messages = [_message(i, "alice" if i % 2 == 0 else "bot") for i in range(250)]
        source = FakeHoncho("src", [FakeSession("s1", messages, members=["alice", "bot"], metadata={"topic": "x"})])
        dest = FakeHoncho("dst")

        result = replay_lib.Replayer(source, dest).replay_session(source.by_id["s1"])

        sent = dest.by_id["s1"]
        assert [c for c in sent.calls if c[0] == "add_messages"] == [
            ("add_messages", 100), ("add_messages", 100), ("add_messages", 50),
        ]
        assert [(m.peer_id, m.content, m.created_at, m.metadata) for m in sent.stored] == [
            (m.peer_id, m.content, m.created_at, m.metadata) for m in messages
        ]
        assert (result.messages_created, result.messages_skipped) == (250, 0)
        # The session is created with the source's settings and observer config
        # before any message is posted.
        kind, sid, kwargs = dest.writes[-1]
        assert (kind, sid, kwargs["metadata"]) == ("session", "s1", {"topic": "x"})
        assert kwargs["peers"] == [
            ("alice", SessionPeerConfig(observe_others=True)),
            ("bot", SessionPeerConfig(observe_others=False)),
        ]

    def test_departed_author_is_created_then_removed_again(self):
        """An author no longer in the source session must not stay a member."""
        messages = [_message(0), _message(1, "carol"), _message(2)]
        source = FakeHoncho("src", [FakeSession("s1", messages, members=["alice"])])
        dest = FakeHoncho("dst")

        replay_lib.Replayer(source, dest).replay_session(source.by_id["s1"])

        assert ("peer", "carol", {}) in dest.writes
        assert dest.by_id["s1"].calls[-1] == ("remove_peers", ["carol"])

    def test_departed_author_is_removed_when_a_later_batch_fails(self, monkeypatch):
        """A failed run must not leave a departed author as a member until a rerun."""
        monkeypatch.setattr(replay_lib, "MESSAGE_BATCH_SIZE", 1)
        messages = [_message(0, "carol"), _message(1), _message(2)]
        source = FakeHoncho("src", [FakeSession("s1", messages, members=["alice"])])
        dest = FakeHoncho("dst")
        dest_session = dest.by_id.setdefault("s1", FakeSession("s1"))
        send = dest_session.add_messages

        def fail_on_second_batch(batch) -> None:
            if dest_session.calls:
                raise RuntimeError("connection reset")
            send(batch)

        dest_session.add_messages = fail_on_second_batch

        with pytest.raises(RuntimeError, match="connection reset"):
            replay_lib.Replayer(source, dest).replay_session(source.by_id["s1"])

        assert dest_session.calls == [("add_messages", 1), ("remove_peers", ["carol"])]

    def test_rerun_skips_messages_already_in_the_destination(self):
        messages = [_message(i) for i in range(5)]
        source = FakeHoncho("src", [FakeSession("s1", messages)])
        dest = FakeHoncho("dst", [FakeSession("s1", messages[:3])])

        result = replay_lib.Replayer(source, dest).replay_session(source.by_id["s1"])

        assert (result.messages_created, result.messages_skipped) == (2, 3)
        assert [m.content for m in dest.by_id["s1"].stored] == [m.content for m in messages]

    def test_conflicting_destination_session_is_refused_before_any_write(self):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0), _message(1)])])
        unrelated = FakeSession("s1", [_message(7, "zed")])
        dest = FakeHoncho("dst", [unrelated])

        with pytest.raises(replay_lib.ReplayMismatch, match="first difference at message 1"):
            replay_lib.Replayer(source, dest).replay_session(source.by_id["s1"])

        assert dest.writes == []
        assert unrelated.calls == []


class TestCheckSessions:
    """The read-only look at the destination that a dry run reports."""

    def test_says_what_a_replay_would_do_with_each_session(self):
        messages = [_message(i) for i in range(4)]
        sessions = [FakeSession(sid, messages) for sid in ("fresh", "partial", "complete", "clash")]
        source = FakeHoncho("src", sessions)
        dest = FakeHoncho("dst", [
            FakeSession("partial", messages[:1]),
            FakeSession("complete", messages),
            FakeSession("clash", [messages[0], _message(9, "zed")]),
        ])
        plans = _plans(*sessions)

        replay_lib.Replayer(source, dest).check_sessions(plans)

        assert [(p.session.id, p.action, p.messages_in_destination, p.messages_to_send) for p in plans] == [
            ("fresh", "create", 0, 4),
            ("partial", "resume", 1, 3),
            ("complete", "up_to_date", 4, 0),
            ("clash", "conflict", 1, 0),
        ]
        assert plans[3].conflict_at == 2
        assert dest.writes == []

    def test_looking_at_a_missing_destination_does_not_create_it(self):
        s1 = FakeSession("s1", [_message(0)])
        source = FakeHoncho("src", [s1], [SimpleNamespace(id="alice", metadata={}, configuration=None)])
        dest = FakeHoncho("dst")
        replayer = replay_lib.Replayer(source, dest)
        plans = _plans(s1)

        replayer.check_sessions(plans)
        assert replayer.peers_in_destination(replayer.peers_for(plans, all_peers=True)) == set()

        assert plans[0].action == "create"
        assert dest.writes == []
        # The client still creates the workspace when a real replay follows.
        dest.peer("alice")
        assert _writes(dest, "workspace") == [("workspace", "dst")]


class TestReplay:
    def _run(self, source, dest, plans, *, all_peers):
        metadata, configuration = replay_lib.read_workspace(source)
        return replay_lib.replay(
            replay_lib.Replayer(source, dest),
            plans,
            workspace_metadata=metadata,
            workspace_configuration=configuration,
            all_peers=all_peers,
        )

    def test_partial_replay_creates_only_the_peers_its_sessions_use(self):
        peers = [SimpleNamespace(id=p, metadata={"p": p}, configuration=None) for p in ("alice", "bot", "dave")]
        s1 = FakeSession("s1", [_message(0)], members=["alice"])
        source = FakeHoncho("src", [s1], peers)

        partial = FakeHoncho("dst")
        self._run(source, partial, _plans(s1), all_peers=False)
        full = FakeHoncho("dst")
        self._run(source, full, _plans(s1), all_peers=True)

        def created(fake: FakeHoncho) -> list[str]:
            return [w[1] for w in fake.writes if w[0] == "peer"]

        assert created(partial) == ["alice"]
        assert created(full) == ["alice", "bot", "dave"]
        # Peers carry the source's metadata, set before their first message.
        assert ("peer", "alice", {"metadata": {"p": "alice"}}) in partial.writes

    def test_peers_the_destination_already_had_are_counted_apart_from_new_ones(self):
        peers = [SimpleNamespace(id=p, metadata={}, configuration=None) for p in ("alice", "bot")]
        s1 = FakeSession("s1", [_message(0), _message(1, "bot")], members=["alice", "bot"])
        source = FakeHoncho("src", [s1], peers)
        dest = FakeHoncho("dst", peers=[SimpleNamespace(id="bot", metadata={}, configuration=None)])

        result = self._run(source, dest, _plans(s1), all_peers=True)

        assert (result.peers_created, result.peers_existing) == (1, 1)

    def test_workspace_settings_are_copied_only_into_an_empty_destination(self):
        s1 = FakeSession("s1", [_message(0)])
        source = FakeHoncho("src", [s1])

        empty = FakeHoncho("dst")
        assert self._run(source, empty, _plans(s1), all_peers=True).workspace_copied is True
        assert ("set_metadata", {"team": "growth"}) in empty.writes

        in_use = FakeHoncho("dst", [FakeSession("theirs", [_message(0)])])
        assert self._run(source, in_use, _plans(s1), all_peers=True).workspace_copied is False
        assert not [w for w in in_use.writes if w[0].startswith("set_")]

    def test_one_failed_session_does_not_stop_the_others(self):
        good, clash = FakeSession("good", [_message(0)]), FakeSession("clash", [_message(0)])
        source = FakeHoncho("src", [clash, good])
        dest = FakeHoncho("dst", [FakeSession("clash", [_message(9, "zed")])])

        result = self._run(source, dest, _plans(clash, good), all_peers=True)

        assert [(f["session_id"], f["reason"]) for f in result.failed] == [("clash", "conflict")]
        assert [(s.session_id, s.action) for s in result.sessions] == [("good", "created")]

    def test_progress_says_what_happened_to_each_session(self):
        messages = [_message(i) for i in range(3)]
        sessions = [FakeSession(sid, messages) for sid in ("new", "half", "done", "clash")]
        source = FakeHoncho("src", sessions)
        dest = FakeHoncho("dst", [
            FakeSession("half", messages[:2]),
            FakeSession("done", messages),
            FakeSession("clash", [_message(9, "zed")]),
        ])
        lines: list[str] = []

        replay_lib.replay(
            replay_lib.Replayer(source, dest),
            _plans(*sessions),
            workspace_metadata={},
            workspace_configuration=None,
            all_peers=True,
            on_progress=lines.append,
        )

        assert lines == [
            "[1/4] new: 3 sent",
            "[2/4] half: 1 sent, 2 already present",
            "[3/4] done: up to date, 3 already present",
            "[4/4] clash: not replayed, the destination session already has different messages "
            "(first difference at message 1)",
        ]


class TestEndpointCredentials:
    """A key for one instance must never be sent to another."""

    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        f = tmp_path / "config.json"
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
        for k in [k for k in os.environ if k.startswith("HONCHO_")]:
            monkeypatch.delenv(k)
        f.write_text(json.dumps({"apiKey": "file-key", "environmentUrl": "https://prod.test"}))
        return f

    def test_other_instance_gets_neither_the_saved_key_nor_the_env_key(self, cfg, monkeypatch):
        monkeypatch.setenv("HONCHO_API_KEY", "env-key")

        client = get_endpoint_client(workspace_id="w", base_url="http://localhost:8000")

        assert client._http.api_key is None
        assert os.environ["HONCHO_API_KEY"] == "env-key"  # restored for the other side

    def test_explicit_key_is_used_for_its_own_side_only(self, cfg):
        other = get_endpoint_client(workspace_id="w", base_url="https://staging.test", api_key="staging-key")
        configured = get_endpoint_client(workspace_id="w")

        assert (other.base_url, other._http.api_key) == ("https://staging.test", "staging-key")
        assert (configured.base_url, configured._http.api_key) == ("https://prod.test", "file-key")

    def test_configured_instance_named_by_url_keeps_its_credentials(self, cfg):
        client = get_endpoint_client(workspace_id="w", base_url="https://prod.test/")

        assert client._http.api_key == "file-key"


class TestReplayCommand:
    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        f = tmp_path / "config.json"
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
        for k in [k for k in os.environ if k.startswith("HONCHO_")]:
            monkeypatch.delenv(k)
        f.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        return f

    def _invoke(self, args, source, dest):
        clients = iter([source, dest])
        with patch("honcho_cli.commands.workspace.get_endpoint_client", side_effect=lambda **_: next(clients)):
            return CliRunner().invoke(app, ["workspace", "replay", *args])

    def test_same_workspace_on_same_instance_is_refused(self, cfg):
        result = CliRunner().invoke(app, ["workspace", "replay", "--from", "acme"])

        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "SAME_WORKSPACE"

    def test_key_for_another_instance_comes_from_the_environment_only(self, cfg, monkeypatch):
        monkeypatch.setenv("HONCHO_TO_API_KEY", "dest-key")
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])])
        asked: list[dict] = []
        clients = iter([source, FakeHoncho("dst")])

        def endpoint(**kwargs):
            asked.append(kwargs)
            return next(clients)

        args = ["workspace", "replay", "--from", "src", "--to-url", "http://dst.test", "--dry-run"]
        with patch("honcho_cli.commands.workspace.get_endpoint_client", side_effect=endpoint):
            result = CliRunner().invoke(app, args)

        assert result.exit_code == 0, result.stderr
        assert [(k["base_url"], k["api_key"]) for k in asked] == [(None, None), ("http://dst.test", "dest-key")]
        # No flag takes a key: an argument shows up in shell history and process listings.
        for flag in ("--to-api-key", "--from-api-key"):
            assert CliRunner().invoke(app, [*args, flag, "dest-key"]).exit_code == 2

    def test_dry_run_reports_the_selection_and_writes_nothing(self, cfg):
        source = FakeHoncho("src", [
            FakeSession("old", [_message(0, day=1)]),
            FakeSession("new", [_message(0, day=9), _message(1, day=9)]),
        ])
        dest = FakeHoncho("dst")

        result = self._invoke(["--from", "src", "--to", "dst", "--num-sessions", "1", "--dry-run"], source, dest)

        assert result.exit_code == 0, result.stderr
        out = json.loads(result.stdout)
        assert (out["sessions_in_source"], out["sessions_to_replay"], out["messages_to_send"]) == (2, 1, 2)
        assert [(s["id"], s["action"]) for s in out["sessions"]] == [("new", "create")]
        assert out["destination"]["sessions"] == 0
        assert dest.writes == []

    def test_dry_run_against_a_destination_in_use_predicts_each_session(self, cfg):
        messages = [_message(i) for i in range(3)]
        source = FakeHoncho(
            "src",
            [FakeSession(sid, messages) for sid in ("a", "b", "c")],
            [SimpleNamespace(id="alice", metadata={}, configuration=None)],
        )
        dest = FakeHoncho(
            "dst",
            [FakeSession("b", messages[:2]), FakeSession("c", [_message(8, "zed")])],
            [SimpleNamespace(id="alice", metadata={}, configuration=None)],
        )

        result = self._invoke(["--from", "src", "--to", "dst", "--dry-run"], source, dest)

        assert result.exit_code == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["actions"] == {"create": 1, "resume": 1, "up_to_date": 0, "conflict": 1}
        assert (out["messages_in_sessions"], out["messages_to_send"]) == (9, 4)
        assert (out["peers_to_copy"], out["peers_in_destination"]) == (1, 1)
        assert (out["destination"]["sessions"], out["workspace_settings_copied"]) == (2, False)
        assert dest.writes == []

    def test_failed_session_is_reported_and_exits_nonzero(self, cfg):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)]), FakeSession("s2", [_message(0)])])
        dest = FakeHoncho("dst", [FakeSession("s1", [_message(5, "zed")])])

        result = self._invoke(["--from", "src", "--to", "dst"], source, dest)

        assert result.exit_code == 1
        out = json.loads(result.stdout)
        assert (out["sessions_replayed"], out["sessions_failed"], out["messages_created"]) == (1, 1, 1)
        assert [(f["session_id"], f["reason"]) for f in out["failed"]] == [("s1", "conflict")]
        assert out["sessions"] == [
            {"id": "s2", "action": "created", "messages_created": 1, "messages_skipped": 0}
        ]
        assert (out["peers_created"], out["peers_existing"]) == (1, 0)

    def test_both_reports_count_the_conclusions_a_replay_leaves_behind(self, cfg):
        """Conclusions written through the API cannot be derived again, so say how many exist."""
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])], conclusions=12)
        args = ["--from", "src", "--to", "dst"]

        for extra in (["--dry-run"], []):
            out = json.loads(self._invoke([*args, *extra], source, FakeHoncho("dst")).stdout)
            assert out["not_copied"]["conclusions_in_source"] == 12
            assert "written through the API" in out["not_copied"]["note"]

        # A source that cannot count them (an older server) still replays.
        source.conclusions.list = lambda size=50: (_ for _ in ()).throw(RuntimeError("no such route"))
        result = self._invoke(args, source, FakeHoncho("dst"))
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["not_copied"]["conclusions_in_source"] is None

    def test_result_names_the_command_that_shows_the_destination_queue(self, cfg):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])])

        out = json.loads(self._invoke(["--from", "src", "--to", "dst"], source, FakeHoncho("dst")).stdout)
        # Another instance: the command must carry that instance's URL, and
        # name the key variable instead of printing a key.
        assert out["check_derivation"] == (
            'HONCHO_BASE_URL=http://dst.test HONCHO_API_KEY="$HONCHO_TO_API_KEY" '
            "honcho workspace queue-status -w dst"
        )

        local = FakeHoncho("dst")
        local.base_url = "http://localhost:8000/"
        out = json.loads(self._invoke(["--from", "src", "--to", "dst"], source, local).stdout)
        assert out["check_derivation"] == "honcho workspace queue-status -w dst"

        # Nothing left to derive on a rerun, so there is nothing to check.
        out = json.loads(self._invoke(["--from", "src", "--to", "dst"], source, local).stdout)
        assert (out["messages_created"], out["check_derivation"]) == (0, None)


class ReadOnlyServer:
    """A Honcho API in memory, as an httpx transport. It serves reads and refuses the rest.

    ``sessions`` maps a session ID to its messages, each ``(peer_id, content)``.
    Every request that is not a read lands in ``writes`` and gets a 405.
    """

    def __init__(
        self, workspace_id: str, sessions=None, *, conclusions: int = 0, get_or_create: bool = False, scoped_key: bool = False
    ) -> None:
        self.scoped_key = scoped_key  # the caller's key cannot list workspaces
        self.workspace_id = workspace_id
        self.sessions = dict(sessions or {})
        self.conclusions = conclusions
        self.get_or_create = get_or_create  # a source may run the SDK's workspace get-or-create
        self.reads: list[str] = []
        self.writes: list[str] = []

    def client(self) -> honcho.Honcho:
        """The real SDK client, talking to this server."""
        url = f"http://{self.workspace_id}.test"
        http_client = httpx.Client(transport=httpx.MockTransport(self), base_url=url)
        return honcho.Honcho(base_url=url, api_key="k", workspace_id=self.workspace_id, http_client=http_client)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if self.scoped_key and path == "/v3/workspaces/list":
            return httpx.Response(401, json={"message": "admin key required"})
        if request.method == "GET" or path.endswith("/list"):
            self.reads.append(f"{request.method} {path}")
            return httpx.Response(200, json=self._read(request))
        if self.get_or_create and (request.method, path) == ("POST", "/v3/workspaces"):
            return httpx.Response(200, json=self._workspace())
        self.writes.append(f"{request.method} {path}")
        return httpx.Response(405, json={"message": "this server is read-only"})

    def _workspace(self) -> dict:
        return {"id": self.workspace_id, "metadata": {}, "configuration": {}, "created_at": T0.isoformat()}

    def _messages(self, session_id: str) -> list[dict]:
        return [
            {
                "id": f"{session_id}-{i}", "content": content, "peer_id": peer_id, "session_id": session_id,
                "workspace_id": self.workspace_id, "metadata": {}, "token_count": 1,
                "created_at": (T0 + timedelta(minutes=i)).isoformat(),
            }
            for i, (peer_id, content) in enumerate(self.sessions[session_id])
        ]

    def _peers(self, session_ids) -> list[dict]:
        names = dict.fromkeys(peer_id for sid in session_ids for peer_id, _ in self.sessions[sid])
        return [
            {"id": name, "workspace_id": self.workspace_id, "metadata": {}, "configuration": {}, "created_at": T0.isoformat()}
            for name in names
        ]

    def _read(self, request: httpx.Request) -> dict:
        parts = request.url.path.removeprefix("/v3/workspaces/").split("/")
        body = json.loads(request.content or b"{}") or {}
        if parts == ["list"]:
            items = [self._workspace()]
        elif parts[1:] == ["sessions", "list"]:
            items = [
                {"id": sid, "is_active": True, "workspace_id": self.workspace_id, "metadata": {}, "configuration": {}, "created_at": T0.isoformat()}
                for sid in self.sessions
            ]
        elif parts[1:] == ["peers", "list"]:
            wanted = ((body.get("filters") or {}).get("id") or {}).get("in")
            items = [peer for peer in self._peers(self.sessions) if wanted is None or peer["id"] in wanted]
        elif parts[1:] == ["conclusions", "list"]:
            return {"items": [], "total": self.conclusions, "page": 1, "size": 1, "pages": self.conclusions}
        elif parts[1] == "sessions" and parts[3:] == ["messages", "list"]:
            items = self._messages(parts[2]) if parts[2] in self.sessions else []
            if request.url.params.get("reverse") == "true":
                items = items[::-1]
        elif parts[1] == "sessions" and parts[3:] == ["peers"]:
            items = self._peers([parts[2]])
        else:
            raise AssertionError(f"unexpected read: {request.method} {request.url.path}")
        page, size = int(request.url.params.get("page", 1)), int(request.url.params.get("size", 50))
        return {
            "items": items[(page - 1) * size : page * size],
            "total": len(items), "page": page, "size": size, "pages": max(-(-len(items) // size), 1),
        }


class TestRealSDKReadOnly:
    """A dry run goes through the real SDK client and sends the destination no write.

    ``read_only`` depends on a private SDK attribute. These tests use the SDK
    itself, not ``FakeHoncho``, so a change to that attribute fails here.
    """

    CONVERSATION = [("alice", "hi"), ("bob", "hello"), ("alice", "how are you")]

    @pytest.fixture(autouse=True)
    def real_session(self):
        with patch.object(replay_lib, "Session", honcho.Session):
            yield

    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        f = tmp_path / "config.json"
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
        for k in [k for k in os.environ if k.startswith("HONCHO_")]:
            monkeypatch.delenv(k)
        f.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        return f

    def _dry_run(self, source: ReadOnlyServer, dest: ReadOnlyServer, *extra: str):
        clients = iter([source.client(), dest.client()])
        args = ["workspace", "replay", "--from", "src", "--to", "dst", "--dry-run", *extra]
        with patch("honcho_cli.commands.workspace.get_endpoint_client", side_effect=lambda **_: next(clients)):
            return CliRunner().invoke(app, args)

    def _source(self) -> ReadOnlyServer:
        sessions = {sid: self.CONVERSATION for sid in ("a", "b", "c")}
        return ReadOnlyServer("src", sessions, conclusions=7, get_or_create=True)

    def test_an_sdk_read_outside_read_only_creates_the_workspace(self):
        """The hazard itself: without the guard, looking at a workspace creates it."""
        dest = ReadOnlyServer("dst")
        client = dest.client()

        with pytest.raises(honcho.APIError):
            client.sessions(size=1)
        assert dest.writes == ["POST /v3/workspaces"]

        with replay_lib.read_only(client):
            assert client.sessions(size=1).total == 0
        assert dest.writes == ["POST /v3/workspaces"]  # nothing new

    def test_dry_run_sends_no_write_to_a_destination_that_does_not_exist(self, cfg):
        dest = ReadOnlyServer("dst")

        result = self._dry_run(self._source(), dest)

        assert result.exit_code == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["actions"] == {"create": 3, "resume": 0, "up_to_date": 0, "conflict": 0}
        assert (out["peers_to_copy"], out["peers_in_destination"]) == (2, 0)
        # Read through the real SDK: a missing `conclusions.list` would give None.
        assert out["not_copied"]["conclusions_in_source"] == 7
        assert dest.reads and dest.writes == []

    @pytest.mark.parametrize("extra", [(), ("--num-sessions", "3")], ids=["full", "partial"])
    def test_dry_run_sends_no_write_to_a_destination_in_use(self, cfg, extra):
        dest = ReadOnlyServer("dst", {"b": self.CONVERSATION[:2], "c": [("zed", "something else")]})

        result = self._dry_run(self._source(), dest, *extra)

        assert result.exit_code == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["actions"] == {"create": 1, "resume": 1, "up_to_date": 0, "conflict": 1}
        assert (out["messages_in_sessions"], out["messages_to_send"]) == (9, 4)
        assert (out["peers_to_copy"], out["peers_in_destination"]) == (2, 2)
        # Every kind of destination read ran, and none of them was a write.
        assert {read.rsplit("/", 2)[-2] for read in dest.reads} == {"sessions", "messages", "peers"}
        assert dest.writes == []

    def test_scoped_key_does_not_create_a_missing_source(self, cfg):
        """A key that cannot list workspaces must not find out by get-or-create."""
        source = ReadOnlyServer("src", scoped_key=True)

        result = self._dry_run(source, ReadOnlyServer("dst"))

        assert result.exit_code == 1
        assert json.loads(result.stderr.splitlines()[-1])["error"]["code"] == "NOTHING_TO_REPLAY"
        assert source.reads and source.writes == []

    def test_scoped_key_reads_a_source_that_has_content(self, cfg):
        source = self._source()
        source.scoped_key = True

        result = self._dry_run(source, ReadOnlyServer("dst"))

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["sessions_to_replay"] == 3

    def test_dry_run_refuses_an_sdk_it_cannot_keep_read_only(self, cfg):
        """If the SDK drops the attribute, the dry run stops instead of writing."""
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])])
        dest = FakeHoncho("dst")
        del dest._workspace_ensured
        clients = iter([source, dest])

        with patch("honcho_cli.commands.workspace.get_endpoint_client", side_effect=lambda **_: next(clients)):
            result = CliRunner().invoke(app, ["workspace", "replay", "--from", "src", "--to", "dst", "--dry-run"])

        assert result.exit_code == 1
        assert json.loads(result.stderr.splitlines()[-1])["error"]["code"] == "SDK_UNSUPPORTED"
        assert dest.writes == []


class TestHumanOutput:
    """In a terminal the same reports are rendered for reading, not as JSON."""

    @pytest.fixture(autouse=True)
    def terminal(self, monkeypatch):
        from honcho_cli import output

        monkeypatch.setattr(output, "is_tty", lambda: True)

    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        f = tmp_path / "config.json"
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
        for k in [k for k in os.environ if k.startswith("HONCHO_")]:
            monkeypatch.delenv(k)
        monkeypatch.setenv("COLUMNS", "200")
        f.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        return f

    def _invoke(self, args, source, dest):
        clients = iter([source, dest])
        with patch("honcho_cli.commands.workspace.get_endpoint_client", side_effect=lambda **_: next(clients)):
            return CliRunner().invoke(app, ["workspace", "replay", *args])

    def test_dry_run_shows_one_row_per_session_and_explains_a_conflict(self, cfg):
        messages = [_message(i) for i in range(3)]
        source = FakeHoncho("src", [FakeSession(sid, messages) for sid in ("fresh", "partial", "clash")])
        dest = FakeHoncho("dst", [FakeSession("partial", messages[:1]), FakeSession("clash", [_message(8, "zed")])])

        result = self._invoke(["--from", "src", "--to", "dst", "--dry-run"], source, dest)

        assert result.exit_code == 0, result.stderr
        out = result.stdout
        assert "Dry run. Nothing was written." in out
        assert "It has 2 sessions." in out
        assert "1 to create · 1 to resume · 1 in conflict" in out
        assert "resume, send 2" in out
        assert "differs at message 1" in out
        assert "1 session in conflict." in out
        assert "Not copied  Peer cards. The source has no conclusions." in out
        assert "{" not in out  # no JSON inside the report

    def test_dry_run_in_a_narrow_terminal_keeps_each_session_readable(self, cfg, monkeypatch):
        monkeypatch.setenv("COLUMNS", "48")
        messages = [_message(i) for i in range(3)]
        source = FakeHoncho("src", [FakeSession("a-session-with-a-long-name", messages), FakeSession("clash", messages)])
        dest = FakeHoncho("dst", [FakeSession("clash", [_message(8, "zed")])])

        out = self._invoke(["--from", "src", "--to", "dst", "--dry-run"], source, dest).stdout

        # No table: at this width every cell would wrap to a few characters.
        assert "┃" not in out
        assert "  a-session-with-a-long-name  create" in out
        assert "  clash  conflict, skipped" in out
        assert "      destination differs at message 1" in out
        assert "      3 messages, newest 2025-03-01 09:02 UTC" in out

    def test_dry_run_of_a_large_workspace_lists_the_sessions_that_need_attention(self, cfg):
        sessions = [FakeSession(f"s{i:03d}", [_message(0)]) for i in range(60)]
        source = FakeHoncho("src", sessions)
        dest = FakeHoncho("dst", [FakeSession("s059", [_message(4, "zed")])])

        out = self._invoke(["--from", "src", "--to", "dst", "--dry-run"], source, dest).stdout

        assert "Showing 40 of 60 sessions." in out
        assert "s059" in out  # the conflict is shown although it is last in the source
        assert "s058" not in out

    def test_summary_lists_failures_and_the_next_command(self, cfg):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)]), FakeSession("s2", [_message(0)])])
        dest = FakeHoncho("dst", [FakeSession("s1", [_message(5, "zed")])])

        result = self._invoke(["--from", "src", "--to", "dst"], source, dest)

        assert result.exit_code == 1
        out = result.stdout
        assert "Replayed 1 of 2 sessions" in out
        assert "1 failed." in out
        assert "first difference at message 1" in out
        assert "A conflict does not clear on a rerun." in out
        assert "kept its own" in out  # the destination's workspace settings
        assert 'HONCHO_API_KEY="$HONCHO_TO_API_KEY" honcho workspace queue-status -w dst' in out
        assert "{" not in out

    def test_summary_of_a_clean_run_has_no_empty_rows(self, cfg):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])], conclusions=412)

        out = self._invoke(["--from", "src", "--to", "dst"], source, FakeHoncho("dst")).stdout

        assert "Replayed 1 session in" in out
        assert "1 sent" in out
        assert "Not copied  412 conclusions and any peer cards." in out
        assert "written through the API are not recreated." in out
        for absent in ("already present", "failed", "Failed", "Settings", "already existed"):
            assert absent not in out

    def test_rerun_with_nothing_left_to_send_says_so(self, cfg):
        source = FakeHoncho("src", [FakeSession("s1", [_message(0)])])
        dest = FakeHoncho("dst", [FakeSession("s1", [_message(0)])])

        result = self._invoke(["--from", "src", "--to", "dst"], source, dest)

        assert result.exit_code == 0
        assert "Already up to date. Nothing was sent" in result.stdout
        assert "queue-status" not in result.stdout  # nothing new to derive
        assert "[1/1] s1: up to date, 1 already present" in result.stderr
