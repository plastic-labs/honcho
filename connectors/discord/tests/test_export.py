"""Walking a server through a fake Discord API, into JSONL and the ingest API."""

import json
import os
from pathlib import Path
from typing import Any, final

import httpx
import pytest
from honcho_discord.__main__ import main
from honcho_discord.client import DiscordClient, DiscordError
from honcho_discord.export import (
    IngestError,
    IngestSink,
    JsonlSink,
    SelectionError,
    export,
)

GUILD = "77"


def _user(uid: str, name: str, **extra: Any) -> dict[str, Any]:
    return {"id": uid, "username": name, "global_name": None, **extra}


def _msg(
    mid: int, channel: str, author: dict[str, Any], content: str, **extra: Any
) -> dict[str, Any]:
    return {
        "id": str(mid),
        "type": 0,
        "channel_id": channel,
        "author": author,
        "content": content,
        "timestamp": f"2026-11-04T10:{mid % 60:02d}:00+00:00",
        "edited_timestamp": None,
        "mentions": [],
        "attachments": [],
        "embeds": [],
        **extra,
    }


JESS, MAYA, BOT = _user("1", "jess"), _user("3", "maya"), _user("9", "ci", bot=True)


@final
class FakeDiscord:
    """Channels, threads, members and messages, served the way Discord pages them."""

    def __init__(self, page: int = 100) -> None:
        self.page = page
        self.requests: list[str] = []
        self.rate_limited_once = False
        self.channels: list[dict[str, Any]] = [
            {"id": "100", "type": 4, "name": "Engineering", "position": 0},
            {"id": "200", "type": 0, "name": "dev", "parent_id": "100", "position": 1},
            {
                "id": "210",
                "type": 0,
                "name": "random",
                "parent_id": None,
                "position": 2,
            },
            {
                "id": "220",
                "type": 15,
                "name": "ideas",
                "parent_id": "100",
                "position": 3,
            },
            {"id": "230", "type": 2, "name": "voice", "position": 4},
        ]
        self.active: list[dict[str, Any]] = [
            {"id": "300", "type": 11, "name": "sso", "parent_id": "200"}
        ]
        self.archived: dict[str, list[dict[str, Any]]] = {
            "220": [
                {
                    "id": "310",
                    "type": 11,
                    "name": "post-a",
                    "parent_id": "220",
                    "thread_metadata": {
                        "archive_timestamp": "2026-11-01T00:00:00+00:00"
                    },
                }
            ]
        }
        self.messages: dict[str, list[dict[str, Any]]] = {
            "200": [
                _msg(i, "200", JESS if i % 2 else MAYA, f"dev {i}") for i in range(1, 8)
            ]
            + [_msg(8, "200", JESS, "", type=7)],
            "300": [_msg(20, "300", MAYA, "thread reply", mentions=[JESS])],
            "210": [_msg(30, "210", BOT, "build green")],
            "310": [_msg(40, "310", JESS, "forum post")],
            "230": [_msg(50, "230", MAYA, "voice chat")],
        }
        self.members: list[dict[str, Any]] = [
            {"user": JESS, "nick": "JessD"},
            {"user": MAYA, "nick": None},
        ]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v10")
        self.requests.append(path)
        if path.endswith("/messages") and not self.rate_limited_once:
            self.rate_limited_once = True
            return httpx.Response(429, json={"retry_after": 0.01})
        if path == f"/guilds/{GUILD}/channels":
            return httpx.Response(200, json=self.channels)
        if path == f"/guilds/{GUILD}/threads/active":
            return httpx.Response(200, json={"threads": self.active})
        if path.endswith("/threads/archived/public"):
            channel = path.split("/")[2]
            return httpx.Response(
                200, json={"threads": self.archived.get(channel, []), "has_more": False}
            )
        if path == f"/guilds/{GUILD}/members":
            return httpx.Response(200, json=self.members)
        if path.endswith("/messages"):
            channel = path.split("/")[2]
            after = int(request.url.params.get("after", "0"))
            newer = [m for m in self.messages.get(channel, []) if int(m["id"]) > after]
            # Discord answers with the page closest to `after`, newest first.
            page = sorted(newer, key=lambda m: int(m["id"]))[: self.page]
            return httpx.Response(200, json=list(reversed(page)))
        return httpx.Response(404, json={"message": "Unknown"})


def _client(fake: FakeDiscord) -> DiscordClient:
    return DiscordClient(
        "test-token",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(fake.handler),
        sleep=lambda _s: None,
    )


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_walks_channels_then_threads_oldest_first(tmp_path: Path) -> None:
    fake = FakeDiscord(page=3)
    with _client(fake) as client:
        stats, _ = export(client, GUILD, lambda: JsonlSink(tmp_path))

    records = _read(tmp_path / "records.jsonl")
    order = [(r["container"]["id"], r["source_id"]) for r in records]
    # dev, then its thread, then random; the forum has no messages of its own,
    # only its post's; the voice channel's text chat comes last, by position.
    assert order == (
        [("200", str(i)) for i in range(1, 8)]
        + [("300", "20"), ("210", "30"), ("310", "40"), ("230", "50")]
    )
    assert stats.records == 11
    assert stats.complete
    assert stats.skipped == {"system_or_empty": 1}
    # Three pages of three for dev's eight messages, so paging resumes in order.
    assert fake.requests.count("/channels/200/messages") >= 3

    containers = _read(tmp_path / "containers.jsonl")
    assert [(c["id"], c["parent_id"]) for c in containers] == [
        ("200", "100"),
        ("300", "200"),
        ("210", None),
        ("310", "220"),
        ("230", None),
    ]
    accounts = {a["id"]: a for a in _read(tmp_path / "accounts.jsonl")}
    assert accounts["1"]["display_name"] == "JessD"
    assert accounts["9"]["bot"] is True
    assert records[-4]["participants"] == [
        {"id": "1", "name": "JessD", "handle": "jess"}
    ]
    summary = json.loads((tmp_path / "export.json").read_text())
    assert summary["categories"] == {"100": "Engineering"}
    assert "test-token" not in (tmp_path / "records.jsonl").read_text()


def test_export_can_take_some_channels_only(tmp_path: Path) -> None:
    with _client(FakeDiscord()) as client:
        export(client, GUILD, lambda: JsonlSink(tmp_path), channel_ids={"210"})
    assert [r["source_id"] for r in _read(tmp_path / "records.jsonl")] == ["30"]


def test_a_member_list_it_cant_read_falls_back_to_global_names(tmp_path: Path) -> None:
    fake = FakeDiscord()
    original = fake.handler

    def no_members(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/members"):
            return httpx.Response(403, json={"message": "Missing Access"})
        return original(request)

    client = DiscordClient(
        "t",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(no_members),
        sleep=lambda _s: None,
    )
    with client:
        export(client, GUILD, lambda: JsonlSink(tmp_path), channel_ids={"200"})
    authors = {r["author"]["name"] for r in _read(tmp_path / "records.jsonl")}
    assert authors == {"jess", "maya"}


@final
class FakeIngest:
    def __init__(self, workspace_status: int = 201) -> None:
        self.batches: list[dict[str, Any]] = []
        self.workspace_bodies: list[dict[str, Any]] = []
        self.workspace_status = workspace_status
        self.busy_once = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/v3/workspaces":
            self.workspace_bodies.append(body)
            return httpx.Response(self.workspace_status, json={"detail": "nope"})
        if self.busy_once:
            self.busy_once = False
            return httpx.Response(503, headers={"Retry-After": "0"})
        self.batches.append(body)
        return httpx.Response(
            200,
            json={
                "records": [
                    {"index": i, "outcome": "created", "reason": None}
                    for i in range(len(body["records"]))
                ]
            },
        )


def test_the_ingest_sink_sends_each_container_in_order_and_defers() -> None:
    ingest = FakeIngest()
    sink = IngestSink(
        "http://honcho.test",
        "discord-replay",
        transport=httpx.MockTransport(ingest.handler),
        sleep=lambda _s: None,
    )
    with _client(FakeDiscord()) as client:
        export(client, GUILD, lambda: sink)

    assert ingest.workspace_bodies == [{"id": "discord-replay", "auto": True}]
    assert all(b["derive"] == "later" for b in ingest.batches)
    # One batch per container: a container's history is finished before the next.
    per_batch = [{r["container"]["id"] for r in b["records"]} for b in ingest.batches]
    assert per_batch == [{"200"}, {"300"}, {"210"}, {"310"}, {"230"}]
    assert sink.outcomes == {"created": 11}


def test_the_ingest_sink_refuses_a_workspace_it_cant_use() -> None:
    ingest = FakeIngest(workspace_status=409)
    with pytest.raises(IngestError, match="409"):
        IngestSink(
            "http://honcho.test", "plain", transport=httpx.MockTransport(ingest.handler)
        )


def test_the_cli_wont_write_messages_into_a_git_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    token = tmp_path / "token"
    token.write_text("t")
    with pytest.raises(SystemExit, match="git checkout"):
        main(
            [
                "export",
                "--token-file",
                str(token),
                "--guild",
                GUILD,
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_an_empty_or_unknown_selection_is_refused_before_any_output(
    tmp_path: Path,
) -> None:
    made: list[str] = []

    def sink() -> JsonlSink:
        made.append("sink")
        return JsonlSink(tmp_path)

    with _client(FakeDiscord()) as client:
        with pytest.raises(SelectionError, match="No channels"):
            export(client, GUILD, sink, channel_ids=set())
        with pytest.raises(SelectionError, match="100, 999"):
            # A category and a channel the guild doesn't have.
            export(client, GUILD, sink, channel_ids={"200", "100", "999"})
    assert made == []


def test_a_channels_file_listing_nothing_is_an_error(tmp_path: Path) -> None:
    token, empty = tmp_path / "token", tmp_path / "channels.txt"
    token.write_text("t")
    empty.write_text("# nothing here\n\n")
    with pytest.raises(SystemExit, match="lists no channel IDs"):
        main(
            [
                "export",
                "--token-file",
                str(token),
                "--guild",
                GUILD,
                "--channels-file",
                str(empty),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_a_token_file_must_hold_one_line(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("abc\nAuthorization: x")
    with pytest.raises(SystemExit, match="one line"):
        main(["channels", "--token-file", str(token), "--guild", GUILD])


def test_threads_it_couldnt_list_make_the_export_incomplete(tmp_path: Path) -> None:
    fake = FakeDiscord()
    original = fake.handler

    def archive_fails(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/220/threads/archived/public"):
            return httpx.Response(503)
        return original(request)

    client = DiscordClient(
        "t",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(archive_fails),
        sleep=lambda _s: None,
    )
    with client:
        stats, _ = export(client, GUILD, lambda: JsonlSink(tmp_path))
    assert stats.incomplete == ["220"]
    assert not stats.complete
    summary = json.loads((tmp_path / "export.json").read_text())
    assert summary["complete"] is False
    assert "220" in summary["incomplete_thread_lists"]


def test_a_thread_unarchived_during_discovery_is_still_found(tmp_path: Path) -> None:
    fake = FakeDiscord()
    fake.archived = {}
    late = {"id": "320", "type": 11, "name": "late", "parent_id": "210"}
    fake.messages["320"] = [_msg(60, "320", JESS, "back from the archive")]
    reads: list[int] = []
    original = fake.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/threads/active"):
            reads.append(1)
            if len(reads) == 2:
                fake.active = [*fake.active, late]
        return original(request)

    client = DiscordClient(
        "t",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with client:
        export(client, GUILD, lambda: JsonlSink(tmp_path))
    ids = {r["container"]["id"] for r in _read(tmp_path / "records.jsonl")}
    assert "320" in ids


def test_output_files_are_never_written_through_a_symlink(tmp_path: Path) -> None:
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere.jsonl"
    out.mkdir()
    (out / "records.jsonl").symlink_to(elsewhere)
    sink = JsonlSink(out)
    sink.close([], {})
    assert not elsewhere.exists()
    assert not (out / "records.jsonl").is_symlink()


def test_a_hard_linked_output_file_is_replaced_not_overwritten(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("keep me")
    outside.chmod(0o644)
    out = tmp_path / "export"
    out.mkdir()
    os.link(outside, out / "records.jsonl")

    sink = JsonlSink(out)
    sink.record({"source_id": "a"})
    sink.close([], {})

    assert outside.read_text() == "keep me"
    assert outside.stat().st_mode & 0o777 == 0o644
    assert _read(out / "records.jsonl") == [{"source_id": "a"}]
    assert (out / "records.jsonl").stat().st_mode & 0o777 == 0o600


def test_rate_limits_dont_use_up_the_failure_retries() -> None:
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 10:
            return httpx.Response(429, json={"retry_after": 0.0})
        return httpx.Response(200, json=[])

    client = DiscordClient(
        "t",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with client:
        assert client.get("/guilds/77/channels") == []


def test_network_failures_are_retried_and_never_echo_the_token() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError(f"boom {request.headers['Authorization']}")

    client = DiscordClient(
        "secret-token",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with client, pytest.raises(DiscordError) as raised:
        client.get("/guilds/77/channels")
    assert len(calls) == 6
    assert raised.value.retryable
    assert "secret-token" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_an_ingest_refusal_doesnt_echo_the_records() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/workspaces":
            return httpx.Response(201, json={})
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "records", 0, "content"],
                        "type": "string_too_long",
                        "input": "private message text",
                    }
                ]
            },
        )

    sink = IngestSink("http://honcho.test", "w", transport=httpx.MockTransport(handler))
    sink.record({"content": "private message text"})
    with pytest.raises(IngestError) as raised:
        sink.close([], {})
    assert "body.records.0.content: string_too_long" in str(raised.value)
    assert "private message text" not in str(raised.value)


def test_a_symlinked_output_directory_is_refused(tmp_path: Path) -> None:
    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        JsonlSink(link)


def test_an_earlier_exports_files_become_owner_only(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    old = out / "records.jsonl"
    old.write_text("old")
    old.chmod(0o644)
    sink = JsonlSink(out)
    sink.close([], {})
    assert (old.stat().st_mode & 0o777) == 0o600


def test_rate_limit_count_resets_between_answers() -> None:
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        n = len(calls)
        if n <= 150 or 151 < n <= 301:
            return httpx.Response(429, json={"retry_after": 0.0})
        if n == 151:
            return httpx.Response(500)
        return httpx.Response(200, json=[])

    client = DiscordClient(
        "t",
        base_url="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    with client:
        assert client.get("/guilds/77/channels") == []


def test_an_ingest_transport_failure_doesnt_echo_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"boom {request.headers.get('Authorization')}")

    with pytest.raises(IngestError) as raised:
        IngestSink(
            "http://honcho.test",
            "w",
            api_key="secret-key",
            transport=httpx.MockTransport(handler),
        )
    assert "secret-key" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_forwarded_mentions_reach_the_account_list(tmp_path: Path) -> None:
    fake = FakeDiscord()
    fake.messages = {
        "210": [
            _msg(
                70,
                "210",
                JESS,
                "",
                message_snapshots=[
                    {
                        "message": {
                            "content": "hi <@4>",
                            "mentions": [_user("4", "omar")],
                        }
                    }
                ],
            )
        ]
    }
    with _client(fake) as client:
        export(client, GUILD, lambda: JsonlSink(tmp_path), channel_ids={"210"})
    ids = {a["id"] for a in _read(tmp_path / "accounts.jsonl")}
    assert ids == {"1", "4"}


def test_a_replied_to_webhook_is_a_bot_in_the_account_list(tmp_path: Path) -> None:
    hook = _msg(71, "210", _user("8", "deploys"), "deployed", webhook_id="8")
    fake = FakeDiscord()
    fake.messages = {
        "210": [
            _msg(
                72,
                "210",
                JESS,
                "thanks",
                type=19,
                message_reference={"message_id": "71"},
                referenced_message=hook,
            )
        ]
    }
    with _client(fake) as client:
        export(client, GUILD, lambda: JsonlSink(tmp_path), channel_ids={"210"})
    accounts = {a["id"]: a for a in _read(tmp_path / "accounts.jsonl")}
    assert accounts["8"]["bot"] is True
    assert accounts["1"]["bot"] is False


def test_rejected_records_make_the_cli_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDiscord()

    def ingest(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/workspaces":
            return httpx.Response(201, json={})
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "records": [
                    {"index": i, "outcome": "rejected", "reason": "edit_not_supported"}
                    for i in range(len(body["records"]))
                ]
            },
        )

    real_client, real_sink = DiscordClient.__init__, IngestSink.__init__

    def client_init(self: DiscordClient, token: str, **_kwargs: Any) -> None:
        real_client(
            self,
            token,
            base_url="https://discord.test/api/v10",
            transport=httpx.MockTransport(fake.handler),
            sleep=lambda _s: None,
        )

    def sink_init(
        self: IngestSink, base_url: str, workspace: str, **_kwargs: Any
    ) -> None:
        real_sink(self, base_url, workspace, transport=httpx.MockTransport(ingest))

    monkeypatch.setattr(DiscordClient, "__init__", client_init)
    monkeypatch.setattr(IngestSink, "__init__", sink_init)
    token = tmp_path / "token"
    token.write_text("t")
    status = main(
        [
            "export",
            "--token-file",
            str(token),
            "--guild",
            GUILD,
            "--channel",
            "210",
            "--ingest-url",
            "http://honcho.test",
            "--workspace",
            "w",
        ]
    )
    assert status == 1


def test_other_honcho_error_text_is_not_shown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/workspaces":
            return httpx.Response(201, json={})
        return httpx.Response(400, json={"detail": "bad record: private message text"})

    sink = IngestSink("http://honcho.test", "w", transport=httpx.MockTransport(handler))
    sink.record({"content": "private message text"})
    with pytest.raises(IngestError) as raised:
        sink.close([], {})
    assert "private message text" not in str(raised.value)

    def conflict(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Workspace w isn't auto"})

    with pytest.raises(IngestError, match="isn't auto"):
        IngestSink("http://honcho.test", "w", transport=httpx.MockTransport(conflict))


def test_channels_fails_when_thread_lists_are_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDiscord()
    original = fake.handler

    def archive_fails(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/threads/archived/public"):
            return httpx.Response(403, json={"message": "Missing Access"})
        return original(request)

    real_init = DiscordClient.__init__

    def client_init(self: DiscordClient, token: str, **_kwargs: Any) -> None:
        real_init(
            self,
            token,
            base_url="https://discord.test/api/v10",
            transport=httpx.MockTransport(archive_fails),
            sleep=lambda _s: None,
        )

    monkeypatch.setattr(DiscordClient, "__init__", client_init)
    token = tmp_path / "token"
    token.write_text("t")
    assert main(["channels", "--token-file", str(token), "--guild", GUILD]) == 1


def test_the_git_guard_expands_home_and_runs_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "repo" / ".git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    # Neither file exists: the guard must refuse before trying to read them.
    with pytest.raises(SystemExit, match="git checkout"):
        main(
            [
                "export",
                "--token-file",
                str(tmp_path / "missing-token"),
                "--channels-file",
                str(tmp_path / "missing-channels"),
                "--guild",
                GUILD,
                "--out",
                "~/repo/export",
            ]
        )


# --- honcho ingest from discord -------------------------------------------------


class _CliSink:
    """Stands in for the CLI's record sink."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def container(self, name: str) -> None:
        self.events.append(("container", name))

    def record(self, record: Any) -> None:
        self.events.append(("record", record["source_id"]))


def test_ingest_from_discord_hands_each_container_to_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from honcho_discord import source

    fake = FakeDiscord(page=3)
    token = tmp_path / "token"
    token.write_text("test-token")

    def fake_client(*_args: object, **_kwargs: object) -> DiscordClient:
        return _client(fake)

    monkeypatch.setattr(source, "DiscordClient", fake_client)
    sink = _CliSink()

    code = source.SOURCE.main(["--token-file", str(token), "--guild", GUILD], sink)

    assert code == 0
    records = [name for kind, name in sink.events if kind == "record"]
    assert records == [str(i) for i in range(1, 8)] + ["20", "30", "40", "50"]
    containers = [name for kind, name in sink.events if kind == "container"]
    assert containers and all(name.startswith("#") for name in containers)
    # Each container's records follow it: the first event names a container.
    assert sink.events[0][0] == "container"


def test_the_discord_source_is_registered() -> None:
    from importlib.metadata import entry_points

    from honcho_discord.source import SOURCE

    points = {p.name: p for p in entry_points(group="honcho.ingest_sources")}
    assert points["discord"].load() is SOURCE
    assert SOURCE.help


def test_the_discord_source_help_exits_cleanly() -> None:
    from honcho_discord.source import SOURCE

    with pytest.raises(SystemExit) as helped:
        SOURCE.main(["--help"], _CliSink())
    assert helped.value.code == 0


def _refused(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"message": "401: Unauthorized"})


def _refusing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    real_init = DiscordClient.__init__

    def client_init(self: DiscordClient, token: str, **_kwargs: Any) -> None:
        real_init(
            self,
            token,
            base_url="https://discord.test/api/v10",
            transport=httpx.MockTransport(_refused),
            sleep=lambda _s: None,
        )

    monkeypatch.setattr(DiscordClient, "__init__", client_init)


def test_a_refused_token_is_the_sources_error_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from honcho_discord.source import SOURCE

    _refusing_client(monkeypatch)
    token = tmp_path / "token"
    token.write_text("t")

    # The CLI reports a source's exit message as SOURCE_ERROR.
    with pytest.raises(SystemExit) as exited:
        SOURCE.main(["--token-file", str(token), "--guild", GUILD], _CliSink())
    assert exited.value.code == (
        f"Discord 401 on /guilds/{GUILD}/channels: 401: Unauthorized"
    )
    with pytest.raises(SystemExit) as exited:
        main(["channels", "--token-file", str(token), "--guild", GUILD])
    assert str(exited.value.code).startswith("Discord 401 on ")


def test_unreadable_token_and_channels_files_are_errors_not_crashes(
    tmp_path: Path,
) -> None:
    from honcho_discord.source import SOURCE

    missing = str(tmp_path / "missing")
    with pytest.raises(SystemExit, match="Can't read .*missing"):
        SOURCE.main(["--token-file", missing, "--guild", GUILD], _CliSink())
    binary = tmp_path / "binary"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(SystemExit, match="one line"):
        main(["channels", "--token-file", str(binary), "--guild", GUILD])
    token = tmp_path / "token"
    token.write_text("t")
    with pytest.raises(SystemExit, match="Can't read .*missing"):
        main(
            [
                "channels",
                "--token-file",
                str(token),
                "--guild",
                GUILD,
                "--channels-file",
                missing,
            ]
        )
