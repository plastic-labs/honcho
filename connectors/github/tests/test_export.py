"""Walking repositories through a fake GitHub GraphQL API, into JSONL and ingest."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, final

import httpx
import pytest
from honcho import HonchoError
from honcho_github.__main__ import main
from honcho_github.client import GitHubClient, GitHubError, graphql_url
from honcho_github.export import (
    IngestError,
    IngestSink,
    JsonlSink,
    SelectionError,
    export,
    select,
)
from honcho_github.mapping import NO_REASONING, Repo

from .fakes import TOKEN, FakeGitHub
from .fakes import client_for as _client


def by_time(records: list[dict[str, Any]]) -> bool:
    times = [r["created_at"] for r in records]
    return times == sorted(times)


def _read(directory: Path, name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (directory / name).read_text().splitlines()]


def test_export_writes_a_repo_as_one_session_in_time_order(tmp_path: Path) -> None:
    fake = FakeGitHub()
    out = tmp_path / "export"

    stats, _ = export(_client(fake), ["acme/web"], [], lambda: JsonlSink(out))

    records = _read(out, "records.jsonl")
    assert stats.complete
    assert [r["source_id"] for r in records] == [
        "PR_13",  # opened 2026-09-01: no --since, so the old PR is in
        "C_old",  # 11:00+02:00 is 09:00 UTC, so it sorts by UTC time
        "I_1",
        "IC_1",
        "IC_2",  # from the comments' second page
        "PR_12",
        "PC_1",
        "PC_2",
        "RC_1",
        "RC_2",
        "RC_3",  # from the thread's second page
        "PRR_1",
        "C_1",
    ]  # pending drafts (RC_P, PRR_P) are never exported
    assert by_time(records)
    assert {r["container"]["id"] for r in records} == {"R_web"}
    assert {r["source_instance"] for r in records} == {"github-com"}
    by_id = {r["source_id"]: r for r in records}
    # The commit with no account is skipped and counted.
    assert stats.skipped["commit_without_account"] == 1
    # Mentions resolve through a lookup; an unknown login isn't a participant.
    assert by_id["IC_2"]["participants"] == [
        {"name": "newbie", "handle": "newbie", "aliases": ["New"], "id": "33"}
    ]
    assert by_id["RC_2"]["participants"] == [
        {"name": "sam", "handle": "sam", "id": "22"}
    ]
    assert by_id["PRR_1"]["configuration"] == NO_REASONING
    assert by_id["PC_1"]["author"]["bot"] is True
    assert "machinery" not in by_id["PC_1"]["content"]
    assert by_id["C_1"]["content"].endswith("touched: src/auth/, tests/")
    assert by_id["PR_12"]["metadata"]["state"] == "merged"
    assert by_id["C_old"]["created_at"] == "2026-10-01T09:00:00Z"
    # `@coderabbitai` is the app; `@sam[bot]` is no one known, and never `sam`.
    assert by_id["PC_2"]["participants"] == [
        {"name": "coderabbitai", "handle": "coderabbitai", "id": "99", "bot": True}
    ]
    # Neither a bot nor a pending draft's mention is looked up.
    assert "sam[bot]" not in fake.looked_up
    assert "secretuser" not in fake.looked_up

    accounts = {a["login"]: a for a in _read(out, "accounts.jsonl")}
    assert set(accounts) == {"alex", "sam", "coderabbitai", "newbie"}
    assert accounts["coderabbitai"]["bot"] is True
    assert accounts["alex"]["name"] == "Alex Doe"
    summary = json.loads((out / "export.json").read_text())
    assert summary["complete"] is True
    assert summary["graphql_cost"] > 0


def test_a_person_comes_before_an_app_with_the_same_login(tmp_path: Path) -> None:
    fake = FakeGitHub()
    fake.users["coderabbitai"] = {
        "__typename": "User",
        "login": "coderabbitai",
        "databaseId": 77,
        "name": None,
    }
    out = tmp_path / "export"

    export(_client(fake), ["acme/web"], [], lambda: JsonlSink(out))

    by_id = {r["source_id"]: r for r in _read(out, "records.jsonl")}
    assert "coderabbitai" in fake.looked_up
    assert by_id["PC_2"]["participants"] == [
        {"name": "coderabbitai", "handle": "coderabbitai", "id": "77"}
    ]


def test_a_commits_text_doesnt_depend_on_since(tmp_path: Path) -> None:
    full, scoped = tmp_path / "full", tmp_path / "scoped"
    export(_client(FakeGitHub()), ["acme/web"], [], lambda: JsonlSink(full))
    # The commit is after this cutoff; its pull request was last updated before.
    export(
        _client(FakeGitHub()),
        ["acme/web"],
        [],
        lambda: JsonlSink(scoped),
        since="2026-11-06T01:00:00Z",
    )

    def commit(directory: Path) -> dict[str, Any]:
        return next(
            r for r in _read(directory, "records.jsonl") if r["source_id"] == "C_1"
        )

    assert commit(full)["content"] == commit(scoped)["content"]
    assert commit(scoped)["content"].endswith("touched: src/auth/, tests/")


def test_since_keeps_recent_threads_whole_and_recent_commits(tmp_path: Path) -> None:
    fake = FakeGitHub()
    out = tmp_path / "export"

    export(
        _client(fake),
        ["acme/web"],
        [],
        lambda: JsonlSink(out),
        since="2026-11-01T00:00:00Z",
    )

    ids = [r["source_id"] for r in _read(out, "records.jsonl")]
    assert "PR_13" not in ids  # updated before --since: the walk stops there
    assert "IC_1" in ids  # a recent issue comes whole, old comments included
    assert "C_1" in ids
    assert "C_old" not in ids  # committed before --since
    history = [v for q, v in fake.queries if "history(first:" in q]
    assert history[0]["since"] == "2026-11-01T00:00:00Z"


def test_an_org_means_its_public_repos_and_private_ones_must_be_named() -> None:
    fake = FakeGitHub()

    chosen = select(_client(fake), ["acme/secret"], ["acme"])

    assert [c.repo.name_with_owner for c in chosen] == [
        "acme/api",
        "acme/secret",
        "acme/web",
    ]


def test_a_bad_selection_makes_no_sink() -> None:
    fake = FakeGitHub()
    made: list[str] = []

    def make_sink() -> JsonlSink:
        made.append("sink")
        raise AssertionError("no sink for a bad selection")

    for repos, owners in (
        (["acme/missing"], []),
        ([], ["nobody"]),
        (["noslash"], []),
        ([], []),
    ):
        with pytest.raises(SelectionError):
            export(_client(fake), repos, owners, make_sink)
    assert made == []


def test_an_unreadable_repo_sends_nothing_and_the_export_is_incomplete(
    tmp_path: Path,
) -> None:
    fake = FakeGitHub()
    fake.forbidden = {"web"}
    out = tmp_path / "export"

    stats, _ = export(
        _client(fake), ["acme/web", "acme/api"], [], lambda: JsonlSink(out)
    )

    assert not stats.complete
    assert list(stats.unreadable) == ["acme/web"]
    assert _read(out, "records.jsonl") == []
    assert [c["name"] for c in _read(out, "containers.jsonl")] == ["acme/api"]
    assert TOKEN not in json.dumps(stats.unreadable)


def _scripted(responses: list[httpx.Response]) -> httpx.MockTransport:
    def handle(_request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    return httpx.MockTransport(handle)


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"data": {"viewer": {"login": "x"}}})


def test_the_client_waits_out_rate_limits_and_retries_server_errors() -> None:
    slept: list[float] = []
    secondary = {"message": "You have exceeded a secondary rate limit."}
    client = GitHubClient(
        TOKEN,
        transport=_scripted(
            [
                # The hour's budget is spent: wait for the reset.
                httpx.Response(
                    200,
                    headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1100"},
                    json={"errors": [{"type": "RATE_LIMITED"}]},
                ),
                # Retry-After wins, even with a reset already past.
                httpx.Response(
                    200,
                    headers={"Retry-After": "120", "X-RateLimit-Reset": "900"},
                    json={"errors": [{"type": "RATE_LIMITED"}]},
                ),
                httpx.Response(403, headers={"Retry-After": "7"}, json={}),
                # A secondary limit with no headers: a minute, doubled for every
                # limit this query has already waited out (the fourth and fifth).
                httpx.Response(403, json=secondary),
                httpx.Response(403, json=secondary),
                httpx.Response(502),
                _ok(),
            ]
        ),
        sleep=slept.append,
        now=lambda: 1_000.0,
    )

    assert client.query("query { viewer { login } }") == {"viewer": {"login": "x"}}
    assert slept == [101.0, 120.0, 7.0, 480.0, 900.0, 2.0]


def test_a_long_retry_after_is_honored_in_full() -> None:
    slept: list[float] = []
    client = GitHubClient(
        TOKEN,
        transport=_scripted(
            [httpx.Response(429, headers={"Retry-After": "7200"}), _ok()]
        ),
        sleep=slept.append,
    )

    client.query("query { viewer { login } }")

    assert slept == [7200.0]


def test_the_endpoint_and_the_instance_agree_on_the_host() -> None:
    assert graphql_url("API.github.com.") == "https://api.github.com/graphql"
    assert graphql_url("ghe.acme.test.") == "https://ghe.acme.test/api/graphql"


def test_a_query_error_is_reported_without_githubs_message() -> None:
    client = GitHubClient(
        TOKEN,
        transport=_scripted(
            [
                httpx.Response(
                    200,
                    json={
                        "errors": [
                            {
                                "type": "FORBIDDEN",
                                "message": f"cannot read {TOKEN}",
                                "path": ["repository", "issues"],
                            }
                        ]
                    },
                )
            ]
        ),
        sleep=lambda _s: None,
    )

    with pytest.raises(GitHubError) as raised:
        client.query("query { x }")

    assert str(raised.value) == "GitHub 200: FORBIDDEN at repository.issues"


def test_a_refused_token_fails_without_retrying_or_echoing_it() -> None:
    client = GitHubClient(
        TOKEN,
        transport=_scripted([httpx.Response(401, json={"message": TOKEN})]),
        sleep=lambda _s: pytest.fail("a refused token isn't retried"),
    )

    with pytest.raises(GitHubError) as raised:
        client.query("query { viewer { login } }")

    assert raised.value.retryable is False
    assert TOKEN not in str(raised.value)


@final
class FakeIngester:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[list[dict[str, Any]], str]] = []
        self.error = error

    def ingest(self, records: Iterable[Any], *, derive: Any = "now") -> Any:
        batch = list(records)
        self.calls.append((batch, derive))
        if self.error is not None:
            raise self.error

        @final
        class Result:
            def __init__(self, outcome: str, reason: str | None) -> None:
                self.outcome = outcome
                self.reason = reason

        @final
        class Response:
            records = [Result("created", None) for _ in batch[:-1]] + [
                Result("rejected", "content_conflict")
            ]

        return Response()


REPO = Repo(id="R_1", name_with_owner="acme/web", owner_id=None, url="u")


def _valid(source_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "source": "github",
        "source_instance": "github-com",
        "source_id": source_id,
        "created_at": "2026-11-04T10:00:00Z",
        "content": source_id,
        **fields,
    }


def test_the_ingest_sink_sends_each_repo_deriving_later() -> None:
    ingester = FakeIngester()
    sink = IngestSink("http://honcho", "acme", client=ingester)

    sink.container(REPO)
    sink.record(_valid("a"))
    sink.record(_valid("b"))
    sink.container(Repo(id="R_2", name_with_owner="acme/api", owner_id=None, url="u"))
    sink.record(_valid("c"))
    sink.close([], {})

    assert [
        ([r["source_id"] for r in batch], derive) for batch, derive in ingester.calls
    ] == [
        (["a", "b"], "later"),
        (["c"], "later"),
    ]
    assert sink.outcomes == {"created": 1, "rejected": 2}
    assert sink.rejections == {"content_conflict": 2}


def test_the_ingest_sink_holds_back_bot_records_no_one_answered() -> None:
    bot = {"id": "99", "name": "coderabbitai", "bot": True}
    person = {"id": "11", "name": "alex"}

    def record(
        source_id: str, author: dict[str, Any], **metadata: Any
    ) -> dict[str, Any]:
        return {
            "source": "github",
            "source_instance": "github-com",
            "source_id": source_id,
            "created_at": f"2026-11-04T10:0{len(source_id)}:00Z",
            "container": {"id": "R_1"},
            "author": author,
            "participants": [],
            "content": source_id,
            "metadata": {"thread": "pr:1", **metadata},
        }

    ingester = FakeIngester()
    sink = IngestSink("http://honcho", "acme", client=ingester)
    sink.container(REPO)
    sink.record(record("w", bot))  # a walkthrough nobody answered
    sink.record(record("nit", bot))
    sink.record(record("why?", person, reply_to="nit"))
    sink.close([], {})

    [(batch, _)] = ingester.calls
    assert [r["source_id"] for r in batch] == ["nit", "why?"]
    assert sink.held_back == 1

    everything = FakeIngester()
    sink = IngestSink("http://honcho", "acme", client=everything, bots="all")
    sink.container(REPO)
    sink.record(record("w", bot))
    sink.close([], {})
    assert [r["source_id"] for r in everything.calls[0][0]] == ["w"]


def test_an_ingest_failure_doesnt_echo_the_records() -> None:
    sink = IngestSink(
        "http://honcho",
        "acme",
        client=FakeIngester(
            HonchoError("records[0].content: private text", status=422)
        ),
    )
    sink.container(REPO)
    sink.record(_valid("a", content="private text"))

    with pytest.raises(IngestError) as raised:
        sink.close([], {})

    assert "private text" not in str(raised.value)
    assert "422" in str(raised.value)


def test_the_cli_refuses_an_output_inside_a_git_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    token = tmp_path.parent / f"{tmp_path.name}.token"
    token.write_text(TOKEN)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "export",
                "--token-file",
                str(token),
                "--repo",
                "acme/web",
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert "git checkout" in str(raised.value)


def test_the_cli_needs_a_workspace_to_send_to_ingest(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text(TOKEN)

    with pytest.raises(SystemExit) as raised:
        main(
            [
                "export",
                "--token-file",
                str(token),
                "--repo",
                "acme/web",
                "--ingest-url",
                "http://127.0.0.1:8000",
            ]
        )

    assert raised.value.code == 2


def test_a_hard_linked_output_file_is_replaced_not_overwritten(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("keep me")
    outside.chmod(0o644)
    out = tmp_path / "export"
    out.mkdir()
    os.link(outside, out / "records.jsonl")

    sink = JsonlSink(out)
    sink.record(_valid("a"))
    sink.close([], {})

    assert outside.read_text() == "keep me"
    assert outside.stat().st_mode & 0o777 == 0o644
    assert _read(out, "records.jsonl") == [_valid("a")]


# --- honcho ingest from github --------------------------------------------------


class _CliSink:
    """Stands in for the CLI's record sink."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def container(self, name: str) -> None:
        self.events.append(("container", name))

    def record(self, record: Any) -> None:
        self.events.append(("record", record["source_id"]))


def _source_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeGitHub, *argv: str
) -> tuple[int, _CliSink]:
    from honcho_github import source

    token = tmp_path / "token"
    token.write_text(TOKEN)

    def fake_client(*_args: object, **_kwargs: object) -> GitHubClient:
        return _client(fake)

    monkeypatch.setattr(source, "GitHubClient", fake_client)
    sink = _CliSink()
    code = source.SOURCE.main(["--token-file", str(token), *argv], sink)
    return code, sink


def test_ingest_from_github_hands_each_repository_to_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, sink = _source_main(tmp_path, monkeypatch, FakeGitHub(), "--repo", "acme/web")

    assert code == 0
    assert sink.events[0] == ("container", "acme/web")
    assert [name for kind, name in sink.events[1:] if kind == "record"] == [
        "PR_13",
        "C_old",
        "I_1",
        "IC_1",
        "IC_2",
        "PR_12",
        "PC_1",
        "PC_2",
        "RC_1",
        "RC_2",
        "RC_3",
        "PRR_1",
        "C_1",
    ]


def test_the_github_source_is_registered() -> None:
    from importlib.metadata import entry_points

    from honcho_github.source import SOURCE

    points = {p.name: p for p in entry_points(group="honcho.ingest_sources")}
    assert points["github"].load() is SOURCE
    assert SOURCE.help


def test_the_github_source_help_and_a_bad_selection_exit_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from honcho_github.source import SOURCE

    with pytest.raises(SystemExit) as helped:
        SOURCE.main(["--help"], _CliSink())
    assert helped.value.code == 0

    with pytest.raises(SystemExit) as refused:
        _source_main(tmp_path, monkeypatch, FakeGitHub())  # no --repo or --org
    assert isinstance(refused.value.code, str)
