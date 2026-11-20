"""`honcho ingest` and the exclusions: import, status, derive, exclude/include."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from honcho import ConflictError, NotFoundError
from honcho.api_types import (
    IngestAccount,
    IngestDeriveResponse,
    IngestPerson,
    IngestRecordCounts,
    IngestRecordResult,
    IngestResponse,
    IngestSession,
    IngestStatusResponse,
    IngestSuggestion,
    IngestSuggestionAccepted,
    IngestSuggestionCounts,
    IngestSuggestionStageCounts,
    PeerConfig,
    QueueStatusResponse,
    SessionConfiguration,
)
from typer.testing import CliRunner

from honcho_cli.main import app

BASE_URL = "http://localhost:8000"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    # These commands are experimental: on here, off in the tests that check that.
    monkeypatch.setenv("HONCHO_EXPERIMENTAL", "1")
    f.write_text(json.dumps({"apiKey": "k", "environmentUrl": BASE_URL}))
    return f


@pytest.fixture
def runner():
    return CliRunner()


def _record(n: int) -> dict:
    return {
        "source": "discord",
        "source_instance": "g1",
        "source_id": f"m{n}",
        "created_at": "2026-11-04T10:00:00Z",
        "container": {"id": "c1", "name": "dev"},
        "author": {"id": f"u{n % 2}", "name": f"Person {n % 2}"},
        "content": f"message {n}",
    }


def _jsonl(tmp_path: Path, records: list[dict], *, extra: str = "") -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + extra)
    return path


def _ingest_response(count: int, offset: int = 0) -> IngestResponse:
    return IngestResponse(
        records=[
            IngestRecordResult(
                index=offset + i,
                outcome="created",
                session_id="dev-1",
                peer_id=f"person-{(offset + i) % 2}",
                derivation="deferred",
            )
            for i in range(count)
        ]
    )


def _client(*, api_key: str = "k", **methods) -> MagicMock:
    client = MagicMock()
    client.base_url = BASE_URL
    client._http.api_key = api_key
    for name, value in methods.items():
        setattr(client, name, value)
    return client


def _config(workspace: str = "ws1") -> MagicMock:
    return MagicMock(workspace_id=workspace, peer_id="", session_id="")


def _api_error(cls, _status: int):
    return cls("nope")


# --- import -------------------------------------------------------------------


def test_import_sends_the_file_deferred_and_summarizes(cfg, runner, tmp_path):
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    path = _jsonl(tmp_path, [_record(n) for n in range(3)], extra="\n")
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=ingest), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path), "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    [(chunk,), kwargs] = ingest.call_args
    assert kwargs == {"derive": "later"}
    assert [r.source_id for r in chunk] == ["m0", "m1", "m2"]
    summary = json.loads(result.stdout)
    assert summary == {
        "workspace": "ws1",
        "records": 3,
        "created": 3,
        "duplicate": 0,
        "rejected": 0,
        "rejected_reasons": {},
        "bot_records_held_back": 0,
        "sessions": 1,
        "authors": 2,
        "derive": "later",
    }


def _bot_thread() -> list[dict]:
    bot = {"id": "b1", "name": "ci", "bot": True}
    return [
        _record(0),
        {**_record(1), "author": bot},  # nobody answers this one
        {**_record(2), "author": bot},
        {**_record(3), "metadata": {"reply_to": "m2"}},  # a person replies to m2
    ]


def test_import_holds_back_bot_records_no_one_interacted_with(cfg, runner, tmp_path):
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    path = _jsonl(tmp_path, _bot_thread())
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=ingest), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path)])

    assert result.exit_code == 0, result.stderr
    [(chunk,), _] = ingest.call_args
    assert [r.source_id for r in chunk] == ["m0", "m2", "m3"]
    summary = json.loads(result.stdout)
    assert summary["records"] == 3
    assert summary["bot_records_held_back"] == 1


@pytest.mark.parametrize(("bots", "sent"), [("all", 4), ("none", 2)])
def test_import_bots_all_or_none(cfg, runner, tmp_path, bots, sent):
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    path = _jsonl(tmp_path, _bot_thread())
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=ingest), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path), "--bots", bots])

    assert result.exit_code == 0, result.stderr
    assert len(ingest.call_args.args[0]) == sent


def test_import_refuses_an_unknown_bots_choice(cfg, runner, tmp_path):
    path = _jsonl(tmp_path, [_record(0)])
    result = runner.invoke(app, ["ingest", "import", str(path), "--bots", "some"])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


def test_import_reads_a_large_file_in_chunks(cfg, runner, tmp_path):
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    path = _jsonl(tmp_path, [_record(n) for n in range(2500)])
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=ingest), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path), "--derive", "now"])

    assert result.exit_code == 0, result.stderr
    assert [len(call.args[0]) for call in ingest.call_args_list] == [1000, 1000, 500]
    assert {call.kwargs["derive"] for call in ingest.call_args_list} == {"now"}
    assert json.loads(result.stdout)["records"] == 2500


def test_a_bad_line_stops_the_import_with_its_line_number(cfg, runner, tmp_path):
    bad = _record(2)
    del bad["created_at"]
    ingest = MagicMock()
    path = _jsonl(tmp_path, [_record(0), _record(1), bad])
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=ingest), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path)])

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "INVALID_RECORD"
    assert error["details"] == {"line": 3}
    ingest.assert_not_called()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (_api_error(NotFoundError, 404), "WORKSPACE_NOT_FOUND"),
        (_api_error(ConflictError, 409), "NOT_AUTO_WORKSPACE"),
    ],
)
def test_import_explains_a_missing_or_regular_workspace(
    cfg, runner, tmp_path, error, code
):
    path = _jsonl(tmp_path, [_record(0)])
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest=MagicMock(side_effect=error)), _config()),
    ):
        result = runner.invoke(app, ["ingest", "import", str(path)])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == code


def test_an_unknown_derive_mode_is_refused(cfg, runner, tmp_path):
    path = _jsonl(tmp_path, [_record(0)])
    result = runner.invoke(app, ["ingest", "import", str(path), "--derive", "soon"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


# --- status and exclusions ------------------------------------------------------


def _person(peer_id: str, name: str, messages: int, modeled: bool = True) -> IngestPerson:
    return IngestPerson(
        peer_id=peer_id,
        display_name=name,
        accounts=[IngestAccount(source="discord", source_instance="g1", id=peer_id, display_name=name)],
        messages=messages,
        modeled=modeled,
    )


PEOPLE = [
    _person("jess-1", "Jessica", 30),
    _person("alex-1", "Alex", 20),
    _person("alex-2", "alex", 5),
    _person("stats-1", "Usage Stats", 3),
]
SESSIONS = [
    IngestSession(
        session_id="dev-1", display_name="#dev", source="discord",
        source_instance="g1", messages=40, modeled=True,
    ),
    IngestSession(
        session_id="random-1", display_name="#random", source="discord",
        source_instance="g1", messages=18, modeled=True,
    ),
]
STATUS = IngestStatusResponse(
    records=IngestRecordCounts(deferred=58, pending=0, queued=0),
    deferred_tokens=1234,
    people=4,
    people_not_modeled=0,
    sessions=2,
    sessions_not_modeled=0,
)


def _review_client(**extra) -> MagicMock:
    return _client(
        ingest_status=MagicMock(return_value=STATUS),
        ingest_people=MagicMock(return_value=iter(PEOPLE)),
        ingest_sessions=MagicMock(return_value=iter(SESSIONS)),
        **extra,
    )


def _status(runner, client) -> object:
    with patch(
        "honcho_cli.commands.ingest.get_client", return_value=(client, _config())
    ):
        return runner.invoke(app, ["ingest", "status", "-w", "ws1"])


def test_status_lists_people_and_sessions(cfg, runner):
    result = _status(runner, _review_client())

    assert result.exit_code == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["status"]["records"]["deferred"] == 58
    assert [p["peer_id"] for p in out["people"]] == ["jess-1", "alex-1", "alex-2", "stats-1"]
    assert [s["display_name"] for s in out["sessions"]] == ["#dev", "#random"]


def _exclude(runner, command: list[str], client: MagicMock, peer_cls=None, session_cls=None):
    patches = [
        patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config()))
    ]
    if peer_cls is not None:
        patches.append(patch("honcho_cli.commands.ingest.Peer", peer_cls))
    if session_cls is not None:
        patches.append(patch("honcho_cli.commands.ingest.Session", session_cls))
    for p in patches:
        p.start()
    try:
        return runner.invoke(app, [*command, "-w", "ws1"])
    finally:
        for p in patches:
            p.stop()


def _peer_cls(current: PeerConfig | None = None) -> tuple[MagicMock, MagicMock]:
    peer = MagicMock()
    peer.get_configuration.return_value = current or PeerConfig()
    return MagicMock(return_value=peer), peer


def test_exclude_a_person_by_their_number_from_status(cfg, runner):
    assert _status(runner, _review_client()).exit_code == 0
    peer_cls, peer = _peer_cls()

    result = _exclude(runner, ["peer", "exclude", "4"], _review_client(), peer_cls=peer_cls)

    assert result.exit_code == 0, result.stderr
    assert peer_cls.call_args.args[0] == "stats-1"
    peer.set_configuration.assert_called_once_with(PeerConfig(observe_me=False))
    assert json.loads(result.stdout) == {
        "peer_id": "stats-1", "display_name": "Usage Stats", "modeled": False,
    }


def test_exclude_a_person_by_display_name_or_id(cfg, runner):
    for ref in ("usage stats", "stats-1"):
        peer_cls, peer = _peer_cls()
        result = _exclude(runner, ["peer", "exclude", ref], _review_client(), peer_cls=peer_cls)
        assert result.exit_code == 0, result.stderr
        assert peer_cls.call_args.args[0] == "stats-1"


def test_a_name_two_people_share_is_ambiguous(cfg, runner):
    peer_cls, peer = _peer_cls()
    result = _exclude(runner, ["peer", "exclude", "Alex"], _review_client(), peer_cls=peer_cls)

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "AMBIGUOUS"
    assert sorted(error["details"]["matches"]) == ["alex-1", "alex-2"]
    peer.set_configuration.assert_not_called()


def test_an_unknown_person_is_never_created(cfg, runner):
    peer_cls, _ = _peer_cls()
    result = _exclude(runner, ["peer", "exclude", "Nobody"], _review_client(), peer_cls=peer_cls)

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "PERSON_NOT_FOUND"
    peer_cls.assert_not_called()


def test_a_number_needs_a_status_listing_first(cfg, runner):
    peer_cls, _ = _peer_cls()
    result = _exclude(runner, ["peer", "exclude", "2"], _review_client(), peer_cls=peer_cls)

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "NO_LISTING"


def test_include_keeps_other_settings(cfg, runner):
    peer_cls, peer = _peer_cls()
    result = _exclude(runner, ["peer", "include", "Jessica"], _review_client(), peer_cls=peer_cls)

    assert result.exit_code == 0, result.stderr
    peer.set_configuration.assert_called_once_with(PeerConfig(observe_me=True))


def test_exclude_a_session_turns_its_reasoning_off(cfg, runner):
    assert _status(runner, _review_client()).exit_code == 0
    session = MagicMock()
    session.configuration = SessionConfiguration.model_validate(
        {"summary": {"enabled": True}}
    )
    client = _review_client()
    client._workspace_ensured = False
    ensured_during_read: list[bool] = []

    def sessions_read(**_kwargs):
        ensured_during_read.append(client._workspace_ensured)
        return iter([session])

    session.set_configuration.side_effect = lambda _config: ensured_during_read.append(
        client._workspace_ensured
    )

    sessions = MagicMock(side_effect=sessions_read)
    client.sessions = sessions

    result = _exclude(runner, ["session", "exclude", "2"], client)

    assert result.exit_code == 0, result.stderr
    # Read through the list, never a get-or-create, and without ensuring the
    # workspace (which would create it if it had just been deleted).
    sessions.assert_called_once_with(filters={"id": "random-1"})
    # Neither the read nor the update can create the workspace.
    assert ensured_during_read == [True, True]
    assert client._workspace_ensured is False
    [(sent,), _] = session.set_configuration.call_args
    assert sent.reasoning.enabled is False
    assert sent.summary.enabled is True


def test_a_session_deleted_since_the_listing_is_not_recreated(cfg, runner):
    result = _exclude(
        runner,
        ["session", "exclude", "#random"],
        _review_client(sessions=MagicMock(return_value=iter([]))),
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "SESSION_NOT_FOUND"


def test_numbers_from_another_credential_are_not_used(cfg, runner):
    """Two tenants can share an instance and a workspace name."""
    assert _status(runner, _review_client(api_key="tenant-a")).exit_code == 0
    peer_cls, _ = _peer_cls()

    result = _exclude(
        runner,
        ["peer", "exclude", "4"],
        _review_client(api_key="tenant-b"),
        peer_cls=peer_cls,
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "NO_LISTING"
    peer_cls.assert_not_called()


def test_exclusions_need_an_auto_workspace(cfg, runner):
    client = _client(ingest_people=MagicMock(side_effect=_api_error(ConflictError, 409)))
    result = _exclude(runner, ["peer", "exclude", "Jessica"], client)

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "NOT_AUTO_WORKSPACE"


# --- derive ---------------------------------------------------------------------


def test_derive_releases_everything(cfg, runner):
    derive = MagicMock(return_value=IngestDeriveResponse(pending=58, remaining=0))
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_client(ingest_derive=derive), _config()),
    ):
        result = runner.invoke(app, ["ingest", "derive", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"pending": 58, "remaining": 0}


def test_derive_watch_follows_the_queue_until_it_is_empty(cfg, runner):
    def queue(pending: int, in_progress: int) -> QueueStatusResponse:
        return QueueStatusResponse(
            total_work_units=10, completed_work_units=10 - pending - in_progress,
            in_progress_work_units=in_progress, pending_work_units=pending,
        )

    pending = STATUS.model_copy(
        update={"records": IngestRecordCounts(deferred=0, pending=5, queued=53)}
    )
    done = STATUS.model_copy(
        update={"records": IngestRecordCounts(deferred=0, pending=0, queued=58)}
    )
    client = _client(
        ingest_derive=MagicMock(return_value=IngestDeriveResponse(pending=58, remaining=0)),
        # An empty queue with records still pending isn't finished.
        queue_status=MagicMock(
            side_effect=[queue(6, 2), queue(0, 0), queue(1, 1), queue(0, 0)]
        ),
        ingest_status=MagicMock(side_effect=[pending, pending, done, done]),
    )
    with (
        patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())),
        patch("honcho_cli.commands.ingest.time.sleep"),
    ):
        result = runner.invoke(app, ["ingest", "derive", "--watch", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"released": 58, "finished": True}
    assert client.queue_status.call_count == 4


# --- workspace create --auto ---------------------------------------------------------


def test_workspace_create_auto_asks_for_an_auto_workspace(cfg, runner):
    created = MagicMock()
    honcho_cls = MagicMock(return_value=created)
    with (
        patch("honcho_cli.commands.workspace.get_client", return_value=(_client(), _config(""))),
        patch("honcho_cli.commands.workspace.Honcho", honcho_cls),
    ):
        result = runner.invoke(app, ["workspace", "create", "acme", "--auto"])

    assert result.exit_code == 0, result.stderr
    assert honcho_cls.call_args.kwargs["auto"] is True
    assert json.loads(result.stdout) == {"workspace_id": "acme", "auto": True}


def test_workspace_create_auto_on_a_regular_workspace_explains(cfg, runner):
    created = MagicMock()
    created.get_configuration.side_effect = _api_error(ConflictError, 409)
    with (
        patch("honcho_cli.commands.workspace.get_client", return_value=(_client(), _config(""))),
        patch("honcho_cli.commands.workspace.Honcho", MagicMock(return_value=created)),
    ):
        result = runner.invoke(app, ["workspace", "create", "acme", "--auto"])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "NOT_AUTO_WORKSPACE"


# --- reset, link, unlink ----------------------------------------------------------


def test_reset_asks_first_and_waits_for_the_deletion(cfg, runner):
    from honcho.api_types import IngestResetResponse

    deleting = STATUS.model_copy(update={"sessions_being_deleted": 2})
    done = STATUS.model_copy(update={"sessions_being_deleted": 0})
    client = _client(
        ingest_reset=MagicMock(return_value=IngestResetResponse(sessions=["a", "b"])),
        ingest_status=MagicMock(side_effect=[deleting, done]),
    )
    with (
        patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())),
        patch("honcho_cli.commands.ingest.time.sleep"),
    ):
        refused = runner.invoke(app, ["ingest", "reset", "-w", "ws1"])
        result = runner.invoke(app, ["ingest", "reset", "-w", "ws1", "--yes"])

    assert refused.exit_code == 1
    assert json.loads(refused.stderr)["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {"sessions": 2, "waited": True}
    assert client.ingest_reset.call_count == 1
    assert client.ingest_status.call_count == 2


def test_link_resolves_people_and_keeps_the_first(cfg, runner):
    from honcho.api_types import IngestLinkResponse

    assert _status(runner, _review_client()).exit_code == 0
    link = MagicMock(
        return_value=IngestLinkResponse(
            peer_id="jess-1", merged=["stats-1"], accounts=PEOPLE[0].accounts
        )
    )
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_review_client(ingest_link=link), _config()),
    ):
        result = runner.invoke(app, ["ingest", "link", "1", "Usage Stats", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    link.assert_called_once_with(["jess-1", "stats-1"])


def test_a_refused_link_says_why(cfg, runner):
    link = MagicMock(side_effect=ConflictError("Peer stats-1 already has messages. Reset the import"))
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_review_client(ingest_link=link), _config()),
    ):
        result = runner.invoke(app, ["ingest", "link", "jess-1", "stats-1", "-w", "ws1"])

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "LINK_REFUSED"
    assert "Reset the import" in error["message"]


def test_link_needs_two_people(cfg, runner):
    result = runner.invoke(app, ["ingest", "link", "jess-1", "-w", "ws1"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


def test_unlink_finds_the_account_by_name_or_source_and_id(cfg, runner):
    from honcho.api_types import IngestUnlinkResponse

    linked = _person("jess-1", "Jessica", 30).model_copy(
        update={
            "accounts": [
                IngestAccount(source="discord", source_instance="g1", id="u1", display_name="Jessica"),
                IngestAccount(source="github", source_instance="acme", id="g9", display_name="jessd"),
            ]
        }
    )
    for ref in ("jessd", "github:g9", "g9"):
        unlink = MagicMock(
            return_value=IngestUnlinkResponse(peer_id="jessd-1", account=linked.accounts[1])
        )
        client = _client(
            ingest_people=MagicMock(return_value=iter([linked])), ingest_unlink=unlink
        )
        with patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())):
            result = runner.invoke(app, ["ingest", "unlink", "Jessica", ref, "-w", "ws1"])

        assert result.exit_code == 0, result.stderr
        unlink.assert_called_once_with(
            "jess-1", source="github", source_instance="acme", account_id="g9"
        )


def test_unlink_of_an_unknown_account_lists_the_persons_accounts(cfg, runner):
    client = _client(ingest_people=MagicMock(return_value=iter(PEOPLE)))
    with patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())):
        result = runner.invoke(app, ["ingest", "unlink", "Jessica", "nope", "-w", "ws1"])

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "ACCOUNT_NOT_FOUND"
    assert error["details"]["accounts"] == ["discord:jess-1 (Jessica)"]


def test_link_can_keep_a_peer_no_import_created(cfg, runner):
    from honcho.api_types import IngestLinkResponse

    link = MagicMock(
        return_value=IngestLinkResponse(peer_id="alice", merged=["stats-1"], accounts=[])
    )
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_review_client(ingest_link=link), _config()),
    ):
        result = runner.invoke(app, ["ingest", "link", "alice", "stats-1", "-w", "ws1"])

    assert result.exit_code == 0, result.stderr
    link.assert_called_once_with(["alice", "stats-1"])


def test_unlink_reads_an_account_id_that_contains_a_colon(cfg, runner):
    from honcho.api_types import IngestUnlinkResponse

    account = IngestAccount(source="github", source_instance="acme", id="org:user", display_name="org-user")
    person = _person("jess-1", "Jessica", 30).model_copy(update={"accounts": [account]})
    for ref in ("org:user", "github:org:user"):
        unlink = MagicMock(return_value=IngestUnlinkResponse(peer_id="x-1", account=account))
        client = _client(ingest_people=MagicMock(return_value=iter([person])), ingest_unlink=unlink)
        with patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())):
            result = runner.invoke(app, ["ingest", "unlink", "Jessica", ref, "-w", "ws1"])

        assert result.exit_code == 0, (ref, result.stderr)
        unlink.assert_called_once_with(
            "jess-1", source="github", source_instance="acme", account_id="org:user"
        )


def test_keep_takes_a_peer_id_literally(cfg, runner):
    """An ID that reads like a list number or an imported person's name."""
    from honcho.api_types import IngestLinkResponse

    for kept in ("123", "jessica"):
        link = MagicMock(
            return_value=IngestLinkResponse(peer_id=kept, merged=["stats-1"], accounts=[])
        )
        with patch(
            "honcho_cli.commands.ingest.get_client",
            return_value=(_review_client(ingest_link=link), _config()),
        ):
            result = runner.invoke(
                app, ["ingest", "link", "--keep", kept, "stats-1", "-w", "ws1"]
            )

        assert result.exit_code == 0, result.stderr
        link.assert_called_once_with([kept, "stats-1"])


def test_link_with_reset_finds_people_before_resetting(cfg, runner):
    """After a reset everyone has 0 messages and the list re-sorts."""
    from honcho.api_types import IngestLinkResponse, IngestResetResponse

    assert _status(runner, _review_client()).exit_code == 0
    calls: list[str] = []
    done = STATUS.model_copy(update={"sessions_being_deleted": 0})
    client = _review_client(
        ingest_reset=MagicMock(
            side_effect=lambda: calls.append("reset") or IngestResetResponse(sessions=["s"])
        ),
        ingest_link=MagicMock(
            side_effect=lambda ids: calls.append(f"link {ids}")
            or IngestLinkResponse(peer_id=ids[0], merged=ids[1:], accounts=[])
        ),
    )
    client.ingest_status = MagicMock(return_value=done)
    with patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())):
        result = runner.invoke(
            app, ["ingest", "link", "1", "4", "--reset", "--yes", "-w", "ws1"]
        )

    assert result.exit_code == 0, result.stderr
    assert calls == ["reset", "link ['jess-1', 'stats-1']"]


def test_a_refused_link_suggests_reset(cfg, runner):
    link = MagicMock(side_effect=ConflictError("Peer stats-1 already has messages. Reset the import first."))
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_review_client(ingest_link=link), _config()),
    ):
        result = runner.invoke(app, ["ingest", "link", "jess-1", "stats-1", "-w", "ws1"])

    assert result.exit_code == 1
    assert "--reset" in json.loads(result.stderr)["error"]["message"]


def test_reset_before_link_needs_confirmation(cfg, runner):
    link = MagicMock()
    with patch(
        "honcho_cli.commands.ingest.get_client",
        return_value=(_review_client(ingest_link=link), _config()),
    ):
        result = runner.invoke(app, ["ingest", "link", "jess-1", "stats-1", "--reset", "-w", "ws1"])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "CONFIRMATION_REQUIRED"
    link.assert_not_called()


_EXPERIMENTAL = [
    ["ingest", "import", "records.jsonl", "-w", "ws1"],
    ["ingest", "status", "-w", "ws1"],
    ["ingest", "derive", "-w", "ws1"],
    ["ingest", "reset", "-w", "ws1", "--yes"],
    ["ingest", "link", "1", "2", "-w", "ws1"],
    ["ingest", "unlink", "1", "alex", "-w", "ws1"],
    ["ingest", "from", "github", "-w", "ws1"],
    ["ingest", "suggestions", "-w", "ws1"],
    ["ingest", "suggestions", "accept", "1", "-w", "ws1"],
    ["ingest", "suggestions", "dismiss", "1", "-w", "ws1"],
    ["peer", "exclude", "1", "-w", "ws1"],
    ["peer", "include", "1", "-w", "ws1"],
    ["session", "exclude", "1", "-w", "ws1"],
    ["session", "include", "1", "-w", "ws1"],
    ["workspace", "create", "ws1", "--auto"],
]


@pytest.mark.parametrize("args", _EXPERIMENTAL, ids=lambda a: " ".join(a[:2]))
def test_experimental_commands_are_off_unless_opted_in(cfg, runner, monkeypatch, args):
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")
    reached = MagicMock(side_effect=AssertionError("ran without opting in"))
    with (
        patch("honcho_cli.commands.ingest.get_client", reached),
        patch("honcho_cli.commands.workspace.get_client", reached),
    ):
        result = runner.invoke(app, ["--json", *args])

    assert result.exit_code == 1, result.output
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "EXPERIMENTAL"
    assert "HONCHO_EXPERIMENTAL=1" in error["message"]
    reached.assert_not_called()


def test_a_plain_workspace_create_needs_no_opt_in(cfg, runner, monkeypatch):
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")
    client = MagicMock()
    with (
        patch(
            "honcho_cli.commands.workspace.get_client", return_value=(client, _config())
        ),
        patch("honcho_cli.commands.workspace._with_workspace", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["workspace", "create", "ws1"])

    assert result.exit_code == 0, result.output


def test_experimental_commands_say_so_in_help(cfg, runner, monkeypatch):
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")

    ingest_help = runner.invoke(app, ["ingest", "--help"])
    peer_help = runner.invoke(app, ["peer", "--help"])

    assert ingest_help.exit_code == 0
    assert "HONCHO_EXPERIMENTAL=1" in ingest_help.output
    for name in ("import", "status", "derive", "reset", "link", "unlink"):
        command_help = runner.invoke(app, ["ingest", name, "--help"])
        assert command_help.exit_code == 0, name
        assert "Experimental:" in command_help.output, name
    assert "Experimental:" in peer_help.output


def test_the_refusal_honors_a_commands_own_json_flag(cfg, runner, monkeypatch):
    # In a terminal, only `--json` makes the error JSON; here it comes after
    # the command, where the command itself parses it.
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")
    monkeypatch.setattr("honcho_cli.output.is_tty", lambda: True)

    result = runner.invoke(app, ["ingest", "derive", "-w", "ws1", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "EXPERIMENTAL"


# --- import --check -------------------------------------------------------------


def _no_client(*_args, **_kwargs):
    raise AssertionError("--check needs no workspace or credentials")


def test_import_check_validates_without_sending(cfg, runner, tmp_path):
    path = _jsonl(tmp_path, [_record(0), *_bot_thread()[1:]])
    with patch("honcho_cli.commands.ingest.get_client", _no_client):
        result = runner.invoke(app, ["ingest", "import", str(path), "--check"])

    assert result.exit_code == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["valid"] is True
    assert summary["records"] == 4
    assert summary["invalid"] == 0
    assert summary["sources"] == {"discord:g1": 4}
    assert summary["containers"] == 1
    assert summary["bot_records"] == 2
    # The bot post no one answered is held back; the one a person replied to isn't.
    assert summary["bot_records_held_back"] == 1
    assert summary["would_send"] == 3


def test_import_check_lists_every_invalid_line(cfg, runner, tmp_path):
    bad = {**_record(1)}
    del bad["source"]
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(_record(0)) + "\n" + json.dumps(bad) + "\n"
        + json.dumps(_record(2)) + "\nnot json\n"
    )
    with patch("honcho_cli.commands.ingest.get_client", _no_client):
        result = runner.invoke(app, ["ingest", "import", str(path), "--check"])

    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["valid"] is False
    assert summary["invalid"] == 2
    assert [e["line"] for e in summary["errors"]] == [2, 4]
    assert summary["errors"][0]["error"].startswith("source:")


# --- ingest from ----------------------------------------------------------------


class _FakeSource:
    help = "A fake chat"

    def __init__(self, containers: dict[str, list[dict]], code: int = 0) -> None:
        self.containers = containers
        self.code = code
        self.argv: list[str] | None = None

    def main(self, argv, sink):
        self.argv = argv
        if "--help" in argv:
            print("usage: honcho ingest from fake [--guild ID]")
            raise SystemExit(0)
        for name, records in self.containers.items():
            sink.container(name)
            for record in records:
                sink.record(record)
        return self.code


class _FakePoint:
    def __init__(self, name: str, source: object) -> None:
        self.name = name
        self.value = f"fake_connector:{name.upper()}"
        self._source = source

    def load(self):
        return self._source


def _installed(**sources):
    return patch(
        "honcho_cli.ingest_sources.installed",
        return_value={name: _FakePoint(name, src) for name, src in sources.items()},
    )


def _from(runner, args, client, **sources):
    with (
        _installed(**sources),
        patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())),
    ):
        return runner.invoke(app, ["ingest", "from", *args])


def _sending_client():
    ingest = MagicMock(side_effect=lambda chunk, derive: _ingest_response(len(chunk)))
    return _client(ingest=ingest, ingest_status=MagicMock(return_value=STATUS)), ingest


def test_from_lists_the_installed_sources(cfg, runner):
    with _installed(fake=_FakeSource({})), patch(
        "honcho_cli.commands.ingest.get_client", _no_client
    ):
        result = runner.invoke(app, ["ingest", "from"])

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"source": "fake", "help": "A fake chat", "entry_point": "fake_connector:FAKE"}
    ]


def test_from_runs_the_source_and_imports_each_container(cfg, runner):
    source = _FakeSource({"#dev": [_record(0)], "#ci": _bot_thread()[1:]})
    client, ingest = _sending_client()

    result = _from(
        runner, ["fake", "--guild", "1", "-w", "ws1", "--token-file", "t"], client,
        fake=source,
    )

    assert result.exit_code == 0, result.stderr
    # Everything that isn't this command's own goes to the source.
    assert source.argv == ["--guild", "1", "--token-file", "t"]
    sent = [r.source_id for (chunk,), _ in ingest.call_args_list for r in chunk]
    # #ci's unanswered bot post is held back; the rest are sent, a container at a time.
    assert sent == ["m0", "m2", "m3"]
    assert all(kwargs == {"derive": "later"} for _, kwargs in ingest.call_args_list)
    summary = json.loads(result.stdout)
    assert summary["source"] == "fake"
    assert summary["containers"] == 2
    assert summary["records"] == 3
    assert summary["bot_records_held_back"] == 1


def test_from_an_unknown_source_is_an_error(cfg, runner):
    client, _ = _sending_client()

    result = _from(runner, ["nope", "-w", "ws1"], client, fake=_FakeSource({}))

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "UNKNOWN_SOURCE"
    assert "fake" in error["message"]


def test_from_passes_help_to_the_source_without_credentials(cfg, runner):
    with _installed(fake=_FakeSource({})), patch(
        "honcho_cli.commands.ingest.get_client", _no_client
    ):
        result = runner.invoke(app, ["ingest", "from", "fake", "--help"])

    assert result.exit_code == 0
    assert "usage: honcho ingest from fake" in result.stdout


def test_from_refuses_a_workspace_that_isnt_auto_before_reading(cfg, runner):
    source = _FakeSource({"#dev": [_record(0)]})
    client = _client(ingest_status=MagicMock(side_effect=_api_error(ConflictError, 409)))

    result = _from(runner, ["fake", "-w", "ws1"], client, fake=source)

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "NOT_AUTO_WORKSPACE"
    assert source.argv is None  # never ran


def test_from_names_the_container_of_an_invalid_record(cfg, runner):
    bad = {**_record(1)}
    del bad["source"]
    client, ingest = _sending_client()

    result = _from(
        runner, ["fake", "-w", "ws1"], client,
        fake=_FakeSource({"#dev": [_record(0)], "#random": [bad]}),
    )

    assert result.exit_code == 1
    # What was stored before the error is said first; the error comes last.
    assert "1 records were stored before this" in result.stderr
    error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
    assert error["code"] == "INVALID_RECORD"
    assert error["details"]["container"] == "#random"
    # The container before it was complete, and was sent.
    assert [r.source_id for (chunk,), _ in ingest.call_args_list for r in chunk] == ["m0"]


def test_from_sends_what_it_read_and_keeps_the_source_exit_code(cfg, runner):
    client, ingest = _sending_client()

    result = _from(
        runner, ["fake", "-w", "ws1"], client,
        fake=_FakeSource({"#dev": [_record(0)]}, code=1),
    )

    assert result.exit_code == 1
    assert ingest.called
    assert json.loads(result.stdout)["records"] == 1


def test_from_reports_a_source_message(cfg, runner):
    class Refusing(_FakeSource):
        def main(self, argv, sink):
            raise SystemExit("no channels match --channel 9")

    client, _ = _sending_client()

    result = _from(runner, ["fake", "-w", "ws1"], client, fake=Refusing({}))

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "SOURCE_ERROR"
    assert "no channels match" in error["message"]


def test_import_check_applies_the_servers_rules(cfg, runner, tmp_path):
    long = {**_record(1), "content": "x" * 25_001}
    slashed = {**_record(2), "source": "bad/source"}
    path = _jsonl(tmp_path, [_record(0), long, slashed])
    with patch("honcho_cli.commands.ingest.get_client", _no_client):
        result = runner.invoke(app, ["ingest", "import", str(path), "--check"])

    assert result.exit_code == 1
    errors = json.loads(result.stdout)["errors"]
    assert [e["line"] for e in errors] == [2, 3]
    assert "25,001 characters" in errors[0]["error"]
    assert errors[1]["error"].startswith("source:")


def test_from_wants_the_source_name_first(cfg, runner):
    client, _ = _sending_client()

    result = _from(
        runner, ["--token-file", "t", "fake", "-w", "ws1"], client, fake=_FakeSource({})
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_from_help_shows_its_options_and_the_sources(cfg, runner, monkeypatch, flag):
    # Like `--help` on the other ingest commands: no opt-in needed, and text
    # even when the output isn't a terminal (as under the test runner).
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")
    with _installed(fake=_FakeSource({})), patch(
        "honcho_cli.commands.ingest.get_client", _no_client
    ):
        result = runner.invoke(app, ["ingest", "from", flag])

    assert result.exit_code == 0, result.output
    assert "Experimental:" in result.stdout
    assert "--bots" in result.stdout
    assert "fake" in result.stdout


def test_a_sources_help_needs_no_opt_in(cfg, runner, monkeypatch):
    monkeypatch.delenv("HONCHO_EXPERIMENTAL")
    with _installed(fake=_FakeSource({})), patch(
        "honcho_cli.commands.ingest.get_client", _no_client
    ):
        result = runner.invoke(app, ["ingest", "from", "fake", "--help"])

    assert result.exit_code == 0, result.output
    assert "usage: honcho ingest from fake" in result.stdout


# --- suggestions --------------------------------------------------------------


def _suggestion(n: int, confidence: str = "high", **changes) -> IngestSuggestion:
    data = {
        "id": f"sg{n}",
        "kind": "same_person",
        "confidence": confidence,
        "score": 3.0 - n,
        "status": "open",
        "linked": False,
        "accounts": [
            {"source": "discord", "source_instance": "g1", "id": f"d{n}",
             "display_name": "Alex", "handle": "alex_d", "peer_id": f"alex-{n}",
             "records": 3},
            {"source": "github", "source_instance": "github-com", "id": f"g{n}",
             "display_name": "alexdev", "handle": "alexdev", "peer_id": f"alexdev-{n}",
             "records": 1},
        ],
        "evidence": [
            {"signal": "self_reference", "level": "strong",
             "detail": "discord account alex_d gives alexdev on github as its own"},
        ],
        "stage": "rules",
        "refreshed_at": "2026-11-19T20:00:00Z",
    }  # fmt: skip
    data.update(changes)
    return IngestSuggestion.model_validate(data)


SUGGESTIONS = [_suggestion(1), _suggestion(2, "low")]


class _Page(list):
    total = len(SUGGESTIONS)


def _suggestions_client(**extra) -> MagicMock:
    return _client(
        ingest_refresh_suggestions=MagicMock(return_value=True),
        ingest_status=MagicMock(return_value=STATUS),
        ingest_suggestions=MagicMock(return_value=_Page(SUGGESTIONS)),
        **extra,
    )


def _invoke(runner, client, *args: str):
    with patch("honcho_cli.commands.ingest.get_client", return_value=(client, _config())):
        return runner.invoke(app, ["ingest", "suggestions", *args, "-w", "ws1"])


def test_suggestions_refresh_wait_and_list_numbered(cfg, runner):
    refreshing = STATUS.model_copy(
        update={
            "suggestions": IngestSuggestionCounts(
                rules=IngestSuggestionStageCounts(high=1, low=1), refreshing=True
            )
        }
    )
    client = _suggestions_client()
    client.ingest_status = MagicMock(side_effect=[refreshing, STATUS])
    with patch("honcho_cli.commands.ingest.REFRESH_POLL_SECONDS", 0):
        result = _invoke(runner, client, "--json")

    assert result.exit_code == 0, result.stderr
    client.ingest_refresh_suggestions.assert_called_once()
    assert client.ingest_status.call_count == 2
    assert [s["id"] for s in json.loads(result.stdout)] == ["sg1", "sg2"]

    # The numbers name the listed suggestions.
    accept = MagicMock(
        return_value=IngestSuggestionAccepted(suggestion=SUGGESTIONS[1], link=None)
    )
    result = _invoke(runner, _suggestions_client(ingest_accept_suggestion=accept),
                     "accept", "2")  # fmt: skip
    assert result.exit_code == 0, result.stderr
    accept.assert_called_once_with("sg2", keep=None)


def test_suggestions_without_refresh_show_the_label_and_evidence(cfg, runner):
    client = _suggestions_client()
    result = _invoke(runner, client, "--no-refresh", "--only", "rules:high")

    assert result.exit_code == 0, result.stderr
    client.ingest_refresh_suggestions.assert_not_called()
    kwargs = client.ingest_suggestions.call_args.kwargs
    assert (kwargs["stage"], kwargs["confidence"]) == ("rules", "high")
    # Each suggestion carries its label, as `accept` takes it.
    assert [s["label"] for s in json.loads(result.stdout)] == ["rules:high", "rules:low"]
    assert "alexdev on github as its own" in result.stdout


def test_an_unknown_label_is_refused(cfg, runner):
    result = _invoke(runner, _suggestions_client(), "--no-refresh", "--only", "high")
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"

    result = _invoke(runner, _suggestions_client(), "accept", "rules:medium")
    assert result.exit_code == 1
    assert "rules:high, rules:low" in json.loads(result.stderr)["error"]["message"]


def test_accept_by_label_resets_once_then_links_each(cfg, runner):
    from honcho.api_types import IngestLinkResponse, IngestResetResponse

    calls: list[str] = []
    done = STATUS.model_copy(update={"sessions_being_deleted": 0})

    def accept(suggestion_id: str, keep=None):
        calls.append(f"accept {suggestion_id}")
        return IngestSuggestionAccepted(
            suggestion=SUGGESTIONS[0],
            link=IngestLinkResponse(peer_id="alex-1", merged=["alexdev-1"], accounts=[]),
        )

    client = _suggestions_client(
        ingest_reset=MagicMock(
            side_effect=lambda: calls.append("reset") or IngestResetResponse(sessions=["s"])
        ),
        ingest_accept_suggestion=MagicMock(side_effect=accept),
    )
    client.ingest_status = MagicMock(return_value=done)
    client.ingest_suggestions = MagicMock(
        return_value=_Page([SUGGESTIONS[0], _suggestion(3, linked=True)])
    )
    result = _invoke(runner, client, "accept", "rules:high", "--reset", "--yes")

    assert result.exit_code == 0, result.stderr
    kwargs = client.ingest_suggestions.call_args.kwargs
    assert (kwargs["status"], kwargs["stage"], kwargs["confidence"]) == (
        "open",
        "rules",
        "high",
    )
    # Already one person: nothing to accept for it.
    assert calls == ["reset", "accept sg1"]


def test_accept_mixes_numbers_and_labels_once_each(cfg, runner):
    accept = MagicMock(
        return_value=IngestSuggestionAccepted(suggestion=SUGGESTIONS[0], link=None)
    )
    client = _suggestions_client(ingest_accept_suggestion=accept)
    client.ingest_suggestions = MagicMock(return_value=_Page([SUGGESTIONS[0]]))
    result = _invoke(runner, client, "accept", "sg2", "rules:high", "sg1", "--json")

    assert result.exit_code == 0, result.stderr
    assert [c.args[0] for c in accept.call_args_list] == ["sg2", "sg1"]


def test_nothing_to_accept_names_the_label(cfg, runner):
    client = _suggestions_client()
    client.ingest_suggestions = MagicMock(return_value=_Page([]))
    result = _invoke(runner, client, "accept", "rules:low")

    assert result.exit_code == 1
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "NOTHING_TO_ACCEPT"
    assert "rules:low" in error["message"]


def test_a_refused_accept_is_reported_and_fails(cfg, runner):
    refused = MagicMock(side_effect=ConflictError("Reset the import first."))
    result = _invoke(
        runner, _suggestions_client(ingest_accept_suggestion=refused),
        "accept", "sg1", "--json",
    )  # fmt: skip

    assert result.exit_code == 1
    assert json.loads(result.stdout)["failed"][0]["suggestion"] == "sg1"


def test_accept_needs_a_suggestion_and_keep_needs_exactly_one(cfg, runner):
    assert _invoke(runner, _suggestions_client(), "accept").exit_code == 1
    result = _invoke(runner, _suggestions_client(), "accept", "sg1", "sg2", "--keep", "x")
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"
    result = _invoke(runner, _suggestions_client(), "accept", "rules:high", "--keep", "x")
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_ARGUMENT"


def test_dismiss_by_id(cfg, runner):
    dismiss = MagicMock(return_value=SUGGESTIONS[0].model_copy(update={"status": "dismissed"}))
    result = _invoke(runner, _suggestions_client(ingest_dismiss_suggestion=dismiss),
                     "dismiss", "sg1")  # fmt: skip

    assert result.exit_code == 0, result.stderr
    dismiss.assert_called_once_with("sg1")


def test_a_batch_reports_what_was_done_before_a_failure(cfg, runner):
    def accept(suggestion_id: str, keep=None):
        if suggestion_id == "sg2":
            raise RuntimeError("server error")
        return IngestSuggestionAccepted(suggestion=SUGGESTIONS[0], link=None)

    client = _suggestions_client(ingest_accept_suggestion=MagicMock(side_effect=accept))
    result = _invoke(runner, client, "accept", "sg1", "sg2", "--json")
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert [a["suggestion"]["id"] for a in body["accepted"]] == ["sg1"]
    assert body["failed"][0]["suggestion"] == "sg2"

    def dismiss(suggestion_id: str):
        if suggestion_id == "sg2":
            raise ConflictError("was accepted")
        return SUGGESTIONS[0].model_copy(update={"status": "dismissed"})

    client = _suggestions_client(ingest_dismiss_suggestion=MagicMock(side_effect=dismiss))
    result = _invoke(runner, client, "dismiss", "sg1", "sg2", "--json")
    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert [d["id"] for d in body["dismissed"]] == ["sg1"]
    assert body["failed"][0]["suggestion"] == "sg2"
