"""Historical import uses the real CLI and read-only SQLite source."""

import argparse
import copy
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from hermes_plugin_honcho import cli, import_state
from hermes_plugin_honcho import history_import as importer
from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_state import SessionDB
from honcho import Honcho


def test_import_command_is_registered():
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    args = parser.parse_args(["import", "--dry-run"])
    assert args.honcho_command == "import"
    assert args.days == 30
    assert args.source == "cli,desktop"
    assert args.dry_run is True
    assert args.yes is False


def run_cli(arguments):
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    args = parser.parse_args(arguments)
    try:
        args.func(args)
    except SystemExit as exc:
        return exc.code
    return 0


class FakeHonchoAPI:
    """Real SDK HTTP boundary with in-memory storage and scripted write failures."""

    def __init__(self):
        self.sessions = {}
        self.requests = []
        self.writes = []
        self.failure = None
        self.reverse_response = False
        self.closed_clients = []

    @staticmethod
    def session_response(workspace, session):
        return {
            "id": session,
            "workspace_id": workspace,
            "is_active": True,
            "created_at": "2020-01-01T00:00:00Z",
            "metadata": {},
            "configuration": {},
        }

    def handle(self, request):
        parts = request.url.path.strip("/").split("/")
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            (
                request.method,
                request.url.path,
                copy.deepcopy(body),
                dict(request.url.params),
            )
        )
        if parts == ["v3", "workspaces"]:
            return httpx.Response(200, json={"id": body["id"]})
        workspace = parts[2]
        if len(parts) == 4 and parts[3] == "sessions":
            session_id = body["id"]
            self.sessions.setdefault(
                (workspace, session_id), {"messages": [], "peers": {}}
            )
            return httpx.Response(
                200, json=self.session_response(workspace, session_id)
            )
        session_id = parts[4]
        session = self.sessions.get((workspace, session_id))
        if session is None:
            return httpx.Response(404, json={"detail": "Session not found"})
        if parts[5:] == ["peers"]:
            if request.method == "GET":
                peers = [
                    {
                        "id": peer_id,
                        "workspace_id": workspace,
                        "created_at": "2020-01-01T00:00:00Z",
                    }
                    for peer_id in session["peers"]
                ]
                return httpx.Response(
                    200,
                    json={
                        "items": peers,
                        "page": 1,
                        "size": 100,
                        "pages": 1,
                        "total": len(peers),
                    },
                )
            session["peers"].update(body)
            return httpx.Response(200, json={})
        if parts[5:] == ["messages", "list"]:
            messages = session["messages"]
            filters = body.get("filters") if body else None
            if filters:
                allowed = filters.get("OR", [filters])
                messages = [
                    m
                    for m in messages
                    if any(
                        all(m["metadata"].get(k) == v for k, v in f["metadata"].items())
                        for f in allowed
                    )
                ]
            size, page = (
                int(request.url.params.get("size", 50)),
                int(request.url.params.get("page", 1)),
            )
            return httpx.Response(
                200,
                json={
                    "items": messages[(page - 1) * size : page * size],
                    "page": page,
                    "size": size,
                    "total": len(messages),
                    "pages": math.ceil(len(messages) / size),
                },
            )
        if parts[5:] == ["messages"] and request.method == "POST":
            self.writes.append(copy.deepcopy(body["messages"]))
            failure = self.failure
            self.failure = None
            if failure == "reject":
                return httpx.Response(
                    403, json={"detail": "private-response-token-must-not-be-printed"}
                )
            if failure == "timeout_before":
                raise httpx.ReadTimeout("private-transport-details", request=request)
            if failure == "connection_before":
                raise httpx.ConnectError("private-transport-details", request=request)
            if failure == "server_before":
                return httpx.Response(500, json={"detail": "private-response-details"})
            accepted = (
                body["messages"][:1]
                if failure == "partial_accept"
                else body["messages"]
            )
            response = []
            for message in accepted:
                stored = {
                    **copy.deepcopy(message),
                    "id": f"message-{len(session['messages']) + 1}",
                    "workspace_id": workspace,
                    "session_id": session_id,
                    "token_count": 1,
                }
                # Match MessageCreate.sanitize_content at the actual API boundary.
                stored["content"] = stored["content"].replace("\x00", "")
                session["messages"].append(stored)
                response.append(stored)
            if failure == "timeout_after":
                raise httpx.ReadTimeout("private-transport-details", request=request)
            if failure == "server_after":
                return httpx.Response(500, json={"detail": "private-response-details"})
            if failure == "partial_response":
                response = response[:1]
            if self.reverse_response:
                response = response[::-1]
            return httpx.Response(200, json=response)
        raise AssertionError(
            f"Unexpected fake API route: {request.method} {request.url.path}"
        )

    def client(self, config):
        http_client = httpx.Client(transport=httpx.MockTransport(self.handle))
        client = Honcho(
            api_key=config.api_key,
            base_url=config.base_url,
            workspace_id=config.workspace_id,
            max_retries=0,
            http_client=http_client,
        )
        original_close = client._http.close

        def close():
            original_close()
            http_client.close()
            self.closed_clients.append(client)

        client._http.close = close
        return client


@pytest.fixture
def history_source(monkeypatch):
    # Keep the source outside HOME/.hermes, which the live-DB guard protects on POSIX.
    # HOME still points to the isolated test home; the guard remains enabled.
    home = Path(os.environ["HERMES_HOME"]).with_name("import-fixture")
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home.parent)
    monkeypatch.setattr(
        "hermes_plugin_honcho.client._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    config_path = home / "honcho.json"
    raw = {
        "apiKey": "obviously-fake-test-key",
        "baseUrl": "http://127.0.0.1:9876/v3",
        "workspace": "history-test",
        "peerName": "owner",
        "aiPeer": "helper",
        "enabled": True,
        "sessionStrategy": "global",
    }
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    db = SessionDB(db_path=home / "state.db")
    api = FakeHonchoAPI()
    real_factory = importer._private_client
    client_factory = MagicMock(side_effect=api.client)
    monkeypatch.setattr(importer, "_private_client", client_factory)
    now = time.time()

    def session(session_id="source-one", source="cli", **kwargs):
        db.create_session(
            session_id,
            source,
            profile_name=kwargs.pop("profile_name", "default"),
            **kwargs,
        )
        return session_id

    def message(
        session_id="source-one",
        role="user",
        content="A historical fact",
        timestamp=None,
        **kwargs,
    ):
        return db.append_message(
            session_id,
            role,
            content,
            timestamp=now - 86400 if timestamp is None else timestamp,
            **kwargs,
        )

    def configure(**changes):
        raw.update(changes)
        config_path.write_text(json.dumps(raw), encoding="utf-8")

    yield SimpleNamespace(
        db=db,
        api=api,
        factory=client_factory,
        home=home,
        config_path=config_path,
        ledger=home.parent / ".honcho" / "hermes-import-state.json",
        session=session,
        message=message,
        configure=configure,
        now=now,
        raw=raw,
        real_factory=real_factory,
    )
    db.close()


def test_dry_run_is_read_only_and_prints_no_content_or_credentials(
    history_source, monkeypatch, capsys
):
    source = history_source
    source.session()
    source.message(content="PRIVATE-CONVERSATION-TEXT")
    source.message(role="assistant", content="PRIVATE-ASSISTANT-TEXT")
    yaml_path = source.home / "config.yaml"
    yaml_path.write_text("honcho:\n  timeout: 7\n", encoding="utf-8")
    before_config = source.config_path.read_bytes()
    before_yaml = yaml_path.read_bytes()
    before_db = (source.home / "state.db").read_bytes()
    wal = source.home / "state.db-wal"
    before_wal = wal.read_bytes() if wal.exists() else None
    oauth = MagicMock(side_effect=AssertionError("dry-run refreshed OAuth"))
    monkeypatch.setattr("hermes_plugin_honcho.client._refresh_oauth", oauth)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        MagicMock(side_effect=AssertionError("writable config loader")),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        MagicMock(side_effect=AssertionError("backup-writing loader")),
    )

    assert run_cli(["import", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Source messages: 2; message chunks: 2" in output
    assert "Logical sessions: 1; target sessions: 1" in output
    assert "PRIVATE-" not in output
    assert source.raw["apiKey"] not in output
    assert not source.ledger.exists() and not source.ledger.parent.exists()
    assert not (source.home / "backups").exists()
    assert source.config_path.read_bytes() == before_config
    assert yaml_path.read_bytes() == before_yaml
    assert (source.home / "state.db").read_bytes() == before_db
    if before_wal is not None:
        assert wal.read_bytes() == before_wal
    source.factory.assert_not_called()
    oauth.assert_not_called()


def test_fixture_cli_import_then_noop_then_append(history_source, capsys):
    source = history_source
    source.session()
    user_id = source.message(content="Original user text")
    assistant_id = source.message(role="assistant", content="Original assistant text")
    assert run_cli(["import", "--yes"]) == 0
    messages = source.api.sessions[("history-test", "history-test")]["messages"]
    assert [m["peer_id"] for m in messages] == ["owner", "helper"]
    assert [m["content"] for m in messages] == [
        "Original user text",
        "Original assistant text",
    ]
    assert [m["metadata"]["hermes_message_id"] for m in messages] == [
        user_id,
        assistant_id,
    ]
    for message in messages:
        assert datetime.fromisoformat(
            message["created_at"].replace("Z", "+00:00")
        ).timestamp() == pytest.approx(source.now - 86400, abs=1e-6)
        assert message["metadata"]["backfill"] is True
        assert message["metadata"]["hermes_root_session_id"] == "source-one"
        assert message["metadata"]["hermes_session_id"] == "source-one"
    assert len(source.api.writes) == 1
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1
    source.message(content="Only this appended row is new")
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 2
    assert len(source.api.writes[-1]) == 1
    assert source.api.writes[-1][0]["content"] == "Only this appended row is new"
    ledger = source.ledger.read_text(encoding="utf-8")
    assert "Original user text" not in ledger and source.raw["apiKey"] not in ledger
    assert len(source.api.closed_clients) == 2
    assert "Import complete" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args",
    [
        ["--days", "0"],
        ["--days", "-5"],
        ["--source", "unknown-source"],
        ["--before", "2026-01-01T00:00:00"],
        ["--before", "invalid"],
    ],
)
def test_invalid_parameters_do_not_call_sdk(history_source, args):
    assert run_cli(["import", *args, "--yes"]) == 2
    history_source.factory.assert_not_called()
    assert not history_source.ledger.exists()


def test_non_tty_requires_yes_without_side_effects(history_source, monkeypatch):
    source = history_source
    source.session()
    source.message()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert run_cli(["import"]) == 2
    source.factory.assert_not_called()
    assert not source.ledger.parent.exists()


def test_cancel_does_not_create_progress_or_remote_objects(history_source, monkeypatch):
    source = history_source
    source.session()
    source.message()
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert run_cli(["import"]) == 0
    source.factory.assert_not_called()
    assert not source.ledger.parent.exists()


def test_workspace_override_does_not_write_configuration(history_source):
    source = history_source
    source.session()
    source.message()
    original = source.config_path.read_bytes()
    assert run_cli(["import", "--workspace", "separate-workspace", "--yes"]) == 0
    assert ("separate-workspace", "separate-workspace") in source.api.sessions
    assert source.config_path.read_bytes() == original


@pytest.mark.parametrize("bad_config", ["{broken", "[]"])
def test_invalid_config_does_not_fall_back_to_environment(
    history_source, monkeypatch, bad_config
):
    source = history_source
    source.config_path.write_text(bad_config, encoding="utf-8")
    monkeypatch.setenv("HONCHO_API_KEY", "fake-environment-key")
    assert run_cli(["import", "--yes"]) == 2
    source.factory.assert_not_called()


def test_invalid_yaml_and_foreign_target_profile_fail_closed(history_source):
    source = history_source
    (source.home / "config.yaml").write_text("honcho: [unterminated", encoding="utf-8")
    assert run_cli(["import", "--yes"]) == 2
    (source.home / "config.yaml").unlink()
    assert run_cli(["--target-profile", "work", "import", "--yes"]) == 2
    source.factory.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["timeout_after", "server_after", "partial_response"]
)
def test_unknown_outcome_is_reconciled_without_resending(history_source, failure):
    source = history_source
    source.session()
    source.message(content="One")
    source.message(role="assistant", content="Two")
    source.api.failure = failure
    assert run_cli(["import", "--yes"]) == 3
    saved = import_state.read_state(source.ledger)
    assert next(iter(saved["scopes"].values()))["pending"] is not None
    assert len(source.api.writes) == 1
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1
    assert (
        next(iter(import_state.read_state(source.ledger)["scopes"].values()))["pending"]
        is None
    )
    filters = [
        request[2]["filters"]
        for request in source.api.requests
        if request[2] and request[2].get("filters")
    ]
    assert len(filters[-1]["OR"]) == 2
    assert all("honcho_import_key" in part["metadata"] for part in filters[-1]["OR"])


@pytest.mark.parametrize(
    "failure",
    ["timeout_before", "connection_before", "server_before", "partial_accept"],
)
def test_missing_or_partial_pending_batch_is_not_resent(
    history_source, failure, capsys
):
    source = history_source
    source.session()
    source.message(content="One")
    source.message(role="assistant", content="Two")
    source.api.failure = failure
    assert run_cli(["import", "--yes"]) == 3
    assert run_cli(["import", "--yes"]) == 3
    assert len(source.api.writes) == 1
    assert (
        next(iter(import_state.read_state(source.ledger)["scopes"].values()))["pending"]
        is not None
    )
    assert "private-" not in capsys.readouterr().out


def test_definite_rejection_keeps_cursors_and_allows_retry(history_source, capsys):
    source = history_source
    source.session()
    source.message()
    source.api.failure = "reject"
    assert run_cli(["import", "--yes"]) == 1
    scope = next(iter(import_state.read_state(source.ledger)["scopes"].values()))
    assert scope["pending"] is None
    assert scope["segments"]["source-one"]["last_scanned_message_id"] == 0
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 2
    assert "private-response" not in capsys.readouterr().out


def test_long_source_message_crosses_batches_without_early_cursor(
    history_source, monkeypatch
):
    source = history_source
    source.configure(messageMaxChars=20)
    source.session()
    row_id = source.message(content="X" * 850)
    snapshots = []
    original_write = importer.write_state

    def capture_write(path, state):
        snapshots.append(copy.deepcopy(state))
        original_write(path, state)

    monkeypatch.setattr(importer, "write_state", capture_write)
    assert run_cli(["import", "--yes"]) == 0
    assert [len(batch) for batch in source.api.writes] == [100, 5]
    assert all(len(m["content"]) <= 20 for batch in source.api.writes for m in batch)
    complete = "".join(
        m["content"].removeprefix("[continued] ")
        for batch in source.api.writes
        for m in batch
    )
    assert complete == "X" * 850
    scope_states = [next(iter(state["scopes"].values())) for state in snapshots]
    first_commit = next(scope for scope in scope_states if scope["pending"] is None)
    assert first_commit["segments"]["source-one"]["last_scanned_message_id"] == 0
    assert (
        first_commit["segments"]["source-one"]["rows"][str(row_id)]["confirmed_parts"]
        == 100
    )
    assert (
        scope_states[-1]["segments"]["source-one"]["last_scanned_message_id"] == row_id
    )


def test_originally_repeated_text_is_not_deduplicated_by_content(history_source):
    source = history_source
    source.session()
    source.message(content="A real repeated statement")
    source.message(content="A real repeated statement")
    assert run_cli(["import", "--yes"]) == 0
    messages = source.api.writes[0]
    assert len(messages) == 2
    assert (
        messages[0]["metadata"]["honcho_import_key"]
        != messages[1]["metadata"]["honcho_import_key"]
    )


def test_response_order_does_not_determine_confirmation(history_source):
    source = history_source
    source.session()
    source.message(content="One")
    source.message(role="assistant", content="Two")
    source.api.reverse_response = True
    assert run_cli(["import", "--yes"]) == 0
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1


def test_older_session_recent_messages_and_expanded_window(history_source):
    source = history_source
    result = source.db.import_sessions(
        [
            {
                "id": "old-session",
                "source": "cli",
                "started_at": source.now - 500 * 86400,
                "messages": [
                    {
                        "role": "user",
                        "content": "Forty days old",
                        "timestamp": source.now - 40 * 86400,
                    },
                    {
                        "role": "assistant",
                        "content": "Recent in an old session",
                        "timestamp": source.now - 3 * 86400,
                    },
                ],
            }
        ]
    )
    assert result["imported"] == 1
    assert run_cli(["import", "--yes"]) == 0
    assert [m["content"] for m in source.api.writes[0]] == ["Recent in an old session"]
    assert run_cli(["import", "--days", "60", "--yes"]) == 3
    assert len(source.api.writes) == 1
    assert (
        run_cli(["import", "--days", "60", "--workspace", "fresh-window", "--yes"]) == 0
    )
    assert [m["content"] for m in source.api.writes[1]] == [
        "Forty days old",
        "Recent in an old session",
    ]
    assert len(source.api.writes) == 2


def test_content_filters_match_live_without_tool_traces(history_source):
    source = history_source
    source.session()
    source.message(role="system", content="System instructions")
    source.message(
        content="Hello\n<memory-context>\nPrivate stale injection\n</memory-context>"
    )
    source.message(
        role="assistant",
        content="Intermediate",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "tool", "arguments": "{}"},
            }
        ],
    )
    source.message(role="tool", content="Tool result")
    source.message(
        role="assistant",
        content="Final answer",
        reasoning="Private chain of thought",
        finish_reason="stop",
    )
    source.message(content="[ASYNC COMPLETE] Machine event")
    source.message(role="assistant", content="Response to machine event")
    source.message(content="[{'not': 'a content object'}]")
    source.message(
        role="assistant",
        content=[
            {"type": "text", "text": "Multimodal text"},
            {"type": "image_url", "image_url": {"url": "file:///private.png"}},
        ],
    )
    source.message(content="Compaction summary", _compressed_summary=True)
    source.message(content="", display_kind="hidden")
    assert run_cli(["import", "--yes"]) == 0
    contents = [m["content"] for batch in source.api.writes for m in batch]
    assert contents == [
        "Hello",
        "Final answer",
        "[{'not': 'a content object'}]",
        "Multimodal text",
    ]
    assert not any(
        "Private" in text or "Tool result" in text or "Machine event" in text
        for text in contents
    )


def test_hidden_subagent_delegated_and_inactive_data_are_excluded(history_source):
    source = history_source
    source.session()
    source.message(content="Visible history")
    inactive_id = source.message(content="Rewound history")
    source.db._write_sql("UPDATE messages SET active = 0 WHERE id = ?", (inactive_id,))
    source.session("hidden")
    source.message("hidden", content="Hidden history")
    source.db._write_sql("UPDATE sessions SET hidden = 1 WHERE id = ?", ("hidden",))
    source.session("worker", source="subagent")
    source.message("worker", content="Subagent history")
    source.session("delegated", model_config={"_delegate_from": "source-one"})
    source.message("delegated", content="Delegated history")
    assert run_cli(["import", "--source", "cli,desktop,subagent", "--yes"]) == 0
    assert [m["content"] for batch in source.api.writes for m in batch] == [
        "Visible history"
    ]


def test_invalid_timestamp_is_partial_and_never_uses_upload_time(
    history_source, capsys
):
    source = history_source
    source.session()
    bad_id = source.message(content="Invalid timestamp row")
    source.db._write_sql(
        "UPDATE messages SET timestamp = ? WHERE id = ?", ("not-an-epoch", bad_id)
    )
    source.message(content="Valid timestamp row")
    assert run_cli(["import", "--yes"]) == 3
    assert [m["content"] for batch in source.api.writes for m in batch] == [
        "Valid timestamp row"
    ]
    assert "messages_invalid_timestamp: 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("global", "history-test"),
        ("per-directory", "project-dir"),
        ("per-repo", "project-dir"),
        ("per-session", "source-one"),
    ],
)
def test_mapping_reuses_live_resolver(history_source, strategy, expected, tmp_path):
    source = history_source
    cwd = tmp_path / "project-dir"
    cwd.mkdir()
    source.configure(sessionStrategy=strategy)
    source.session(cwd=str(cwd))
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    assert ("history-test", expected) in source.api.sessions


@pytest.mark.parametrize("strategy", ["per-directory", "per-repo"])
def test_missing_historical_cwd_uses_global_fallback_without_changing_live_default(
    history_source, monkeypatch, strategy, tmp_path
):
    source = history_source
    cwd = tmp_path / "current-folder"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    source.configure(sessionStrategy=strategy)
    source.session()
    source.message()
    config = HonchoClientConfig.from_global_config()
    assert config.resolve_session_name() == "current-folder"
    assert config.resolve_session_name(use_current_cwd=False) == "history-test"
    assert run_cli(["import", "--yes"]) == 0
    assert ("history-test", "history-test") in source.api.sessions


@pytest.mark.parametrize(
    "title_source, expected",
    [("user", "manual-name"), ("llm", "project-dir"), ("derived", "project-dir")],
)
def test_manual_title_provenance_and_map_override(
    history_source, title_source, expected, tmp_path
):
    source = history_source
    cwd = tmp_path / "project-dir"
    cwd.mkdir()
    source.configure(sessionStrategy="per-directory")
    source.session(cwd=str(cwd))
    source.db._write_sql(
        "UPDATE sessions SET title = ?, title_source = ? WHERE id = ?",
        ("manual name", title_source, "source-one"),
    )
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    assert ("history-test", expected) in source.api.sessions


def test_manual_session_map_and_ai_prefix(history_source, tmp_path):
    source = history_source
    cwd = tmp_path / "project-dir"
    cwd.mkdir()
    source.configure(
        sessionStrategy="per-directory",
        sessions={str(cwd): "mapped-session"},
        sessionAiPeerPrefix=True,
    )
    source.session(cwd=str(cwd))
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    assert ("history-test", "helper-mapped-session") in source.api.sessions


def gateway_session(
    source, session_id, user_id="42", chat_type="dm", platform="telegram"
):
    from gateway.config import Platform
    from gateway.session import SessionSource

    origin = SessionSource(
        platform=Platform(platform),
        chat_id=user_id,
        user_id=user_id,
        chat_type=chat_type,
    )
    source.session(
        session_id,
        source=platform,
        user_id=user_id,
        chat_id=user_id,
        chat_type=chat_type,
        session_key=f"{platform}:{user_id}",
        origin_json=json.dumps(origin.to_dict()),
    )
    source.message(session_id, content=f"History from {session_id}")


def test_gateway_is_opt_in_and_owner_alias_is_resolved(history_source):
    source = history_source
    source.configure(userPeerAliases={"42": "owner"})
    gateway_session(source, "owner-dm")
    assert run_cli(["import", "--yes"]) == 0
    assert not source.api.writes
    assert run_cli(["import", "--source", "telegram", "--yes"]) == 0
    message = source.api.writes[0][0]
    assert message["peer_id"] == "owner"
    assert ("history-test", "telegram-42") in source.api.sessions


def test_pin_is_not_proof_that_another_gateway_user_is_owner(history_source):
    source = history_source
    source.configure(pinUserPeer=True)
    gateway_session(source, "other-user-dm", user_id="not-owner")
    assert run_cli(["import", "--source", "telegram", "--yes"]) == 0
    assert not source.api.writes
    source.factory.assert_not_called()


def test_group_history_and_missing_origin_are_not_assigned_to_owner(history_source):
    source = history_source
    source.configure(userPeerAliases={"42": "owner"})
    gateway_session(source, "group", chat_type="group")
    source.session(
        "missing-origin", source="telegram", user_id="42", session_key="telegram:42"
    )
    source.message("missing-origin", content="Ambiguous history")
    assert run_cli(["import", "--source", "telegram", "--yes"]) == 3
    assert not source.api.writes
    source.factory.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_persisted_bot_dm_uses_live_a2a_rule_or_is_skipped(history_source, enabled):
    from hermes_plugin_honcho import HonchoMemoryProvider

    source = history_source
    source.configure(a2aSessions=enabled)
    gateway_session(source, "bot-dm", user_id="bot:other-profile")
    assert run_cli(["import", "--source", "telegram", "--yes"]) == 0
    if not enabled:
        assert not source.api.writes
        return
    config = HonchoClientConfig.from_global_config()
    provider = HonchoMemoryProvider()
    provider._session_key, provider._config = "telegram-bot-other-profile", config
    expected = provider._a2a_session_key({"id": "bot:other-profile", "is_bot": True})
    from hermes_plugin_honcho.session_peers import sanitize_peer_id

    expected = sanitize_peer_id(expected)
    assert ("history-test", expected) in source.api.sessions
    assert source.api.writes[0][0]["peer_id"] == "other-profile"
    assert source.api.writes[0][0]["peer_id"] != "owner"


def test_compression_lineage_is_scanned_once_without_absorbing_fork(
    history_source, capsys
):
    source = history_source
    source.configure(sessionStrategy="per-session")
    source.session("root")
    source.message("root", content="Root text")
    source.db.end_session("root", "compression")
    source.session("continuation", parent_session_id="root")
    source.message("continuation", content="Continuation text")
    source.session(
        "fork", parent_session_id="root", model_config={"_branched_from": "root"}
    )
    source.message("fork", content="Independent fork text")
    assert source.db.get_compression_lineage("continuation") == ["root", "continuation"]
    assert source.db.get_compression_lineage("fork") == ["fork"]
    assert run_cli(["import", "--yes"]) == 0
    messages = [m for batch in source.api.writes for m in batch]
    assert len(messages) == 3
    assert {m["metadata"]["hermes_root_session_id"] for m in messages} == {
        "root",
        "fork",
    }
    assert {session for workspace, session in source.api.sessions} == {
        "root",
        "continuation",
        "fork",
    }
    assert "Logical sessions: 2; target sessions: 3" in capsys.readouterr().out


def test_known_cutoff_does_not_override_existing_sequence_order(history_source):
    source = history_source
    source.session()
    source.message(
        content="History before activation", timestamp=source.now - 5 * 86400
    )
    source.message(content="History after activation", timestamp=source.now - 86400)
    source.api.sessions[("history-test", "history-test")] = {
        "peers": {},
        "messages": [
            {
                "id": "live-one",
                "peer_id": "owner",
                "content": "Existing live text",
                "session_id": "history-test",
                "workspace_id": "history-test",
                "created_at": datetime.fromtimestamp(
                    source.now - 86400, UTC
                ).isoformat(),
                "token_count": 1,
                "metadata": {},
            }
        ],
    }
    assert run_cli(["import", "--yes"]) == 3
    assert not source.api.writes
    before = datetime.fromtimestamp(source.now - 3 * 86400, UTC).isoformat()
    assert run_cli(["import", "--before", before, "--yes"]) == 3
    assert not source.api.writes
    assert (
        run_cli(["import", "--before", before, "--workspace", "fresh-history", "--yes"])
        == 0
    )
    assert [m["content"] for batch in source.api.writes for m in batch] == [
        "History before activation"
    ]


def test_ledger_loss_reconciles_remote_metadata_without_upload(history_source):
    source = history_source
    source.session()
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    source.ledger.unlink()
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1


def test_existing_observation_configuration_is_not_overwritten(history_source):
    source = history_source
    source.session()
    source.message()
    settings = {
        "owner": {"observe_me": False, "observe_others": False},
        "helper": {"observe_me": False, "observe_others": False},
    }
    source.api.sessions[("history-test", "history-test")] = {
        "messages": [],
        "peers": copy.deepcopy(settings),
    }
    assert run_cli(["import", "--yes"]) == 0
    assert source.api.sessions[("history-test", "history-test")]["peers"] == settings
    assert not [
        request
        for request in source.api.requests
        if request[0] == "POST" and request[1].endswith("/peers")
    ]


def test_prepare_save_failure_makes_zero_message_requests(history_source, monkeypatch):
    source = history_source
    source.session()
    source.message()
    monkeypatch.setattr(
        importer, "write_state", MagicMock(side_effect=OSError("fake save error"))
    )
    assert run_cli(["import", "--yes"]) == 1
    assert not source.api.writes


def test_save_failure_after_remote_acceptance_is_recovered(history_source, monkeypatch):
    source = history_source
    source.session()
    source.message()
    original = importer.write_state
    count = 0

    def fail_second(path, state):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("fake save error after success")
        original(path, state)

    monkeypatch.setattr(importer, "write_state", fail_second)
    assert run_cli(["import", "--yes"]) == 3
    assert (
        next(iter(import_state.read_state(source.ledger)["scopes"].values()))["pending"]
        is not None
    )
    monkeypatch.setattr(importer, "write_state", original)
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == 1


@pytest.mark.parametrize("change", ["content", "mapping", "chunk_size"])
def test_changed_payload_or_mapping_is_not_silently_reimported(history_source, change):
    source = history_source
    source.session()
    row_id = source.message()
    assert run_cli(["import", "--yes"]) == 0
    if change == "content":
        source.db._write_sql(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("Edited source text", row_id),
        )
    elif change == "mapping":
        source.configure(aiPeer="different-ai")
    else:
        source.configure(messageMaxChars=80)
    assert run_cli(["import", "--yes"]) == 2
    assert len(source.api.writes) == 1


def test_deleted_ledger_with_partial_remote_batch_never_resends(history_source):
    source = history_source
    source.session()
    source.message(content="First row")
    source.message(role="assistant", content="Second row")
    source.api.failure = "partial_accept"
    assert run_cli(["import", "--yes"]) == 3
    source.ledger.unlink()
    assert run_cli(["import", "--yes"]) == 3
    assert len(source.api.writes) == 1


def test_remote_overlap_is_checked_on_all_pages(history_source):
    source = history_source
    source.session()
    source.message()
    created_at = datetime.fromtimestamp(source.now - 86400, UTC)
    messages = []
    for index in range(101):
        content = f"Prior imported row {index}"
        metadata = {
            "backfill": True,
            "honcho_import_version": 1,
            "honcho_import_source": "a" * 64,
            "honcho_import_key": import_state.fingerprint(index),
            "hermes_session_id": "other-physical",
            "hermes_root_session_id": "other-logical",
            "hermes_message_id": index + 1,
            "part_index": 0,
            "part_count": 1,
            "payload_sha256": importer.payload_hash(
                content, "owner", created_at, "history-test"
            ),
        }
        messages.append(
            {
                "id": f"old-{index}",
                "peer_id": "owner",
                "content": content,
                "session_id": "history-test",
                "workspace_id": "history-test",
                "created_at": created_at.isoformat(),
                "token_count": 1,
                "metadata": metadata,
            }
        )
    messages[-1]["metadata"] = {}
    source.api.sessions[("history-test", "history-test")] = {
        "peers": {},
        "messages": messages,
    }
    assert run_cli(["import", "--yes"]) == 3
    assert not source.api.writes
    pages = [
        int(request[3]["page"])
        for request in source.api.requests
        if request[1].endswith("/messages/list")
    ]
    assert pages == [1, 2]
    assert all(request[1] != "/v3/workspaces" for request in source.api.requests)


def test_message_source_paging_and_batches_are_bounded(history_source, monkeypatch):
    source = history_source
    source.session()
    for index in range(205):
        source.message(content=f"Source row {index}")
    calls = []
    original = SessionDB.get_messages

    @wraps(original)
    def capture(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "get_messages", capture)
    assert run_cli(["import", "--yes"]) == 0
    assert [len(batch) for batch in source.api.writes] == [100, 100, 5]
    assert all(
        call["limit"] == 200
        and call["include_inactive"] is True
        and call["include_compacted"] is False
        for call in calls
    )
    assert any(call["after_id"] == 200 for call in calls)


def test_before_boundary_is_strict_and_normalized_to_utc(history_source):
    source = history_source
    source.session()
    boundary = datetime.fromtimestamp(source.now - 3 * 86400, UTC).replace(
        microsecond=0
    )
    source.message(content="Just before", timestamp=boundary.timestamp() - 1)
    source.message(content="Exactly at", timestamp=boundary.timestamp())
    source.message(content="After", timestamp=boundary.timestamp() + 1)
    offset = timezone(timedelta(hours=8))
    assert (
        run_cli(
            ["import", "--before", boundary.astimezone(offset).isoformat(), "--yes"]
        )
        == 0
    )
    assert [m["content"] for batch in source.api.writes for m in batch] == [
        "Just before"
    ]


def test_missing_database_is_not_created(history_source):
    source = history_source
    source.db.close()
    (source.home / "state.db").unlink()
    assert run_cli(["import", "--dry-run"]) == 2
    assert not (source.home / "state.db").exists()
    source.factory.assert_not_called()


def test_replaced_source_database_cannot_reuse_old_progress(history_source, tmp_path):
    source = history_source
    source.session()
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    source.db.close()
    replacement_path = tmp_path / "replacement.db"
    with SessionDB(db_path=replacement_path) as replacement:
        replacement.create_session("source-one", "cli", profile_name="default")
        replacement.append_message(
            "source-one", "user", "Different database", timestamp=source.now - 86400
        )
    os.replace(replacement_path, source.home / "state.db")
    assert run_cli(["import", "--yes"]) == 2
    assert len(source.api.writes) == 1


def test_different_server_uses_a_different_ledger_scope(history_source):
    source = history_source
    source.session()
    source.message()
    assert run_cli(["import", "--yes"]) == 0
    second_api = FakeHonchoAPI()
    source.factory.side_effect = second_api.client
    source.configure(baseUrl="http://127.0.0.1:9888")
    assert run_cli(["import", "--yes"]) == 0
    assert len(source.api.writes) == len(second_api.writes) == 1
    state = import_state.read_state(source.ledger)
    assert len(state["scopes"]) == 2
    assert {scope["identity"]["server"] for scope in state["scopes"].values()} == {
        "http://127.0.0.1:9876",
        "http://127.0.0.1:9888",
    }


def test_profiles_use_their_own_database_config_credentials_and_progress(
    history_source, monkeypatch
):
    source = history_source
    source.session()
    source.message(content="Default profile history")
    assert run_cli(["import", "--yes"]) == 0
    work_home = source.home / "profiles" / "work"
    work_home.mkdir(parents=True)
    work_raw = {
        **source.raw,
        "workspace": "work-history",
        "peerName": "work-owner",
        "apiKey": "fake-work-key",
    }
    (work_home / "honcho.json").write_text(json.dumps(work_raw), encoding="utf-8")
    with SessionDB(db_path=work_home / "state.db") as work_db:
        work_db.create_session("source-one", "cli", profile_name="work")
        work_db.append_message(
            "source-one", "user", "Work profile history", timestamp=source.now - 86400
        )
    monkeypatch.setenv("HERMES_HOME", str(work_home))
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "work")
    assert run_cli(["import", "--yes"]) == 0
    assert source.factory.call_args.args[0].api_key == "fake-work-key"
    assert source.api.writes[-1][0]["peer_id"] == "work-owner"
    assert source.api.writes[-1][0]["content"] == "Work profile history"
    scopes = import_state.read_state(source.ledger)["scopes"].values()
    assert {
        (scope["identity"]["profile"], scope["identity"]["workspace"])
        for scope in scopes
    } == {("default", "history-test"), ("work", "work-history")}


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.invalid",
        "https://example.invalid?token=fake-secret",
        "file:///tmp/server",
    ],
)
def test_credential_bearing_or_invalid_destination_is_not_previewed(
    history_source, url, capsys
):
    source = history_source
    source.configure(baseUrl=url)
    assert run_cli(["import", "--dry-run"]) == 2
    assert "password@" not in capsys.readouterr().out
    source.factory.assert_not_called()


def test_actual_private_client_disables_retries_and_does_not_join_shared_cache(
    history_source, monkeypatch
):
    source = history_source
    from hermes_plugin_honcho.client_cache import _client_slots

    config, _, _ = importer.load_import_config(SimpleNamespace())
    slots_before = dict(_client_slots)
    config.api_key = None
    monkeypatch.setenv("HONCHO_API_KEY", "unrelated-process-key")
    client = source.real_factory(config)
    try:
        assert client._http.max_retries == 0
        assert client._http.base_url == "http://127.0.0.1:9876"
        assert client._http.api_key == "local"
        assert config.api_key is None
        assert dict(_client_slots) == slots_before
        assert client._workspace_ensured is False
    finally:
        client._http.close()


@pytest.mark.parametrize("mismatch", ["url", "retries"])
def test_private_client_verifies_actual_url_and_retry_settings(
    history_source, monkeypatch, mismatch
):
    source = history_source
    config, _, _ = importer.load_import_config(SimpleNamespace())

    def mismatched_build(private_config, **kwargs):
        client = source.api.client(private_config)
        if mismatch == "url":
            client._http.base_url += "/v3"
        else:
            client._http.max_retries = 2
        return client

    monkeypatch.setattr("hermes_plugin_honcho.client._build_client", mismatched_build)
    with pytest.raises(importer.ImportInputError, match="differ from"):
        source.real_factory(config)
    assert not source.api.requests
    assert len(source.api.closed_clients) == 1


def test_configuration_change_after_preview_stops_before_oauth_or_sdk(
    history_source, monkeypatch
):
    source = history_source
    config, _, _ = importer.load_import_config(SimpleNamespace())
    source.configure(workspace="changed-after-preview")
    refresh = MagicMock(
        side_effect=AssertionError("OAuth must not run after a changed preview")
    )
    monkeypatch.setattr("hermes_plugin_honcho.client._refresh_oauth", refresh)
    with pytest.raises(importer.ImportInputError, match="changed after the preview"):
        source.real_factory(config)
    refresh.assert_not_called()


def test_malformed_managed_configuration_is_not_ignored(
    history_source, monkeypatch, tmp_path
):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("honcho: [unterminated", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    assert run_cli(["import", "--dry-run"]) == 2
    history_source.factory.assert_not_called()
