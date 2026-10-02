"""Conclusion attribution and dialectic evidence (Honcho v3.2.0+).

Uses Typer's CliRunner against the real `app`. stdout is not a TTY under
CliRunner, so the CLI emits JSON — which is what scripts and agents consume.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from honcho_cli.main import app


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    f.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
    return f


@pytest.fixture
def runner():
    return CliRunner()


def _conclusion(cid: str, level: str = "inductive", source_ids: list[str] | None = None) -> MagicMock:
    return MagicMock(
        id=cid,
        content=f"content for {cid}",
        level=level,
        source_ids=source_ids if source_ids is not None else ["p1", "p2"],
        times_derived=3,
        observer_id="alice",
        observed_id="alice",
        session_id="s1",
        created_at="2026-09-15T00:00:00Z",
    )


class TestConclusionAttribution:
    def test_list_reports_level_source_ids_and_times_derived(self, cfg, runner):
        client = MagicMock()
        config = MagicMock(workspace_id="ws1", peer_id="alice")
        client.peer.return_value.conclusions.list.return_value = MagicMock(items=[_conclusion("c1")])

        with patch("honcho_cli.commands.conclusion.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["conclusion", "list", "--level", "inductive", "-p", "alice"])

        assert result.exit_code == 0
        row = json.loads(result.stdout)[0]
        assert row["level"] == "inductive"
        assert row["source_ids"] == ["p1", "p2"]
        assert row["times_derived"] == 3
        # --level reaches the server as a filter rather than being applied locally
        assert client.peer.return_value.conclusions.list.call_args.kwargs["filters"] == {"level": "inductive"}

    def test_list_rejects_an_unknown_level(self, cfg, runner):
        result = runner.invoke(app, ["conclusion", "list", "--level", "nonsense", "-p", "alice"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_LEVEL"

    def test_get_reports_ids_the_server_did_not_return(self, cfg, runner):
        client = MagicMock()
        config = MagicMock(workspace_id="ws1", peer_id="alice")
        client.conclusions.get_many.return_value = [_conclusion("c1", level="explicit", source_ids=[])]

        with patch("honcho_cli.commands.conclusion.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["conclusion", "get", "c1", "gone"])

        assert result.exit_code == 0
        assert client.conclusions.get_many.call_args.args[0] == ["c1", "gone"]
        err = json.loads(result.stderr)["error"]
        assert err["code"] == "MISSING_CONCLUSIONS"
        assert "gone" in err["message"]

    def test_derived_asks_for_conclusions_containing_the_premise(self, cfg, runner):
        client = MagicMock()
        config = MagicMock(workspace_id="ws1", peer_id="alice")
        client.conclusions.list.return_value = MagicMock(items=[_conclusion("c1")])

        with patch("honcho_cli.commands.conclusion.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["conclusion", "derived", "p1"])

        assert result.exit_code == 0
        assert client.conclusions.list.call_args.kwargs["filters"] == {"source_ids": {"contains": "p1"}}


class TestChatEvidence:
    def _evidence(self) -> MagicMock:
        return MagicMock(
            conclusions=[_conclusion("c1", level="explicit", source_ids=[])],
            messages=[
                MagicMock(id="m1", session_id="s1", peer_id="alice", created_at="2026-09-15T00:00:00Z"),
                MagicMock(id="m2", session_id="s1", peer_id="bob", created_at="2026-09-15T00:00:01Z"),
            ],
            tool_calls=[MagicMock(tool_name="search_memory", tool_input={"query": "q"})],
        )

    def test_peer_chat_returns_evidence_only_when_asked(self, cfg, runner):
        client = MagicMock()
        config = MagicMock(workspace_id="ws1", peer_id="alice", session_id=None)
        peer = client.peer.return_value
        peer.chat.return_value = "plain answer"

        with patch("honcho_cli.commands.peer.get_client", return_value=(client, config)):
            plain = runner.invoke(app, ["peer", "chat", "q", "-p", "alice"])
        assert plain.exit_code == 0
        assert json.loads(plain.stdout)["response"] == "plain answer"
        assert "include_evidence" not in peer.chat.call_args.kwargs

        peer.chat.return_value = MagicMock(content="answer", evidence=self._evidence())
        with patch("honcho_cli.commands.peer.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["peer", "chat", "q", "-p", "alice", "--evidence"])

        assert result.exit_code == 0
        assert peer.chat.call_args.kwargs["include_evidence"] is True
        evidence = json.loads(result.stdout)["evidence"]
        assert evidence["conclusions"][0]["level"] == "explicit"
        assert evidence["tool_calls"] == [{"tool_name": "search_memory", "tool_input": {"query": "q"}}]
        # Messages carry identity only, so they collapse to per-session counts
        assert evidence["messages"] == {"total": 2, "sessions": [{"session_id": "s1", "count": 2}]}

    def test_workspace_chat_passes_the_flag_through(self, cfg, runner):
        client = MagicMock()
        config = MagicMock(workspace_id="ws1", session_id=None)
        client.chat.return_value = MagicMock(content="answer", evidence=self._evidence())

        with patch("honcho_cli.commands.workspace.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["workspace", "chat", "q", "-w", "ws1", "--evidence"])

        assert result.exit_code == 0
        assert client.chat.call_args.kwargs["include_evidence"] is True
        assert json.loads(result.stdout)["evidence"]["messages"]["total"] == 2
