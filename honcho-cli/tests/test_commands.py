"""Command-level tests: init flow, destructive confirms, JSON output contract, exit codes.

Uses Typer's CliRunner against the real `app`. stdout is not a TTY under
CliRunner, so `use_json()` returns True and the CLI emits JSON/ndjson —
which is exactly what scripts and agents consume.
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
    """Isolated config file + clean HONCHO_* env."""
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.commands.setup.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    return f


@pytest.fixture
def runner():
    return CliRunner()


# --------------------------------------------------------------------------- #
# 1. `honcho init` end-to-end

class TestInit:
    def test_first_run_writes_exact_shape(self, cfg, runner):
        """First run with --api-key + --base-url writes apiKey + environmentUrl only."""
        with patch("honcho_cli.commands.setup._test_connection", return_value=(True, "OK")):
            result = runner.invoke(
                app,
                ["init", "--api-key", "test-key-123", "--base-url", "http://localhost:8000"],
            )
        assert result.exit_code == 0, result.stderr
        assert json.loads(cfg.read_text()) == {
            "environmentUrl": "http://localhost:8000",
            "apiKey": "test-key-123",
        }

    def test_preserves_foreign_keys(self, cfg, runner):
        """Second run must not clobber sibling-tool keys (`hosts`, `sessions`, ...)."""
        cfg.write_text(json.dumps({
            "apiKey": "old",
            "environmentUrl": "http://old.example",
            "hosts": {"claude_code": {"peerName": "ajspig"}},
            "sessions": {"/Users/ajspig": "home-chat"},
            "sessionStrategy": "chat-instance",
        }))
        with patch("honcho_cli.commands.setup._test_connection", return_value=(True, "OK")):
            result = runner.invoke(
                app,
                ["init", "--api-key", "new-key", "--base-url", "https://api.honcho.dev"],
            )
        assert result.exit_code == 0, result.stderr
        on_disk = json.loads(cfg.read_text())
        assert on_disk["apiKey"] == "new-key"
        assert on_disk["environmentUrl"] == "https://api.honcho.dev"
        assert on_disk["hosts"] == {"claude_code": {"peerName": "ajspig"}}
        assert on_disk["sessions"] == {"/Users/ajspig": "home-chat"}
        assert on_disk["sessionStrategy"] == "chat-instance"


# --------------------------------------------------------------------------- #
# 2. Destructive-confirm guards

class TestDestructiveConfirm:
    def test_workspace_delete_aborts_on_no(self, cfg, runner):
        """`workspace delete` without --yes: 'n' at prompt → no API call, non-zero exit."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        fake = MagicMock()
        fake.sessions.return_value = MagicMock(has_next_page=lambda: False, _raw_items=[])
        with patch("honcho_cli.main.get_client", return_value=(fake, MagicMock())), \
             patch("honcho_cli.commands.workspace._with_workspace", return_value=fake):
            result = runner.invoke(app, ["workspace", "delete", "ws1"], input="n\n")
        assert result.exit_code != 0
        fake.delete_workspace.assert_not_called()

    def test_session_delete_aborts_on_no(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        session = MagicMock()
        client = MagicMock()
        client.session.return_value = session
        config = MagicMock(session_id="s1", workspace_id="ws1")
        with patch("honcho_cli.main.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["session", "delete", "s1"], input="n\n")
        assert result.exit_code != 0
        session.delete.assert_not_called()


# --------------------------------------------------------------------------- #
# 3. JSON output contract — scripts pipe these

class TestJsonContract:
    def test_workspace_list_ndjson_shape(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        client = MagicMock()
        client.workspaces.return_value = ["ws-a", "ws-b"]
        with patch("honcho_cli.main.get_client", return_value=(client, MagicMock())):
            result = runner.invoke(app, ["workspace", "list"])
        assert result.exit_code == 0, result.stderr
        lines = [json.loads(line) for line in result.stdout.strip().splitlines() if line.strip()]
        assert lines == [{"id": "ws-a"}, {"id": "ws-b"}]


# --------------------------------------------------------------------------- #
# 4. Exit codes on error

class TestExitCodes:
    def test_no_workspace_scoped_exits_nonzero_with_code(self, cfg, runner):
        """Running a workspace-scoped command with no workspace → NO_WORKSPACE on stderr, exit 1."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        result = runner.invoke(app, ["peer", "list"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "NO_WORKSPACE"

    def test_not_found_exits_nonzero_with_code(self, cfg, runner):
        """SDK NotFoundError → structured error, exit 1."""
        from honcho import NotFoundError

        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        client = MagicMock()
        client.peer.return_value.get_card.side_effect = NotFoundError("not found")
        config = MagicMock(peer_id="missing", session_id="", workspace_id="ws1")
        with patch("honcho_cli.main.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["peer", "inspect", "missing", "-w", "ws1"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "PEER_NOT_FOUND"
