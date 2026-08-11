"""Command-level tests: init flow, destructive confirms, JSON output contract, exit codes.

Uses Typer's CliRunner against the real `app`. stdout is not a TTY under
CliRunner, so `use_json()` returns True and the CLI emits JSON —
which is exactly what scripts and agents consume.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from honcho_cli.commands.session import _next_page_command
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
# Helpers for `session view` — a fake SDK message, page, and Session

def _view_msg(i: int) -> MagicMock:
    return MagicMock(
        id=f"m{i}",
        peer_id="alice" if i % 2 == 0 else "bob",
        content=f"msg-{i}",
        token_count=i,
        metadata={},
        created_at=f"2026-01-01T00:00:00.{i:03d}Z",
    )


def _fake_page(
    items: list,
    *,
    total: int | None = None,
    page: int | None = None,
    pages: int | None = None,
    has_next: bool = False,
    next_page: MagicMock | None = None,
) -> MagicMock:
    """A stand-in for the SDK's ``SyncPage`` with explicit (non-mock) metadata."""
    fake = MagicMock()
    fake.items = items
    fake.total = total
    fake.page = page
    fake.pages = pages
    fake.has_next_page.return_value = has_next
    fake.get_next_page.return_value = next_page
    return fake


def _fake_session(page: MagicMock) -> MagicMock:
    session = MagicMock()
    session.messages.return_value = page
    return session


def _patch_view(session: MagicMock, *, peer_id: str = ""):
    """Patch `session view`'s client + read-only Session construction."""
    client = MagicMock()
    config = MagicMock(session_id="sess1", workspace_id="ws1", peer_id=peer_id)
    return _nested(
        patch("honcho_cli.commands.session.get_client", return_value=(client, config)),
        patch("honcho_cli.commands.session.Session", return_value=session),
    )


@contextmanager
def _nested(*managers):
    with ExitStack() as stack:
        yield [stack.enter_context(m) for m in managers]


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
            "hosts": {"claude_code": {"peerName": "user"}},
            "sessions": {"/Users/user": "home-chat"},
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
        assert on_disk["hosts"] == {"claude_code": {"peerName": "user"}}
        assert on_disk["sessions"] == {"/Users/user": "home-chat"}
        assert on_disk["sessionStrategy"] == "chat-instance"


# --------------------------------------------------------------------------- #
# 2. Destructive-confirm guards

class TestDestructiveConfirm:
    def test_workspace_delete_aborts_on_no(self, cfg, runner):
        """`workspace delete` without --yes: 'n' at prompt → no API call, non-zero exit."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        fake = MagicMock()
        fake.sessions.return_value = MagicMock(has_next_page=lambda: False, _raw_items=[])
        with patch("honcho_cli.commands.workspace.get_client", return_value=(fake, MagicMock())), \
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
        with patch("honcho_cli.commands.workspace.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["session", "delete", "s1"], input="n\n")
        assert result.exit_code != 0
        session.delete.assert_not_called()


# --------------------------------------------------------------------------- #
# 3. JSON output contract — scripts pipe these

class TestJsonContract:
    def test_workspace_list_json_array_shape(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        client = MagicMock()
        client.workspaces.return_value = ["ws-a", "ws-b"]
        with patch("honcho_cli.commands.workspace.get_client", return_value=(client, MagicMock())):
            result = runner.invoke(app, ["workspace", "list"])
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == [{"id": "ws-a"}, {"id": "ws-b"}]

    def test_workspace_search_preserves_full_content_in_json_mode(self, cfg, runner):
        cfg.write_text(json.dumps({
            "apiKey": "k",
            "environmentUrl": "http://localhost:8000",
            "workspace_id": "ws1",
        }))
        message = MagicMock(
            id="msg1",
            content="x" * 250,
            peer_id="peer1",
            session_id="sess1",
            created_at="2026-01-01T00:00:00Z",
        )
        client = MagicMock()
        client.search.return_value = [message]
        config = MagicMock(workspace_id="ws1")
        with patch("honcho_cli.commands.workspace.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["workspace", "search", "topic", "-w", "ws1"])
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload == [{
            "id": "msg1",
            "content": "x" * 250,
            "peer_id": "peer1",
            "session_id": "sess1",
            "created_at": "2026-01-01T00:00:00Z",
        }]

    def test_message_get_returns_single_json_object(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        msg = MagicMock(
            id="msg1",
            peer_id="peer1",
            content="hello",
            token_count=7,
            metadata={"kind": "demo"},
            created_at="2026-01-01T00:00:00Z",
        )
        session = MagicMock()
        session.get_message.return_value = msg
        client = MagicMock()
        client.session.return_value = session
        config = MagicMock(session_id="sess1", workspace_id="ws1")
        with patch("honcho_cli.commands.message.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["message", "get", "msg1", "-s", "sess1", "-w", "ws1"])
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == {
            "id": "msg1",
            "peer_id": "peer1",
            "content": "hello",
            "token_count": 7,
            "metadata": {"kind": "demo"},
            "created_at": "2026-01-01T00:00:00Z",
        }

    @pytest.mark.parametrize("last", ["0", "-5"])
    def test_message_list_rejects_non_positive_last(self, cfg, runner, last):
        """Non-positive --last silently returned an empty list via slice semantics."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        config = MagicMock(session_id="sess1", workspace_id="ws1", peer_id="")
        with patch("honcho_cli.commands.message.get_client", return_value=(MagicMock(), config)) as get_client:
            result = runner.invoke(
                app,
                ["message", "list", "sess1", "--last", last, "-w", "ws1"],
            )
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_FLAGS"
        get_client.assert_not_called()

    def test_session_view_json_is_chronological_window(self, cfg, runner):
        """`session view` returns the most recent N messages oldest→newest by default."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))

        # Server returns newest-first when reverse=True (m4, m3, m2, m1, m0).
        page = _fake_page([_view_msg(i) for i in range(4, -1, -1)], total=5)
        session = _fake_session(page)

        with _patch_view(session):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--last", "3", "-w", "ws1", "--json"],
            )

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        # Most recent 3 (m4,m3,m2) flipped to chronological: m2, m3, m4.
        assert [m["id"] for m in payload] == ["m2", "m3", "m4"]
        assert [m["content"] for m in payload] == ["msg-2", "msg-3", "msg-4"]
        session.messages.assert_called_once()
        assert session.messages.call_args.kwargs["reverse"] is True

    def test_session_view_rejects_all_with_last(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        config = MagicMock(session_id="sess1", workspace_id="ws1", peer_id="")
        with patch("honcho_cli.commands.session.get_client", return_value=(MagicMock(), config)):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--all", "--last", "10", "-w", "ws1"],
            )
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_FLAGS"

    def test_session_view_page_fetches_exact_server_page(self, cfg, runner):
        """`--page N --size M` hits the API page directly (oldest-first)."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))

        # Page 2 contents, already chronological.
        page = _fake_page(
            [_view_msg(i) for i in (50, 51, 52)],
            total=120,
            page=2,
            pages=3,
            has_next=True,
        )
        session = _fake_session(page)

        with _patch_view(session):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--page", "2", "--size", "50", "-w", "ws1", "--json"],
            )

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert [m["id"] for m in payload] == ["m50", "m51", "m52"]
        session.messages.assert_called_once_with(
            filters=None,
            page=2,
            size=50,
            reverse=False,
        )

    def test_session_view_page_with_reverse_pages_from_newest(self, cfg, runner):
        """`--reverse --page N` pages from the newest end, not the oldest one flipped."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        page = _fake_page([_view_msg(i) for i in (9, 8, 7)], total=30, page=1, pages=10)
        session = _fake_session(page)

        with _patch_view(session):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--page", "1", "--reverse", "-w", "ws1", "--json"],
            )

        assert result.exit_code == 0, result.stderr
        assert session.messages.call_args.kwargs["reverse"] is True
        # Server order is preserved: no local flip on top of a reversed fetch.
        assert [m["id"] for m in json.loads(result.stdout)] == ["m9", "m8", "m7"]

    def test_session_view_rejects_page_with_last(self, cfg, runner):
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        config = MagicMock(session_id="sess1", workspace_id="ws1", peer_id="")
        with patch("honcho_cli.commands.session.get_client", return_value=(MagicMock(), config)):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--page", "2", "--last", "10", "-w", "ws1"],
            )
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_FLAGS"

    @pytest.mark.parametrize(
        "args",
        [
            ["--size", "10"],  # --size requires --page
            ["--page", "1", "--size", "500"],  # over the server's 100 ceiling
            ["--page", "0"],
            ["--last", "0"],
        ],
    )
    def test_session_view_rejects_bad_flags_before_any_api_call(self, cfg, runner, args):
        """Flag validation runs before the client is built, so nothing reaches the API."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        with patch("honcho_cli.commands.session.get_client") as get_client:
            result = runner.invoke(app, ["session", "view", "sess1", *args, "-w", "ws1"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_FLAGS"
        get_client.assert_not_called()

    def test_session_view_does_not_create_the_session(self, cfg, runner):
        """`view` is read-only: it must not use the get-or-create client.session()."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        session = _fake_session(_fake_page([_view_msg(1)], total=1))
        client = MagicMock()
        config = MagicMock(session_id="sess1", workspace_id="ws1", peer_id="")

        with patch("honcho_cli.commands.session.get_client", return_value=(client, config)), \
             patch("honcho_cli.commands.session.Session", return_value=session) as session_cls:
            result = runner.invoke(app, ["session", "view", "sess1", "-w", "ws1", "--json"])

        assert result.exit_code == 0, result.stderr
        client.session.assert_not_called()
        session_cls.assert_called_once_with("sess1", client)

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {},
                "honcho session view s1 --page 2 --size 50",
            ),
            (
                {"reverse": True, "show_ids": True},
                "honcho session view s1 --page 2 --size 50 --reverse --ids",
            ),
            (
                {"workspace": "ws2", "peer": "alice"},
                "honcho session view s1 --page 2 --size 50 -w ws2 -p alice",
            ),
        ],
    )
    def test_next_page_command_carries_the_invocation_scope(self, kwargs, expected):
        """A copied hint must land on the same workspace, peer, and ordering."""
        opts = {"reverse": False, "show_ids": False, "workspace": None, "peer": None, **kwargs}
        assert _next_page_command("s1", 2, 50, **opts) == expected

    @pytest.mark.parametrize(
        ("session_id", "workspace", "expected_fragment"),
        [
            ("has space", None, "'has space'"),
            ("a;rm -rf x", None, "'a;rm -rf x'"),
            ("s1", "ws$(id)", "'ws$(id)'"),
            ("s1", "ws|tee", "'ws|tee'"),
        ],
    )
    def test_next_page_command_shell_quotes_identifiers(
        self, session_id, workspace, expected_fragment
    ):
        """IDs only reject ?#%/\\ and control chars, so spaces and metacharacters reach here."""
        hint = _next_page_command(
            session_id,
            2,
            50,
            reverse=False,
            show_ids=False,
            workspace=workspace,
            peer=None,
        )
        assert expected_fragment in hint

    def test_session_view_hint_carries_group_level_scope(self, cfg, runner):
        """-w/-p also parse at group level, where the command-level params are None."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))
        session = _fake_session(_fake_page([_view_msg(1)], total=10, page=1, pages=5))

        with _patch_view(session), patch("honcho_cli.output.use_json", return_value=False):
            result = runner.invoke(
                app,
                ["session", "-w", "ws2", "-p", "alice", "view", "sess1", "--page", "1"],
            )

        assert result.exit_code == 0, result.stderr
        assert "-w ws2" in result.stderr
        assert "-p alice" in result.stderr

    def test_session_view_last_walks_pages_past_the_page_cap(self, cfg, runner):
        """`--last N` above the 100-item server cap keeps walking instead of truncating."""
        cfg.write_text(json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"}))

        # Newest-first pages of 100: m149..m50, then m49..m0.
        second = _fake_page([_view_msg(i) for i in range(49, -1, -1)], total=150)
        first = _fake_page(
            [_view_msg(i) for i in range(149, 49, -1)],
            total=150,
            has_next=True,
            next_page=second,
        )
        session = _fake_session(first)

        with _patch_view(session):
            result = runner.invoke(
                app,
                ["session", "view", "sess1", "--last", "120", "-w", "ws1", "--json"],
            )

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert len(payload) == 120
        # Oldest of the 120-message tail first, newest last.
        assert payload[0]["id"] == "m30"
        assert payload[-1]["id"] == "m149"
        assert session.messages.call_args.kwargs["size"] == 100


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
        with patch("honcho_cli.commands.peer.get_client", return_value=(client, config)):
            result = runner.invoke(app, ["peer", "inspect", "missing", "-w", "ws1"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "PEER_NOT_FOUND"
