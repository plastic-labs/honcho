"""3.4.0 chat surface: response_format, uid, and named-scope keys.

Wire tests use a real SDK client with a mocked HTTP transport, so they assert
the JSON body Honcho 3.4.0 receives rather than the kwargs a MagicMock saw.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from honcho import Honcho, Peer
from honcho_cli import common
from honcho_cli.credentials import (
    arm_named_scope_client,
    is_named_scope_client,
    is_named_scope_key,
    named_scope_name,
)
from honcho_cli.main import app


def _jwt(payload: dict[str, object]) -> str:
    def seg(obj: dict[str, object]) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(payload)}.sig"


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

_WORKSPACE = {
    "id": "ws",
    "uid": "wuid",
    "metadata": {},
    "configuration": {},
    "created_at": "2026-01-01T00:00:00Z",
}


def _client(*, named_scope: bool = False) -> tuple[Honcho, MagicMock]:
    token = _jwt({"aud": "honcho:named-scope", "sc": "support", "w": "ws"})
    client = Honcho(workspace_id="ws", api_key=token, base_url="http://127.0.0.1:9")
    if named_scope:
        arm_named_scope_client(client, token)
    http = MagicMock()
    client._http = http
    return client, http


class TestNamedScopeKey:
    def test_audience_and_sc_claim(self):
        assert is_named_scope_key(_jwt({"aud": "honcho:named-scope", "sc": "support"}))
        assert is_named_scope_key(
            _jwt({"aud": ["honcho:named-scope"], "sc": "support"})
        )
        assert is_named_scope_key(_jwt({"sc": "support"}))

    def test_workspace_and_opaque_keys_are_not_named_scope(self):
        assert not is_named_scope_key(_jwt({"aud": "honcho", "w": "ws"}))
        assert not is_named_scope_key("hch-v3-managed")
        assert not is_named_scope_key("")
        assert not is_named_scope_key(None)

    def test_get_client_skips_workspace_provisioning(self, tmp_path, monkeypatch):
        from honcho_cli.common import get_client

        token = _jwt({"aud": "honcho:named-scope", "sc": "support", "w": "ws"})
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"apiKey": token, "environmentUrl": "http://127.0.0.1:9"})
        )
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", cfg)
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setenv("HONCHO_WORKSPACE_ID", "ws")
        for key in [
            k
            for k in list(__import__("os").environ)
            if k.startswith("HONCHO_") and k != "HONCHO_WORKSPACE_ID"
        ]:
            monkeypatch.delenv(key, raising=False)

        client, _config = get_client()
        assert client._workspace_ensured is True
        assert is_named_scope_client(client)
        assert named_scope_name(client) == "support"


class TestChatWire:
    def test_named_scope_peer_chat_sends_schema_scope_and_evidence(self):
        client, http = _client(named_scope=True)
        http.post.return_value = {
            "content": '{"answer": "yes"}',
            "evidence": {"conclusions": [], "messages": [], "tool_calls": []},
        }
        Peer("alice", client).chat(
            "how?",
            scope="support",
            response_format=SCHEMA,
            include_evidence=True,
        )
        http.post.assert_called_once()
        assert http.post.call_args.args[0] == "/v3/workspaces/ws/peers/alice/chat"
        body = http.post.call_args.kwargs["body"]
        assert body["scope"] == "support"
        assert body["response_format"] == SCHEMA
        assert body["include_evidence"] is True
        assert "session_id" not in body

    def test_default_client_provisions_workspace_before_peer_chat(self):
        client, http = _client(named_scope=False)
        http.post.side_effect = [_WORKSPACE, {"content": "hi"}]
        Peer("alice", client).chat("how?", scope="therapy")
        routes = [call.args[0] for call in http.post.call_args_list]
        assert routes == [
            "/v3/workspaces",
            "/v3/workspaces/ws/peers/alice/chat",
        ]
        assert http.post.call_args_list[1].kwargs["body"]["scope"] == "therapy"

    def test_workspace_chat_scope_is_a_string_or_list_on_the_wire(self):
        client, http = _client(named_scope=True)
        http.post.return_value = {"content": "ok"}
        client.chat("themes?", scope="smoke-scope", response_format=SCHEMA)
        body = http.post.call_args.kwargs["body"]
        assert http.post.call_args.args[0] == "/v3/workspaces/ws/chat"
        assert body["scope"] == "smoke-scope"
        assert body["response_format"] == SCHEMA
        assert isinstance(body["response_format"], dict)

        http.post.reset_mock()
        http.post.return_value = {"content": "ok"}
        client.chat("themes?", scope=["therapy", "work"])
        assert http.post.call_args.kwargs["body"]["scope"] == ["therapy", "work"]

    def test_session_get_and_scope_get_skip_workspace_provisioning(self):
        client, http = _client(named_scope=True)
        http.post.return_value = {
            "id": "in-scope",
            "uid": "suid",
            "is_active": True,
            "workspace_id": "ws",
            "metadata": {},
            "configuration": {},
            "created_at": "2026-01-01T00:00:00Z",
        }
        http.get.return_value = {
            "id": "support",
            "metadata": {},
            "created_at": "2026-01-01T00:00:00Z",
        }
        session = client.session("in-scope")
        assert session.uid == "suid"
        scope = client.get_scope("support")
        assert scope.id == "support"
        assert [call.args[0] for call in http.post.call_args_list] == [
            "/v3/workspaces/ws/sessions"
        ]
        assert http.get.call_args.args[0] == "/v3/workspaces/ws/scopes/support"


class TestChatCommand:
    @pytest.fixture
    def cfg(self, tmp_path, monkeypatch):
        f = tmp_path / "config.json"
        monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
        for key in [
            k for k in list(__import__("os").environ) if k.startswith("HONCHO_")
        ]:
            monkeypatch.delenv(key)
        f.write_text(
            json.dumps({"apiKey": "k", "environmentUrl": "http://localhost:8000"})
        )
        return f

    @pytest.fixture
    def runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_peer_chat_forwards_schema_object_and_nests_the_answer(self, cfg, runner):
        peer = MagicMock()
        peer.chat.return_value = '{"answer": "yes"}'
        client = MagicMock()
        client.peer.return_value = peer
        config = MagicMock(workspace_id="ws1", peer_id="alice", session_id="")
        with patch(
            "honcho_cli.commands.peer.get_client", return_value=(client, config)
        ):
            result = runner.invoke(
                app,
                [
                    "peer",
                    "chat",
                    "how?",
                    "-w",
                    "ws1",
                    "-p",
                    "alice",
                    "--response-format",
                    json.dumps(SCHEMA),
                ],
            )
        assert result.exit_code == 0, result.stderr
        assert peer.chat.call_args.kwargs["response_format"] == SCHEMA
        assert json.loads(result.stdout)["response"] == {"answer": "yes"}

    def test_response_format_file_and_rejections(self, cfg, runner, tmp_path):
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA))
        peer = MagicMock()
        peer.chat.return_value = '{"answer": "yes"}'
        client = MagicMock()
        client.peer.return_value = peer
        config = MagicMock(workspace_id="ws1", peer_id="alice", session_id="")
        with patch(
            "honcho_cli.commands.peer.get_client", return_value=(client, config)
        ):
            ok = runner.invoke(
                app,
                [
                    "peer",
                    "chat",
                    "how?",
                    "-p",
                    "alice",
                    "--response-format",
                    f"@{schema_path}",
                ],
            )
            assert ok.exit_code == 0, ok.stderr
            assert peer.chat.call_args.kwargs["response_format"] == SCHEMA

            for bad in ("[]", '{"type": "string"}', "not-json"):
                result = runner.invoke(
                    app,
                    ["peer", "chat", "how?", "-p", "alice", "--response-format", bad],
                )
                assert result.exit_code == 1
                assert (
                    json.loads(result.stderr)["error"]["code"]
                    == "INVALID_RESPONSE_FORMAT"
                )

    def test_named_scope_peer_chat_does_not_get_or_create_the_peer(self, cfg, runner):
        client = MagicMock()
        client._honcho_cli_named_scope = True
        client._honcho_cli_scope_name = "support"
        config = MagicMock(workspace_id="ws1", peer_id="alice", session_id="from-env")
        fake_peer = MagicMock()
        fake_peer.chat.return_value = "scoped"
        with (
            patch("honcho_cli.commands.peer.get_client", return_value=(client, config)),
            patch("honcho_cli.credentials.Peer", return_value=fake_peer) as peer_cls,
        ):
            result = runner.invoke(
                app, ["peer", "chat", "how?", "-p", "alice", "--scope", "support"]
            )
        assert result.exit_code == 0, result.stderr
        client.peer.assert_not_called()
        peer_cls.assert_called_once_with("alice", client)
        assert fake_peer.chat.call_args.kwargs["session"] is None
        assert fake_peer.chat.call_args.kwargs["scope"] == "support"

    def test_named_scope_rejects_session_and_other_scope(self, cfg, runner):
        client = MagicMock()
        client._honcho_cli_named_scope = True
        client._honcho_cli_scope_name = "support"
        config = MagicMock(workspace_id="ws1", peer_id="alice", session_id="")
        with patch(
            "honcho_cli.commands.peer.get_client", return_value=(client, config)
        ):
            session = runner.invoke(
                app, ["peer", "chat", "how?", "-p", "alice", "-s", "in-scope"]
            )
            # -s latches a process-global override; clear it so the next
            # invoke is judged on --scope alone.
            common._global_overrides.update(workspace=None, peer=None, session=None)
            other = runner.invoke(
                app, ["peer", "chat", "how?", "-p", "alice", "--scope", "other"]
            )
        assert session.exit_code == 1
        assert json.loads(session.stderr)["error"]["code"] == "INCOMPATIBLE_FLAGS"
        assert other.exit_code == 1
        assert json.loads(other.stderr)["error"]["code"] == "PERMISSION_ERROR"
        client.peer.assert_not_called()

    def test_session_list_nests_metadata_and_includes_uid(self, cfg, runner):
        page = MagicMock()
        page._raw_items = [
            MagicMock(
                id="smoke-test",
                is_active=True,
                metadata={"purpose": "smoke"},
                created_at="2026-01-01T00:00:00Z",
            )
        ]
        page._raw_items[0].uid = "sess-uid"
        page._raw_items[0].metadata = {"purpose": "smoke"}
        page.has_next_page.return_value = False
        client = MagicMock()
        client.sessions.return_value = page
        config = MagicMock(workspace_id="ws1", peer_id="")
        with patch(
            "honcho_cli.commands.session.get_client", return_value=(client, config)
        ):
            result = runner.invoke(app, ["session", "list", "-w", "ws1"])
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == [
            {
                "id": "smoke-test",
                "uid": "sess-uid",
                "is_active": True,
                "metadata": {"purpose": "smoke"},
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
