"""`honcho config` renders the diagnostic report with every secret redacted.

Runs with ``--no-network`` so nothing is probed and Docker is not called; the
local-stack and integrations sections still read files on disk, from an
isolated config dir and a fake home. The JSON report is checked field by
field, both terminal views (summary and ``--all``) are rendered from the same
fixture and checked for leaks, and a ``--deep`` run with the network stubbed
checks exactly which keys are sent where.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import pytest
from honcho_cli.main import app
from typer.testing import CliRunner

FILE_KEY = "eyJhbGciOiJIUzI1NiJ9.file-body-0123456789.file-signature"
ENV_KEY = "hch-v3-env-key-0123456789abcdef"
CODEX_KEY = "hch-v3-codex-copied-key-0123456789"
OPENCLAW_KEY = "hch-v3-openclaw-key-0123456789"
CLAUDE_PROJECT_KEY = "hch-v3-claude-project-key-0123456789"
BARE_KEY = "hch-v3-windsurf-bare-key-0123456789"
CURSOR_KEY = "hch-v3-cursor-args-key-0123456789"
SELF_HOSTED_KEY = "hch-v3-selfhosted-key-0123456789"
DEFAULT_KEY = "hch-v3-default-leak-0123456789"
STATUSLINE_KEY = "hch-v3-statusline-prefix-0123456789"
SENTRY_DSN = "https://abcdef0123456789@o1.ingest.sentry.io/42"
# Every value here must stay out of every view. The tail of the list is the
# shapes a pattern-based redactor missed: they are hidden because their keys
# are not on the allowlist, not because anything recognised them.
SECRETS = (
    FILE_KEY, ENV_KEY, CODEX_KEY, OPENCLAW_KEY, CLAUDE_PROJECT_KEY, BARE_KEY, CURSOR_KEY,
    SELF_HOSTED_KEY, DEFAULT_KEY, STATUSLINE_KEY, "abcdef0123456789",
    "hch-at-access-token-value", "hch-rt-refresh-token-value", "hch-v2-hermes-host-key",
    "nested-agent-key-value", "sk-live-openai-key-value", "toml-jwt-secret-value",
    "pgpass-secret", "urlpass-secret", "redispass-secret", "9081726354",
    "Tr0ub4dor", "hunter2-prod-pass", "90817263", "739154", "hunter2pass",
    "dXNlcjpodW50ZXIycGFzcw==", "a1b2c3d4e5f60718293a4b5c6d7e8f90",
    # ...and the channels found after that: a JSON-valued setting, a nested
    # env name, a bare base64 line, a value after a flag, a key in a
    # workspace field, a token in a localhost webhook path
    "azure-json-key-0123456789abcd", "funckey0123456789abcdef", "c2VjcmV0LWJhc2U2NC12YWx1ZQ",
    "K7MDENG/bPxRfiCYEXAMPLEKEY", "hch-v3-pasted-into-workspace-0123", "tok-secret-path-0123",
)


def _assert_no_secrets(text: str) -> None:
    """No secret appears, even folded across lines by the terminal renderer."""
    squeezed = re.sub(r"\s+", "", text)
    for secret in SECRETS:
        assert secret not in squeezed, secret


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("honcho_cli.integrations.HOME", tmp_path / "home")
    for key in [k for k in os.environ if k.startswith(("HONCHO_", "LLM_"))]:
        monkeypatch.delenv(key)
    for key in ("CODEX_HOME", "HERMES_HOME", "OPENCODE_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HONCHO_NO_UPDATE_CHECK", "1")  # never touch the real update-check file
    return tmp_path


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _seed(cfg_dir) -> None:
    _write_json(cfg_dir / "config.json", {
        "apiKey": FILE_KEY,
        "environmentUrl": "http://proxyuser:urlpass-secret@localhost:8000",
        "oauth": {
            "accessToken": "hch-at-access-token-value",
            "refreshToken": "hch-rt-refresh-token-value",
            "accessExpiresAt": 0,
            "clientId": "honcho-cli",
            "scope": "write",
            "host": "http://proxyuser:urlpass-secret@localhost:8000",
        },
        "sessions": {"/a": "a-chat", "/b": "b-chat"},
        "hosts": {
            "hermes": {"baseUrl": "http://localhost:8000", "apiKey": "hch-v2-hermes-host-key"},
            "claude_code": {"workspace": "agents", "injection": {"sessionContextTokens": 1500}},
            "openclaw": {"agents": {"bot": {"apiKey": "nested-agent-key-value"}}},
            # a password as a number, secrets under a secret-named container, a PIN
            "myhost": {"password": 90817263, "credentials": {"prod": "hunter2-prod-pass"}, "pin": "739154",
                       "webhookUrl": "http://127.0.0.1:9911/hook/tok-secret-path-0123"},
        },
    })
    profile = cfg_dir / "profiles" / "local"
    profile.mkdir(parents=True)
    (profile / "docker-compose.yml").write_text("services: {}\n")
    _write_json(profile / "profile.json", {"apiPort": 8001, "dbPort": 5433, "redisPort": 6380})
    (profile / ".env").write_text(
        "API_PORT=8001\n"
        "LLM_OPENAI_API_KEY=sk-live-openai-key-value\n"
        "LLM_ANTHROPIC_API_KEY=your-api-key-here\n"
        "DERIVER_MODEL_CONFIG__TRANSPORT=openai\n"
        "SUMMARY_MAX_TOKENS_SHORT=400\n"
        "DB_URI=postgresql://honcho:pgpass-secret@db:5432/honcho\n"
        "DB_CONNECTION_URI=postgresql+psycopg://honcho:Tr0ub4dor#3xyz@db:5432/honcho\n"
        "CACHE_URL=redis://:redispass-secret@redis:6379/0\n"
        "POSTGRES_PASSWORD=9081726354\n"
        'SUMMARY_MODEL_CONFIG={"transport":"openai","model":"gpt-x","overrides":{"api_key":"azure-json-key-0123456789abcd"}}\n'
        "DERIVER_MODEL_CONFIG__OVERRIDES__PROVIDER_PARAMS__EXTRA_QUERY__CODE=funckey0123456789abcdef\n"
        "c2VjcmV0LWJhc2U2NC12YWx1ZQ==\n"
    )
    (profile / "config.toml").write_text(
        '[auth]\njwt_secret = "toml-jwt-secret-value"\n\n'
        "[deriver.model_config.overrides.provider_params.extra_headers]\n"
        'Ocp-Apim-Subscription-Key = "a1b2c3d4e5f60718293a4b5c6d7e8f90"\n'
    )

    home = cfg_dir / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    (codex / "config.toml").write_text(
        "[features]\nhooks = true\n\n[mcp_servers.honcho]\n"
        'url = "https://mcp.honcho.dev"\n'
        f'http_headers = {{ Authorization = "Bearer {CODEX_KEY}", X-Honcho-Workspace-ID = "codex" }}\n'
    )
    (codex / "hooks.json").write_text('{"hooks": {"Stop": [{"command": "node ~/.codex/honcho/codex-honcho.mjs"}]}}')
    # mcp-remote: the URL and key live in args, which are masked but never
    # parsed for testing; other secrets sit beside them.
    _write_json(home / ".cursor" / "mcp.json", {"mcpServers": {"honcho": {
        "command": "npx",
        "args": ["mcp-remote", "https://mcp.honcho.dev", "--header", f"Authorization: Bearer {CURSOR_KEY}",
                 "--header", "Authorization: Basic dXNlcjpodW50ZXIycGFzcw==", "--db-password", "hunter2pass",
                 "--aws-secret", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"],
        "env": {"SENTRY_DSN": SENTRY_DSN, "AUTH_HEADER": f"Bearer ${{HONCHO_API_KEY:-{DEFAULT_KEY}}}"},
    }}})
    # Zed's settings are JSONC; mcp-remote to a self-hosted MCP is never tested.
    zed = home / ".config" / "zed" / "settings.json"
    zed.parent.mkdir(parents=True)
    zed.write_text("// Zed settings\n" + json.dumps({"context_servers": {"honcho": {
        "command": "npx",
        "args": ["mcp-remote", "http://localhost:8787", "--header", f"Authorization: Bearer {SELF_HOSTED_KEY}"],
    }}}))
    # The hosted MCP server needs "Bearer "; a bare key is malformed, not tested.
    _write_json(home / ".codeium" / "windsurf" / "mcp_config.json", {"mcpServers": {"honcho": {
        "serverUrl": "https://mcp.honcho.dev", "headers": {"Authorization": BARE_KEY},
    }}})
    # Claude Code: a reference at user scope (shown as written), and the
    # local-scope entry `claude mcp add` writes by default.
    _write_json(home / ".claude.json", {
        "mcpServers": {"honcho": {"type": "http", "url": "https://mcp.honcho.dev",
                                  "headers": {"Authorization": "Bearer ${HONCHO_API_KEY}"}}},
        "projects": {"/work/app": {"mcpServers": {"honcho": {
            "type": "http", "url": "https://mcp.honcho.dev",
            "headers": {"Authorization": f"Bearer {CLAUDE_PROJECT_KEY}"},
        }}}},
    })
    _write_json(home / ".claude" / "settings.json", {"statusLine": {
        "type": "command", "command": f"HONCHO_API_KEY={STATUSLINE_KEY} node ~/.claude/honcho/statusline.mjs",
    }})
    openclaw = home / ".openclaw"
    _write_json(openclaw / "extensions" / "openclaw-honcho" / "package.json", {"version": "1.7.0"})
    _write_json(openclaw / "openclaw.json", {"plugins": {"entries": {"openclaw-honcho": {
        "enabled": True,
        "config": {"apiKey": OPENCLAW_KEY, "workspaceId": "openclaw", "baseUrl": "https://api.honcho.dev"},
    }}}})


def test_report_redacts_every_secret_and_flags_drift(cfg_dir, monkeypatch):
    _seed(cfg_dir)
    monkeypatch.setenv("HONCHO_API_KEY", ENV_KEY)
    monkeypatch.setenv("HONCHO_URL", "http://localhost:9999")
    monkeypatch.setenv("HONCHO_WORKSPACE_ID", "hch-v3-pasted-into-workspace-0123")

    result = CliRunner().invoke(app, ["config", "--no-network"])
    assert result.exit_code == 0, result.output
    _assert_no_secrets(result.stdout)
    assert "_keys" not in result.stdout
    report = json.loads(result.stdout)

    assert list(report) == [
        "meta", "files", "config_file", "env", "integrations", "resolved", "probes",
        "local_stack", "warnings",
    ]
    # config.json: known keys shown, everything else hidden, at any depth
    cf = report["config_file"]
    assert cf["apiKey"]["masked"] == "eyJ…" + FILE_KEY[-4:]
    assert cf["oauth"]["refreshToken"]["redacted"] is True
    assert cf["oauth"]["clientId"] == "honcho-cli"
    assert cf["hosts"]["openclaw"]["agents"]["bot"]["apiKey"] == {"set": True}
    assert cf["hosts"]["myhost"] == {"password": {"set": True}, "credentials": {"prod": {"set": True}},
                                     "pin": {"set": True}, "webhookUrl": {"set": True}}
    assert cf["hosts"]["claude_code"]["injection"]["sessionContextTokens"] == 1500
    assert cf["sessions"] == {"count": 2, "current_dir_session": None}
    # env wins for the credential, and fingerprints line up across sections
    cred = report["resolved"]["credential"]
    assert (cred["mode"], cred["source"], cred["masked"]) == (
        "api-key", "env HONCHO_API_KEY", "hch-v3-…" + ENV_KEY[-4:],
    )
    expected = "sha256:" + hashlib.sha256(ENV_KEY.encode()).hexdigest()[:8]
    assert cred["fingerprint"] == report["env"]["honcho"]["HONCHO_API_KEY"]["fingerprint"] == expected
    assert cred["shadows"] == [{"source": "config.json apiKey", "same_key": False}]
    # URLs keep scheme, host, port and path; never the userinfo
    assert report["resolved"]["base_url"] == {
        "value": "http://***@localhost:8000", "source": "config.json environmentUrl",
    }
    assert report["probes"] == {"skipped": "--no-network"}
    # integrations: harness files found, workspaces lifted, keys described
    integ = report["integrations"]["detected"]
    assert integ["codex"]["mcp"]["http_headers"]["Authorization"]["kind"] == "hch"
    assert integ["codex"]["mcp"]["http_headers"]["X-Honcho-Workspace-ID"] == "codex"
    assert integ["openclaw"]["version"] == "1.7.0"
    assert integ["openclaw"]["workspace"] == "openclaw"
    assert integ["claude_code"]["workspace"] == "agents"
    assert integ["claude_code"]["statusline"] == "~/.claude/honcho/statusline.mjs"
    assert integ["claude_code"]["mcp"]["headers"]["Authorization"] == "Bearer ${HONCHO_API_KEY}"
    assert integ["hermes"]["host_blocks"] == ["hermes"]
    # an integration's own key is described, not flagged: untested offline
    assert [(k["kind"], k["matches"], k["status"]) for k in integ["codex"]["keys"]] == [
        ("hch", None, "not checked")
    ]
    assert integ["codex"]["keys"][0]["where"] == (
        "~/.codex/config.toml mcp_servers.honcho.http_headers.Authorization"
    )
    assert [k["where"] for k in integ["claude_code"]["keys"]] == [
        '~/.claude.json projects["/work/app"].mcpServers.honcho.headers.Authorization'
    ]
    # Only an Authorization header on the hosted MCP URL is a testable key
    assert "keys" not in integ["cursor"]
    clients = report["integrations"]["mcp_clients"]
    assert "keys" not in clients["zed"]  # JSONC parsed; mcp-remote args never lifted
    assert clients["zed"]["args"][:2] == ["mcp-remote", "http://localhost:8787"]
    assert [k["status"] for k in clients["windsurf"]["keys"]] == ["malformed"]
    # local stack: provider key redacted, placeholder flagged, plain values kept,
    # every password hidden however it is spelled
    prof = report["local_stack"]["profiles"][0]
    assert (prof["state"], prof["ports"]["api"]) == ("not checked", 8001)
    assert prof["env"]["LLM_OPENAI_API_KEY"]["kind"] == "sk"
    assert prof["env"]["LLM_ANTHROPIC_API_KEY"] == {"placeholder": "your-api-key-here"}
    assert prof["env"]["SUMMARY_MAX_TOKENS_SHORT"] == "400"
    assert prof["env"]["DB_URI"] == "postgresql://***@db:5432/honcho"
    assert prof["env"]["DB_CONNECTION_URI"] == "postgresql+psycopg://***@db:5432/honcho"
    assert prof["env"]["CACHE_URL"] == "redis://***@redis:6379/0"
    assert prof["env"]["POSTGRES_PASSWORD"] == {"set": True}
    assert prof["config_toml"]["auth"]["jwt_secret"] == {"set": True}
    # a dict given as JSON is redacted inside; a nested env name is judged by its last segment
    assert prof["env"]["SUMMARY_MODEL_CONFIG"] == {
        "transport": "openai", "model": "gpt-x", "overrides": {"api_key": {"set": True}},
    }
    assert prof["env"]["DERIVER_MODEL_CONFIG__OVERRIDES__PROVIDER_PARAMS__EXTRA_QUERY__CODE"] == {"set": True}
    assert prof["env_unparsed_lines"] == 1
    # warnings carry a stable id, the integration or stack they are about, and a fix
    by_check = {(w["check"], w["subject"]): w for w in report["warnings"]}
    assert not any(w["check"] == "stack/stale-url" and w["subject"] == "myhost" for w in report["warnings"])
    assert by_check[("stack/stale-url", "hermes")]["short"] == (
        "hosts.hermes.baseUrl → localhost:8000, but no local stack uses that port"
    )
    assert by_check[("integration/auth-format", "windsurf")]["short"] == (
        "MCP Authorization header is missing 'Bearer '"
    )
    assert not any(check.startswith("integration/key") for check, _ in by_check)
    assert ("integration/hosted-mcp", "codex") in by_check
    assert ("stack/placeholder-key", "local") in by_check
    assert ("env/ignored-var", None) in by_check
    assert ("auth/oauth-expired", None) in by_check

    # The terminal views come from the same report and must not leak either.
    monkeypatch.setattr("honcho_cli.commands.config_cmd.use_json", lambda: False)
    for args in (["config", "--no-network"], ["config", "--no-network", "--all"]):
        rendered = CliRunner().invoke(app, args)
        assert rendered.exit_code == 0, rendered.output
        _assert_no_secrets(rendered.output)
        assert "own key" in rendered.output
        assert "hch-v3-…" + ENV_KEY[-4:] in rendered.output

    # --deep tests a key only against the API its own entry names, and only a
    # refused key is a warning. Network and Docker are stubbed: every key but
    # Codex's works.
    tested: list[tuple[str, str]] = []

    def fake_probe(target, key, masked):
        tested.append((target, key))
        if key == CODEX_KEY:
            return {"status": "rejected", "detail": "Invalid API key", "http": 401}
        return {"status": "accepted", "detail": "", "http": 200}

    for name, fake in {
        "_probe_key": fake_probe,
        "_probes": lambda config, mode, warnings: {"skipped": "stubbed"},
        "_server_version": lambda *a, **k: None,
        "list_honcho_containers": lambda **k: [],
        "compose_ps": lambda profile, **k: [],
        "api_healthy": lambda *a, **k: False,
    }.items():
        monkeypatch.setattr(f"honcho_cli.diagnostics.{name}", fake)
    monkeypatch.setattr("honcho_cli.commands.config_cmd.use_json", lambda: True)
    deep = CliRunner().invoke(app, ["config", "--deep"])
    assert deep.exit_code == 0, deep.output
    _assert_no_secrets(deep.stdout)
    report = json.loads(deep.stdout)
    hosted = "https://api.honcho.dev"
    assert sorted(tested) == sorted([(hosted, CODEX_KEY), (hosted, OPENCLAW_KEY), (hosted, CLAUDE_PROJECT_KEY)])
    keys = {n: e["keys"][0] for n, e in report["integrations"]["detected"].items() if e.get("keys")}
    assert (keys["codex"]["accepted"], keys["codex"]["checked_against"]) == (False, hosted)
    assert keys["openclaw"]["accepted"] is True
    rejected = [w for w in report["warnings"] if w["check"] == "integration/key-rejected"]
    assert [(w["subject"], w["short"]) for w in rejected] == [("codex", "MCP key rejected by api.honcho.dev")]
    assert rejected[0]["fix"].startswith("codex-honcho install")
    # A valid key without admin rights is refused by list-workspaces too; the
    # server's message is what separates it from a dead key.
    from honcho_cli.diagnostics import classify_key_response
    assert classify_key_response(401, "Resource requires admin privileges") == "scoped"
    assert classify_key_response(401, "Invalid API key") == "rejected"
    assert classify_key_response(401, "JWT expired") == "expired"


def test_report_survives_unparseable_config(cfg_dir):
    (cfg_dir / "config.json").write_text("{not json")

    result = CliRunner().invoke(app, ["config", "--no-network"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)

    assert "error" in report["config_file"]
    assert report["files"]["config_file"]["parse_ok"] is False
    assert report["resolved"]["credential"]["mode"] == "none"
    assert report["integrations"]["detected"] == {}
    assert report["local_stack"]["profiles"] == []
    assert {w["check"] for w in report["warnings"]} >= {"files/unparseable", "auth/no-credentials"}
