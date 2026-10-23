"""Tests for `honcho install`: agent detection, config checking, CLI reporting and actions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from honcho_cli.agents import (
    Agent,
    find_agent,
    get_all_agents,
)
from honcho_cli.main import app
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def isolated_agents(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ("XDG_CONFIG_HOME", "APPDATA", "HERMES_HOME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clean_config(tmp_path, monkeypatch):
    """Isolated ~/.honcho/config.json."""
    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", cfg_file)
    monkeypatch.setattr("honcho_cli.commands.install.CONFIG_FILE", cfg_file)
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)
    return cfg_file


def test_catalog_contains_expected_agents():
    agents = get_all_agents()
    # 8 plugins + 7 MCP clients = 15 total
    assert len(agents) == 15

    plugins = [a for a in agents if a.kind == "plugin"]
    mcp = [a for a in agents if a.kind == "mcp"]
    assert len(plugins) == 8
    assert len(mcp) == 7

    plugin_ids = {p.id for p in plugins}
    expected_plugins = {
        "claude-code",
        "codex",
        "opencode",
        "pi",
        "kilo",
        "deepseek-harness",
        "hermes",
        "openclaw",
    }
    assert plugin_ids == expected_plugins

    mcp_ids = {m.id for m in mcp}
    expected_mcp = {
        "claude-desktop",
        "cursor",
        "windsurf",
        "vscode",
        "cline",
        "zed",
        "goose",
    }
    assert mcp_ids == expected_mcp


def test_find_agent():
    assert find_agent("claude-code") is not None
    assert find_agent("claude") is not None  # alias
    assert find_agent("CLAUDE_CODE") is not None  # case insensitive
    assert find_agent("codex") is not None
    assert find_agent("cursor") is not None
    assert find_agent("nonexistent-agent") is None


def test_agent_detection_via_binary(monkeypatch):
    def fake_which(cmd):
        return "/usr/bin/custom-test-agent" if cmd == "claude" else None

    monkeypatch.setattr("shutil.which", fake_which)
    agent = find_agent("claude-code")
    assert agent is not None
    assert agent.is_detected() is True


def test_agent_detection_via_config_path(tmp_path, monkeypatch):
    fake_config = tmp_path / ".codex"
    fake_config.mkdir()

    agent = Agent(
        id="test-codex",
        name="Test Codex",
        kind="plugin",
        binaries=["nonexistent-bin"],
        config_paths=[fake_config],
    )
    assert agent.is_detected() is True


def test_agent_not_detected(tmp_path):
    agent = Agent(
        id="test-empty",
        name="Test Empty",
        kind="plugin",
        binaries=["nonexistent-binary-xyz"],
        config_paths=[tmp_path / "not-exists"],
    )
    assert agent.is_detected() is False


def test_honcho_installed_claude_code(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    plugins_dir = tmp_path / ".claude" / "plugins"
    plugins_dir.mkdir(parents=True)
    (plugins_dir / "installed_plugins.json").write_text(
        json.dumps({"plugins": {"honcho@honcho": [{"scope": "user"}]}})
    )
    agent = find_agent("claude-code")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_codex(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    (codex_dir / "config.toml").write_text("[mcp_servers.honcho]\ncommand = 'npx'")
    agent = find_agent("codex")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_cursor(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir(parents=True)
    (cursor_dir / "mcp.json").write_text(json.dumps({"mcpServers": {"honcho": {}}}))
    agent = find_agent("cursor")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_opencode(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    opencode_dir = tmp_path / ".config" / "opencode"
    opencode_dir.mkdir(parents=True)
    (opencode_dir / "opencode.json").write_text(
        json.dumps({"plugin": ["@honcho-ai/opencode-honcho"]})
    )
    agent = find_agent("opencode")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_pi(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir(parents=True)
    (pi_dir / "agent").mkdir()
    (pi_dir / "agent" / "settings.json").write_text(
        json.dumps({"packages": ["npm:@honcho-ai/pi-honcho"]})
    )
    agent = find_agent("pi")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_kilo(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kilo_dir = tmp_path / ".config" / "kilo"
    kilo_dir.mkdir(parents=True)
    (kilo_dir / "kilo.jsonc").write_text('{"plugin": ["@honcho-ai/kilo-honcho"]}')
    agent = find_agent("kilo")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_hermes(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir(parents=True)
    (hermes_dir / "honcho.json").write_text("{}")
    agent = find_agent("hermes")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_openclaw(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    openclaw_dir = tmp_path / ".openclaw"
    openclaw_dir.mkdir(parents=True)
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps({"plugins": {"openclaw-honcho": {}}})
    )
    agent = find_agent("openclaw")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_claude_desktop(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    desktop_dir = tmp_path / ".config" / "Claude"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "claude_desktop_config.json").write_text(
        json.dumps({"mcpServers": {"honcho": {}}})
    )
    agent = find_agent("claude-desktop")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_windsurf(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    windsurf_dir = tmp_path / ".codeium" / "windsurf"
    windsurf_dir.mkdir(parents=True)
    (windsurf_dir / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"honcho": {}}})
    )
    agent = find_agent("windsurf")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_vscode(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    vscode_dir = tmp_path / ".config" / "Code" / "User"
    vscode_dir.mkdir(parents=True)
    (vscode_dir / "settings.json").write_text(
        json.dumps({"mcp": {"servers": {"honcho": {}}}})
    )
    agent = find_agent("vscode")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_cline(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    cline_dir = (
        tmp_path
        / ".config"
        / "Code"
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
    )
    cline_dir.mkdir(parents=True)
    (cline_dir / "cline_mcp_settings.json").write_text(
        json.dumps({"mcpServers": {"honcho": {}}})
    )
    agent = find_agent("cline")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_zed(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    zed_dir = tmp_path / ".config" / "zed"
    zed_dir.mkdir(parents=True)
    (zed_dir / "settings.json").write_text(
        json.dumps({"context_servers": {"honcho": {}}})
    )
    agent = find_agent("zed")
    assert agent is not None
    assert agent.has_honcho() is True


def test_honcho_installed_goose(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    goose_dir = tmp_path / ".config" / "goose"
    goose_dir.mkdir(parents=True)
    (goose_dir / "config.yaml").write_text("extensions:\n  honcho:\n    enabled: true")
    agent = find_agent("goose")
    assert agent is not None
    assert agent.has_honcho() is True


# --- CLI command tests ---


def test_install_list_flag(runner, monkeypatch):
    dummy = Agent(
        id="dummy-agent",
        name="Dummy Agent",
        kind="plugin",
        install_command="echo dummy",
    )
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [dummy])

    res = runner.invoke(app, ["install", "--list"])
    assert res.exit_code == 0
    assert "Dummy Agent" in res.output
    assert "echo dummy" in res.output


def test_install_json_flag(runner, monkeypatch):
    dummy = Agent(
        id="dummy-agent",
        name="Dummy Agent",
        kind="plugin",
        install_command="echo dummy",
        guide_url="https://honcho.dev/guide",
    )
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [dummy])

    res = runner.invoke(app, ["install", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "dummy-agent"
    assert data[0]["name"] == "Dummy Agent"
    assert data[0]["command"] == "echo dummy"
    assert data[0]["detected"] is True
    assert data[0]["installed"] is False


def test_install_unknown_target(runner):
    res = runner.invoke(app, ["install", "nonexistent-id-12345"])
    assert res.exit_code == 1
    assert "Unknown agent 'nonexistent-id-12345'" in res.output


def test_install_single_plugin_success(runner, clean_config, monkeypatch):
    # Set api key so it doesn't prompt
    clean_config.write_text(json.dumps({"apiKey": "hch-test-123"}))

    mock_run = MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_run)

    res = runner.invoke(app, ["install", "opencode"])
    assert res.exit_code == 0
    assert "Installing Honcho for OpenCode" in res.output
    assert "Successfully installed Honcho for OpenCode" in res.output


def test_install_single_mcp_client(runner, monkeypatch):
    mock_open = MagicMock()
    monkeypatch.setattr("webbrowser.open", mock_open)

    res = runner.invoke(app, ["install", "cursor"])
    assert res.exit_code == 0
    assert "Opening Cursor MCP guide" in res.output
    mock_open.assert_called_once_with(
        "https://honcho.dev/docs/v3/guides/integrations/mcp#cursor"
    )


def test_install_all_skips_already_installed(runner, clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"apiKey": "hch-test-123"}))

    agent1 = Agent(
        id="ag1",
        name="Agent 1",
        kind="plugin",
        install_command="echo 1",
        honcho_checker=lambda: True,
    )
    agent2 = Agent(
        id="ag2",
        name="Agent 2",
        kind="plugin",
        install_command="echo 2",
        honcho_checker=lambda: False,
    )

    monkeypatch.setattr(
        "honcho_cli.commands.install.detect_agents", lambda: [agent1, agent2]
    )
    mock_run = MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_run)

    res = runner.invoke(app, ["install", "all"])
    assert res.exit_code == 0
    assert "Agent 2" in res.output
    assert "Agent 1" not in res.output


def test_install_all_when_all_installed(runner, monkeypatch):
    agent1 = Agent(id="ag1", name="Agent 1", kind="plugin", honcho_checker=lambda: True)
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [agent1])

    res = runner.invoke(app, ["install", "all"])
    assert res.exit_code == 0
    assert "All detected agents already have Honcho installed." in res.output


def test_install_all_when_none_detected(runner, monkeypatch):
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [])

    res = runner.invoke(app, ["install", "all"])
    assert res.exit_code == 0
    assert "No supported AI agents detected on this machine." in res.output


def test_install_prompts_for_api_key_when_missing(runner, clean_config, monkeypatch):
    mock_run = MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_run)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    # Provide the API key via input prompt
    res = runner.invoke(app, ["install", "opencode"], input="hch-new-api-key\n")
    assert res.exit_code == 0
    assert "Enter your Honcho API key" in res.output
    assert "Saved API key to" in res.output

    # Verify key was written to config.json
    saved = json.loads(clean_config.read_text())
    assert saved["apiKey"] == "hch-new-api-key"


def test_install_multiple_targets(runner, clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"apiKey": "hch-test-123"}))

    mock_run = MagicMock(returncode=0)
    mock_open = MagicMock()
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_run)
    monkeypatch.setattr("webbrowser.open", mock_open)

    res = runner.invoke(app, ["install", "opencode", "cursor"])
    assert res.exit_code == 0
    assert "Installing Honcho for OpenCode" in res.output
    assert "Opening Cursor MCP guide" in res.output
    mock_open.assert_called_once()


def test_install_no_args_non_interactive(runner):
    # Under CliRunner, sys.stdin.isatty is False
    res = runner.invoke(app, ["install"])
    assert res.exit_code == 1
    assert "No targets specified" in res.output


def test_install_no_args_none_detected(runner, monkeypatch):
    monkeypatch.setattr("honcho_cli.commands.install._is_interactive", lambda: True)
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [])

    res = runner.invoke(app, ["install"])
    assert res.exit_code == 0
    assert "No supported AI agents detected on this machine." in res.output


def test_install_interactive_picker_selection(runner, clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"apiKey": "hch-test-123"}))
    agent = Agent(
        id="test-ag", name="Test Agent", kind="plugin", install_command="echo test"
    )

    monkeypatch.setattr("honcho_cli.commands.install._is_interactive", lambda: True)
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [agent])
    monkeypatch.setattr(
        "honcho_cli.commands.install.prompt_checkbox_picker", lambda agents: [agent]
    )
    mock_run = MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_run)

    res = runner.invoke(app, ["install"])
    assert res.exit_code == 0
    assert "Installing Honcho for Test Agent" in res.output


def test_fallback_picker():
    from honcho_cli.picker import _fallback_picker

    agent1 = Agent(id="a1", name="Agent 1", kind="plugin")
    agent2 = Agent(id="a2", name="Agent 2", kind="mcp")

    with patch("sys.stdin.readline", return_value="1, 2\n"):
        res = _fallback_picker([agent1, agent2])
        assert res == [agent1, agent2]

    with patch("sys.stdin.readline", return_value="1\n"):
        res = _fallback_picker([agent1, agent2])
        assert res == [agent1]

    with patch("sys.stdin.readline", return_value="all\n"):
        res = _fallback_picker([agent1, agent2])
        assert res == [agent1, agent2]

    with patch("sys.stdin.readline", return_value="\n"):
        res = _fallback_picker([agent1, agent2])
        assert res == []


def test_raw_checkbox_picker_keys():
    from honcho_cli.picker import _raw_checkbox_picker

    agent1 = Agent(id="a1", name="Agent 1", kind="plugin")
    agent2 = Agent(id="a2", name="Agent 2", kind="mcp")

    # Sequence: Space (select 1st), Down (cursor to 2nd), Space (select 2nd), Enter (submit)
    key_seq = ["space", "down", "space", "enter"]
    key_iter = iter(key_seq)

    with (
        patch("honcho_cli.picker._read_key", side_effect=lambda: next(key_iter)),
        patch("sys.stdout.write"),
        patch("sys.stdout.flush"),
    ):
        chosen = _raw_checkbox_picker([agent1, agent2])
        assert chosen == [agent1, agent2]


def test_environment_key_is_persisted(runner, clean_config, monkeypatch):
    monkeypatch.setenv("HONCHO_API_KEY", "hch-environment-key")
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(app, ["install", "codex"])
    assert result.exit_code == 0
    assert json.loads(clean_config.read_text())["apiKey"] == "hch-environment-key"
    run.assert_called_once()


def test_missing_key_cancels_install(runner, clean_config, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(app, ["install", "codex"], input="\n")
    assert result.exit_code == 1
    run.assert_not_called()
    assert not clean_config.exists()


def test_plugin_failure_returns_nonzero_and_continues(
    runner, clean_config, monkeypatch
):
    clean_config.write_text(json.dumps({"apiKey": "hch-test"}))
    run = MagicMock(side_effect=[MagicMock(returncode=42), MagicMock(returncode=0)])
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(app, ["install", "pi", "opencode"])
    assert result.exit_code == 1
    assert "exit code 42" in result.output
    assert "Successfully installed Honcho for OpenCode" in result.output
    assert run.call_count == 2


def test_browser_failure_returns_nonzero(runner, monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: False)
    result = runner.invoke(app, ["install", "cursor"])
    assert result.exit_code == 1
    assert "Could not open a browser" in result.output


def test_deepseek_requires_profile_before_any_install(
    runner, clean_config, monkeypatch
):
    run = MagicMock()
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(app, ["install", "pi", "deepseek-harness"])
    assert result.exit_code == 1
    assert "--profile" in result.output
    run.assert_not_called()
    assert not clean_config.exists()


def test_deepseek_uses_profile(runner, clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"apiKey": "hch-test"}))
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(
        app, ["install", "dsh", "--profile", "my profile; echo nope"]
    )
    assert result.exit_code == 0
    assert run.call_args.args[0] == [
        "dsh",
        "plugin",
        "--profile",
        "my profile; echo nope",
        "add",
        "@honcho-ai/dsh-honcho",
    ]


def test_marketplace_is_not_an_installation(tmp_path):
    plugins = tmp_path / ".claude" / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "known_marketplaces.json").write_text(
        json.dumps({"plastic-labs/claude-honcho": {}})
    )
    agent = find_agent("claude-code")
    assert agent is not None
    assert not agent.has_honcho()


@pytest.mark.parametrize(
    ("agent_id", "path", "contents"),
    [
        ("vscode", ".config/Code/User/settings.json", '{"window.title": "honcho"}'),
        ("codex", ".codex/config.toml", '# honcho setup pending\nmodel = "test"'),
        (
            "goose",
            ".config/goose/config.yaml",
            "# honcho setup pending\nextensions: {}",
        ),
        ("cursor", ".cursor/mcp.json", '// honcho setup pending\n{"mcpServers": {},}'),
    ],
)
def test_unrelated_settings_are_not_installations(tmp_path, agent_id, path, contents):
    config = tmp_path / path
    config.parent.mkdir(parents=True)
    config.write_text(contents)
    agent = find_agent(agent_id)
    assert agent is not None
    assert not agent.has_honcho()


def test_jsonc_mcp_settings(tmp_path):
    config = tmp_path / ".config/Code/User/settings.json"
    config.parent.mkdir(parents=True)
    config.write_text('// MCP servers\n{"mcp": {"servers": {"honcho": {},},},}')
    agent = find_agent("vscode")
    assert agent is not None
    assert agent.has_honcho()


def test_xdg_agent_detection_and_status(tmp_path, monkeypatch):
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    config = config_home / "kilo" / "opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"plugin": ["@honcho-ai/kilo-honcho"]}')
    agent = find_agent("kilo")
    assert agent is not None
    assert agent.is_detected()
    assert agent.has_honcho()


@pytest.mark.parametrize("flag", ["--list", "--json"])
def test_reporting_does_not_write_or_install(runner, clean_config, monkeypatch, flag):
    monkeypatch.setenv("HONCHO_API_KEY", "hch-test")
    run = MagicMock()
    save = MagicMock()
    open_browser = MagicMock()
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("webbrowser.open", open_browser)
    monkeypatch.setattr("honcho_cli.config.CLIConfig.save", save)
    result = runner.invoke(app, ["install", flag, "--all"])
    assert result.exit_code == 0
    run.assert_not_called()
    save.assert_not_called()
    open_browser.assert_not_called()
    assert not clean_config.exists()


def test_picker_displays_selected_checkbox():
    from honcho_cli.picker import _raw_checkbox_picker

    agent = Agent(id="test", name="Test", kind="plugin")
    with (
        patch("honcho_cli.picker.Live") as live,
        patch("honcho_cli.picker._read_key", side_effect=["space", "enter"]),
    ):
        assert _raw_checkbox_picker([agent]) == [agent]
    rendered = live.return_value.__enter__.return_value.update.call_args.args[0]
    assert "[x] Test" in rendered.plain


def test_all_installs_explicit_deepseek_profile(runner, clean_config, monkeypatch):
    clean_config.write_text(json.dumps({"apiKey": "hch-test"}))
    agent = Agent(
        id="deepseek-harness",
        name="DeepSeek Harness",
        kind="plugin",
        honcho_checker=lambda: True,
    )
    monkeypatch.setattr("honcho_cli.commands.install.detect_agents", lambda: [agent])
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    result = runner.invoke(app, ["install", "all", "--profile", "headless"])
    assert result.exit_code == 0
    assert run.call_args.args[0][3] == "headless"
