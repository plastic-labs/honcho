"""`honcho install`: agent detection, and which install commands run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from honcho_cli.agents import AGENTS
from honcho_cli.main import app
from typer.testing import CliRunner

CATALOG = Path(__file__).resolve().parents[2] / "docs" / "snippets" / "integrations-catalog.jsx"
KEY = "hch-test-0123456789abcdef"

# Fake agent CLIs. Claude Code already lists the Honcho plugin; Codex reports
# the MCP server only once something has registered it.
STUBS = {
    "claude": 'case "$*" in "plugin list --json") echo \'[{"id":"honcho@honcho"}]\';; esac',
    "codex": '[ "$*" = "mcp get honcho" ] && { [ -f "$HOME/codex-has-honcho" ] || exit 1; }; exit 0',
    "node": "exit 0",
    "npx": "exit 0",
}


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A fake home with Claude Code (Honcho installed), Codex (not yet) and a Cursor config dir."""
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in STUBS.items():
        exe = bin_dir / name
        exe.write_text(f"#!/bin/sh\n{body}\n")
        exe.chmod(0o755)
    (home / ".cursor").mkdir(parents=True)
    config = home / ".honcho" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"apiKey": KEY}))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", config)
    monkeypatch.setattr("honcho_cli.commands.install.CONFIG_FILE", config)
    for var in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR",
                "OPENCODE_CONFIG_DIR", "HERMES_HOME", "OPENCLAW_STATE_DIR", "APPDATA", "AI_AGENT",
                *(v for a in AGENTS for v in a.caller_env)):
        monkeypatch.delenv(var, raising=False)
    return home


def test_detects_agents_and_installs_only_what_is_missing(machine, monkeypatch):
    runner = CliRunner()

    rows = {r["id"]: r for r in json.loads(runner.invoke(app, ["install", "--json"]).output)}
    assert rows["claude-code"]["found"] and rows["claude-code"]["installed"] is True
    assert rows["codex"]["found"] and rows["codex"]["installed"] is False
    assert rows["codex"]["command"].startswith("codex mcp add honcho -- npx -y mcp-remote https://mcp.honcho.dev")
    assert KEY not in rows["codex"]["command"]  # shown masked
    assert rows["cursor"]["found"] and rows["cursor"]["kind"] == "suggest"
    assert not rows["hermes"]["found"] and not rows["openclaw"]["found"]

    ran: list[tuple[list[str], bool]] = []

    def fake_run(argv, quiet):
        ran.append((list(argv), quiet))
        if argv[:3] == ["codex", "mcp", "add"]:
            (machine / "codex-has-honcho").touch()
        return 0

    monkeypatch.setattr("honcho_cli.commands.install.runner", fake_run)
    result = runner.invoke(app, ["install", "all"])

    # Claude Code already has Honcho and Cursor is set up from its guide, so only Codex runs:
    # its own `mcp add` on the terminal, then the skill install with its output hidden.
    assert result.exit_code == 0, result.output
    assert [(argv[:3], quiet) for argv, quiet in ran] == [
        (["codex", "mcp", "add"], False),
        (["npx", "-y", "skills"], True),
    ]
    assert f"Authorization:Bearer {KEY}" in ran[0][0]
    # A shared user peer is saved next to the key, keeping what was already there.
    saved = json.loads((machine / ".honcho" / "config.json").read_text())
    assert saved["apiKey"] == KEY and saved["peerName"]


@pytest.mark.skipif(not CATALOG.exists(), reason="docs catalog not in this checkout")
def test_every_catalog_agent_is_detected():
    """Adding an agent to the docs catalog should prompt adding it here."""
    sections = CATALOG.read_text().split('title: "')[1:]
    names = {
        name
        for section in sections
        if section.startswith(("General agents", "Coding agents"))
        for name in re.findall(r'name: "([^"]+)"', section)
    }
    assert names <= {a.name for a in AGENTS}
