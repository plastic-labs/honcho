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


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A fake home with Codex and Pi on PATH, Pi already wired, and a Cursor config dir."""
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("codex", "pi", "npx"):
        exe = bin_dir / name
        exe.write_text("#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)

    pi_settings = home / ".pi" / "agent" / "settings.json"
    pi_settings.parent.mkdir(parents=True)
    pi_settings.write_text(json.dumps({"packages": ["npm:@honcho-ai/pi-honcho"]}))
    (home / ".cursor").mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HONCHO_API_KEY", "hch-test")
    for var in ("XDG_CONFIG_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "OPENCODE_CONFIG_DIR", "HERMES_HOME",
                "OPENCLAW_STATE_DIR", "DSH_HOME", "APPDATA"):
        monkeypatch.delenv(var, raising=False)
    return home


def test_detects_agents_and_installs_only_what_is_missing(machine, monkeypatch):
    runner = CliRunner()

    rows = {r["id"]: r for r in json.loads(runner.invoke(app, ["install", "--json"]).output)}
    assert rows["pi"]["found"] and rows["pi"]["installed"] is True
    assert rows["codex"]["found"] and rows["codex"]["installed"] is False
    assert rows["codex"]["command"] == "npx -y @honcho-ai/codex-honcho install"
    assert rows["cursor"]["found"] and rows["cursor"]["kind"] == "mcp"
    assert not rows["claude-code"]["found"]

    ran: list[list[str]] = []
    monkeypatch.setattr("honcho_cli.commands.install.runner", lambda argv: ran.append(list(argv)) or 0)
    result = runner.invoke(app, ["install", "all"])

    # Pi is already wired and Cursor is an MCP client, so only Codex runs.
    assert result.exit_code == 0, result.output
    assert ran == [["npx", "-y", "@honcho-ai/codex-honcho", "install"]]


@pytest.mark.skipif(not CATALOG.exists(), reason="docs catalog not in this checkout")
def test_every_catalog_agent_is_detected():
    """Adding an agent to the docs catalog should prompt adding detection here."""
    sections = CATALOG.read_text().split('title: "')[1:]
    names = {
        name
        for section in sections
        if section.startswith(("General agents", "Coding agents"))
        for name in re.findall(r'name: "([^"]+)"', section)
    }
    assert names == {a.name for a in AGENTS}
