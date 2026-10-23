"""Catalog agents and local detection for `honcho install`.

Detects installed AI agents and MCP clients on the local machine and inspects
their configuration to check whether Honcho memory is already installed.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import json5
import yaml


def _safe_read_text(path: Path) -> str:
    """Read a text file safely; return empty string on any error."""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def _safe_read_json(path: Path) -> dict[str, object] | None:
    """Read and parse a JSON file safely into a dict; return None on error."""
    text = _safe_read_text(path)
    if not text:
        return None
    try:
        data: object = json5.loads(text)
        if isinstance(data, dict):
            return cast(dict[str, object], data)
    except ValueError:
        pass
    return None


def _config_value(data: object, key: str) -> object:
    return cast(dict[str, object], data).get(key) if isinstance(data, dict) else None


def _has_server(paths: list[Path], *keys: str) -> bool:
    """Check server entries, ignoring comments and unrelated settings."""
    for path in paths:
        data: object = _safe_read_json(path)
        for key in keys:
            data = _config_value(data, key)
        if isinstance(data, dict) and "honcho" in data:
            return True
    return False


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _has_package(path: Path, key: str, package: str) -> bool:
    data = _safe_read_json(path) or {}
    entries = data.get(key, [])
    if not isinstance(entries, list):
        return False
    for entry in cast(list[object], entries):
        source = (
            cast(dict[str, object], entry).get("source", "")
            if isinstance(entry, dict)
            else entry
        )
        if isinstance(source, str):
            source = source.removeprefix("npm:")
            if source == package or source.startswith(package + "@"):
                return True
    return False


def _check_claude_code_honcho() -> bool:
    home = Path.home()
    installed_file = home / ".claude" / "plugins" / "installed_plugins.json"
    installed = _safe_read_json(installed_file) or {}
    plugins = installed.get("plugins", {})
    if isinstance(plugins, dict) and any(
        name.split("@")[0] == "honcho" and bool(entries)
        for name, entries in cast(dict[str, object], plugins).items()
    ):
        return True
    claude_json = home / ".claude.json"
    claude_data = _safe_read_json(claude_json)
    if isinstance(claude_data, dict):
        mcp = claude_data.get("mcpServers", {})
        if isinstance(mcp, dict) and "honcho" in mcp:
            return True
    return False


def _check_codex_honcho() -> bool:
    home = Path.home()
    codex_dir = home / ".codex"
    if (codex_dir / "honcho").exists():
        return True
    if (codex_dir / "skills" / "honcho-memory").exists():
        return True
    try:
        cfg = tomllib.loads(_safe_read_text(codex_dir / "config.toml"))
        servers = cfg.get("mcp_servers", {})
        if isinstance(servers, dict) and "honcho" in servers:
            return True
    except tomllib.TOMLDecodeError:
        pass
    hooks_text = _safe_read_text(codex_dir / "hooks.json")
    return "honcho" in hooks_text.lower()


def _check_opencode_honcho() -> bool:
    home = Path.home()
    for p in [
        _config_home() / "opencode" / "opencode.json",
        _config_home() / "opencode" / "opencode.jsonc",
        home / ".opencode" / "opencode.json",
        _config_home() / "opencode" / "config.json",
    ]:
        if _has_package(p, "plugin", "@honcho-ai/opencode-honcho"):
            return True
    plugin_dir = _config_home() / "opencode" / "plugins"
    if plugin_dir.is_dir():
        try:
            if any("opencode-honcho" in x.name for x in plugin_dir.iterdir()):
                return True
        except OSError:
            pass
    return False


def _check_pi_honcho() -> bool:
    home = Path.home()
    for p in [
        home / ".pi" / "agent" / "settings.json",
        Path.cwd() / ".pi" / "settings.json",
        home / ".pi" / "package.json",
        home / ".pi" / "config.json",
        home / ".config" / "pi" / "config.json",
    ]:
        if _has_package(p, "packages", "@honcho-ai/pi-honcho"):
            return True
    ext_dir = home / ".pi" / "agent" / "extensions"
    if ext_dir.is_dir():
        try:
            if any("pi-honcho" in x.name for x in ext_dir.iterdir()):
                return True
        except OSError:
            pass
    return False


def _check_kilo_honcho() -> bool:
    home = Path.home()
    for p in [
        _config_home() / "kilo" / "kilo.jsonc",
        _config_home() / "kilo" / "kilo.json",
        _config_home() / "kilo" / "opencode.json",
        _config_home() / "kilo" / "tui.json",
        home / ".kilo" / "kilo.json",
    ]:
        if _has_package(p, "plugin", "@honcho-ai/kilo-honcho"):
            return True
    return False


def _check_dsh_honcho() -> bool:
    home = Path.home()
    for d in [home / ".dsh", home / ".config" / "dsh"]:
        if d.is_dir():
            try:
                for f in d.rglob("*.json"):
                    if "dsh-honcho" in _safe_read_text(f):
                        return True
            except OSError:
                pass
    return False


def _check_hermes_honcho() -> bool:
    home = Path.home()
    hermes_home = os.environ.get("HERMES_HOME")
    dirs = [home / ".hermes"]
    if hermes_home:
        dirs.insert(0, Path(hermes_home))
    for d in dirs:
        if (d / "honcho.json").is_file():
            return True
        if (d / "plugins" / "honcho").exists():
            return True
        for cfg_name in ["config.json", "config.yaml"]:
            if "honcho" in _safe_read_text(d / cfg_name).lower():
                return True
    return False


def _check_openclaw_honcho() -> bool:
    home = Path.home()
    cfg = home / ".openclaw" / "openclaw.json"
    if "openclaw-honcho" in _safe_read_text(cfg):
        return True
    return (home / ".openclaw" / "plugins" / "openclaw-honcho").exists()


def _check_claude_desktop_honcho() -> bool:
    home = Path.home()
    paths = [
        home
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json",
        home / ".config" / "Claude" / "claude_desktop_config.json",
        home / ".config" / "claude-desktop" / "claude_desktop_config.json",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
    return _has_server(paths, "mcpServers")


def _check_cursor_honcho() -> bool:
    home = Path.home()
    paths = [
        home / ".cursor" / "mcp.json",
        Path.cwd() / ".cursor" / "mcp.json",
    ]
    if _has_server(paths, "mcpServers"):
        return True
    ext_dir = home / ".cursor" / "extensions"
    if ext_dir.is_dir():
        try:
            if any("honcho" in x.name.lower() for x in ext_dir.iterdir()):
                return True
        except OSError:
            pass
    return False


def _check_windsurf_honcho() -> bool:
    home = Path.home()
    paths = [
        home / ".codeium" / "windsurf" / "mcp_config.json",
        home / "Library" / "Application Support" / "Windsurf" / "mcp_config.json",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Windsurf" / "mcp_config.json")
    return _has_server(paths, "mcpServers")


def _check_vscode_honcho() -> bool:
    home = Path.home()
    paths = [
        home / ".config" / "Code" / "User" / "settings.json",
        home / "Library" / "Application Support" / "Code" / "User" / "settings.json",
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Code" / "User" / "settings.json")
    return _has_server(paths, "mcp", "servers") or _has_server(
        [Path.cwd() / ".vscode" / "mcp.json"], "servers"
    )


def _check_cline_honcho() -> bool:
    home = Path.home()
    subpath = (
        Path("Code")
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
        / "cline_mcp_settings.json"
    )
    paths = [
        home / ".config" / subpath,
        home / "Library" / "Application Support" / subpath,
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / subpath)
    return _has_server(paths, "mcpServers")


def _check_zed_honcho() -> bool:
    home = Path.home()
    paths = [
        home / ".config" / "zed" / "settings.json",
        home / "Library" / "Application Support" / "Zed" / "settings.json",
    ]
    return _has_server(paths, "context_servers")


def _check_goose_honcho() -> bool:
    home = Path.home()
    paths = [
        home / ".config" / "goose" / "config.yaml",
        home / ".goose" / "config.yaml",
        home / "Library" / "Application Support" / "goose" / "config.yaml",
    ]
    for p in paths:
        try:
            data = yaml.safe_load(_safe_read_text(p))
        except yaml.YAMLError:
            continue
        extensions: object = (
            cast(dict[str, object], data).get("extensions", {})
            if isinstance(data, dict)
            else None
        )
        if isinstance(extensions, dict) and "honcho" in extensions:
            return True
    return False


@dataclass
class Agent:
    id: str
    name: str
    kind: str  # "plugin" | "mcp"
    aliases: list[str] = field(default_factory=list)
    binaries: list[str] = field(default_factory=list)
    config_paths: list[Path] = field(default_factory=list)
    install_command: str = ""
    guide_url: str = ""
    honcho_checker: Callable[[], bool] = lambda: False

    def is_detected(self) -> bool:
        """Check if binary is on PATH or any config directory/file exists."""
        for b in self.binaries:
            if shutil.which(b) is not None:
                return True
        return any(p.exists() for p in self.config_paths)

    def has_honcho(self) -> bool:
        """Check if Honcho is configured in this agent's config."""
        return self.honcho_checker()

    def get_command(self) -> str:
        """Return the command to execute for this agent."""
        if self.id == "codex":
            if shutil.which("codex-honcho"):
                return "codex-honcho install"
            return "npm install -g @honcho-ai/codex-honcho && codex-honcho install"
        return self.install_command

    def to_dict(self, detected: bool | None = None) -> dict[str, object]:
        det = self.is_detected() if detected is None else detected
        installed = self.has_honcho() if det else False
        return {
            "id": self.id,
            "name": self.name,
            "type": self.kind,
            "detected": det,
            "installed": installed,
            "command": self.get_command() if self.kind == "plugin" else None,
            "guide_url": self.guide_url,
        }


def _build_catalog() -> list[Agent]:
    home = Path.home()
    appdata = os.environ.get("APPDATA")
    appdata_path = Path(appdata) if appdata else None

    # Helpers for config paths
    claude_paths = [home / ".claude", home / ".claude.json"]
    codex_paths = [home / ".codex"]
    opencode_paths = [_config_home() / "opencode", home / ".opencode"]
    pi_paths = [home / ".pi", home / ".config" / "pi"]
    kilo_paths = [_config_home() / "kilo", home / ".kilo"]
    dsh_paths = [home / ".dsh", home / ".config" / "dsh"]
    hermes_paths = [home / ".hermes"]
    if os.environ.get("HERMES_HOME"):
        hermes_paths.append(Path(os.environ["HERMES_HOME"]))
    openclaw_paths = [home / ".openclaw"]

    claude_desktop_paths = [
        home / "Library" / "Application Support" / "Claude",
        home / ".config" / "Claude",
        home / ".config" / "claude-desktop",
        Path("/Applications/Claude.app"),
    ]
    if appdata_path:
        claude_desktop_paths.append(appdata_path / "Claude")

    cursor_paths = [
        home / ".cursor",
        home / "Library" / "Application Support" / "Cursor",
        Path("/Applications/Cursor.app"),
    ]

    windsurf_paths = [
        home / ".codeium" / "windsurf",
        home / "Library" / "Application Support" / "Windsurf",
        Path("/Applications/Windsurf.app"),
    ]
    if appdata_path:
        windsurf_paths.append(appdata_path / "Windsurf")

    vscode_paths = [
        home / ".vscode",
        home / ".config" / "Code",
        home / "Library" / "Application Support" / "Code",
        Path("/Applications/Visual Studio Code.app"),
    ]
    if appdata_path:
        vscode_paths.append(appdata_path / "Code")

    cline_paths = [
        home / ".cline",
        home / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev",
        home
        / "Library"
        / "Application Support"
        / "Code"
        / "User"
        / "globalStorage"
        / "saoudrizwan.claude-dev",
    ]
    if appdata_path:
        cline_paths.append(
            appdata_path / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev"
        )

    zed_paths = [
        home / ".config" / "zed",
        home / "Library" / "Application Support" / "Zed",
        Path("/Applications/Zed.app"),
    ]

    goose_paths = [
        home / ".config" / "goose",
        home / ".goose",
        home / "Library" / "Application Support" / "goose",
    ]

    return [
        # 8 Plugin agents
        Agent(
            id="claude-code",
            name="Claude Code",
            kind="plugin",
            aliases=["claude", "claude_code"],
            binaries=["claude"],
            config_paths=claude_paths,
            install_command="claude plugin marketplace add plastic-labs/claude-honcho && claude plugin install honcho@honcho",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/claude-code",
            honcho_checker=_check_claude_code_honcho,
        ),
        Agent(
            id="codex",
            name="Codex",
            kind="plugin",
            aliases=["openai-codex"],
            binaries=["codex", "codex-honcho"],
            config_paths=codex_paths,
            install_command="npm install -g @honcho-ai/codex-honcho && codex-honcho install",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/codex",
            honcho_checker=_check_codex_honcho,
        ),
        Agent(
            id="opencode",
            name="OpenCode",
            kind="plugin",
            aliases=["open-code"],
            binaries=["opencode"],
            config_paths=opencode_paths,
            install_command='opencode plugin "@honcho-ai/opencode-honcho" --global',
            guide_url="https://honcho.dev/docs/v3/guides/integrations/opencode",
            honcho_checker=_check_opencode_honcho,
        ),
        Agent(
            id="pi",
            name="Pi",
            kind="plugin",
            aliases=[],
            binaries=["pi"],
            config_paths=pi_paths,
            install_command="pi install npm:@honcho-ai/pi-honcho",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/pi",
            honcho_checker=_check_pi_honcho,
        ),
        Agent(
            id="kilo",
            name="Kilo Code",
            kind="plugin",
            aliases=["kilo-code", "kilocode"],
            binaries=["kilo"],
            config_paths=kilo_paths,
            install_command="kilo plugin @honcho-ai/kilo-honcho --global",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/kilo",
            honcho_checker=_check_kilo_honcho,
        ),
        Agent(
            id="deepseek-harness",
            name="DeepSeek Harness",
            kind="plugin",
            aliases=["dsh", "deepseek"],
            binaries=["dsh"],
            config_paths=dsh_paths,
            install_command="dsh plugin --profile <name> add @honcho-ai/dsh-honcho",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/deepseek-harness",
            honcho_checker=_check_dsh_honcho,
        ),
        Agent(
            id="hermes",
            name="Hermes Agent",
            kind="plugin",
            aliases=["hermes-agent"],
            binaries=["hermes"],
            config_paths=hermes_paths,
            install_command="hermes plugins install honcho && hermes memory setup honcho",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/hermes",
            honcho_checker=_check_hermes_honcho,
        ),
        Agent(
            id="openclaw",
            name="OpenClaw",
            kind="plugin",
            aliases=[],
            binaries=["openclaw"],
            config_paths=openclaw_paths,
            install_command="openclaw plugins install @honcho-ai/openclaw-honcho",
            guide_url="https://honcho.dev/docs/v3/guides/integrations/openclaw",
            honcho_checker=_check_openclaw_honcho,
        ),
        # 7 MCP clients
        Agent(
            id="claude-desktop",
            name="Claude Desktop",
            kind="mcp",
            aliases=[],
            binaries=["claude-desktop"],
            config_paths=claude_desktop_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#claude-desktop",
            honcho_checker=_check_claude_desktop_honcho,
        ),
        Agent(
            id="cursor",
            name="Cursor",
            kind="mcp",
            aliases=[],
            binaries=["cursor"],
            config_paths=cursor_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#cursor",
            honcho_checker=_check_cursor_honcho,
        ),
        Agent(
            id="windsurf",
            name="Windsurf",
            kind="mcp",
            aliases=["cascade"],
            binaries=["windsurf"],
            config_paths=windsurf_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#windsurf",
            honcho_checker=_check_windsurf_honcho,
        ),
        Agent(
            id="vscode",
            name="VS Code",
            kind="mcp",
            aliases=["vs-code", "code"],
            binaries=["code"],
            config_paths=vscode_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#vs-code-copilot-chat",
            honcho_checker=_check_vscode_honcho,
        ),
        Agent(
            id="cline",
            name="Cline",
            kind="mcp",
            aliases=[],
            binaries=["cline"],
            config_paths=cline_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#cline",
            honcho_checker=_check_cline_honcho,
        ),
        Agent(
            id="zed",
            name="Zed",
            kind="mcp",
            aliases=["zed-editor"],
            binaries=["zed", "zed-editor"],
            config_paths=zed_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#zed",
            honcho_checker=_check_zed_honcho,
        ),
        Agent(
            id="goose",
            name="Goose",
            kind="mcp",
            aliases=[],
            binaries=["goose"],
            config_paths=goose_paths,
            guide_url="https://honcho.dev/docs/v3/guides/integrations/mcp#goose",
            honcho_checker=_check_goose_honcho,
        ),
    ]


def get_all_agents() -> list[Agent]:
    """Return all catalog agents."""
    return _build_catalog()


def detect_agents() -> list[Agent]:
    """Return catalog agents detected on the local machine."""
    return [agent for agent in get_all_agents() if agent.is_detected()]


def find_agent(target: str) -> Agent | None:
    """Find agent by ID, alias, or lowercase name."""
    t = target.strip().lower()
    for agent in get_all_agents():
        if agent.id.lower() == t:
            return agent
        if t in [a.lower() for a in agent.aliases]:
            return agent
        if agent.name.lower() == t:
            return agent
    return None
