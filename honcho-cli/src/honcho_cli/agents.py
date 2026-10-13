"""Which AI agents are on this machine, and does each one already have Honcho?

Mirrors the "General agents" and "Coding agents" sections of the docs
integrations catalog (``docs/snippets/integrations-catalog.jsx``). Detection is
read-only: binaries on PATH, directories and small config files on disk. The
only subprocess is ``opencode --version``, to pick the right install syntax.

Two kinds of agent:

- ``plugin``: Honcho ships a plugin with a one-line install command.
- ``mcp``: the client talks to the Honcho MCP server; setup is a config edit
  documented in the MCP guide, so we link to it instead of editing the file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DOCS = "https://honcho.dev/docs/v3/guides"
CATALOG_URL = f"{DOCS}/overview"
MCP_DOCS = f"{DOCS}/integrations/mcp"


# --------------------------------------------------------------------------- #
# small readers — every failure means "can't tell", never an exception


def _home() -> Path:
    return Path.home()


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default


def _exists(path: Path) -> bool:
    """``Path.exists`` that treats an unreadable parent as missing (3.11/3.12 raise)."""
    try:
        return path.exists()
    except OSError:
        return False


def _either(*results: bool | None) -> bool | None:
    """Combine installed checks: True if any is True, None if all can't tell."""
    if any(r is True for r in results):
        return True
    return None if all(r is None for r in results) else False


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


_JSONC_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
_JSONC_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _read_json(path: Path) -> object | None:
    """Parse JSON or JSONC (comments and trailing commas, as VS Code and Zed write)."""
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    stripped = _JSONC_COMMENT.sub(lambda m: m.group(0) if m.group(0).startswith('"') else "", text)
    try:
        return json.loads(_JSONC_TRAILING_COMMA.sub(r"\1", stripped))
    except ValueError:
        return None


def _get(obj: object, *keys: str) -> object | None:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _mentions_honcho(value: object) -> bool:
    """True if a config fragment names Honcho: a ``honcho`` key or the hosted MCP URL."""
    if isinstance(value, dict):
        return any("honcho" in str(k).lower() or _mentions_honcho(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_mentions_honcho(v) for v in value)
    return isinstance(value, str) and ("mcp.honcho.dev" in value or "honcho-ai" in value)


def _any_list_has(files: list[Path], key: str, package: str) -> bool | None:
    """Is ``package`` in the ``key`` list of any of these config files?

    Entries may be bare names, names with ``@version``, ``npm:`` prefixes, or
    objects with a ``source``/``name`` field. Missing files mean "not added";
    ``None`` only when a file exists but can't be parsed.
    """
    unreadable = False
    for f in files:
        if not _exists(f):
            continue
        data = _read_json(f)
        if data is None:
            unreadable = True
            continue
        entries = _get(data, key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                entry = entry.get("source") or entry.get("name") or ""
            if isinstance(entry, str) and package in entry:
                return True
    return None if unreadable else False


def _mcp_file_has_honcho(files: list[Path], *section: str) -> bool | None:
    """Look for a Honcho server under ``section`` in the first file that exists."""
    for f in files:
        if not _exists(f):
            continue
        data = _read_json(f)
        return None if data is None else _mentions_honcho(_get(data, *section))
    return False


# --------------------------------------------------------------------------- #
# per-platform locations


def _app_support() -> Path:
    """Per-user application data dir: macOS Application Support, Windows APPDATA, Linux XDG."""
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support"
    if sys.platform == "win32":
        return _env_path("APPDATA", _home() / "AppData" / "Roaming")
    return _env_path("XDG_CONFIG_HOME", _home() / ".config")


def _xdg_config() -> Path:
    return _env_path("XDG_CONFIG_HOME", _home() / ".config")


def _vscode_user() -> Path:
    return _app_support() / "Code" / "User"


def _claude_home() -> Path:
    return _env_path("CLAUDE_CONFIG_DIR", _home() / ".claude")


def _zed_dir() -> Path:
    return _app_support() / "Zed" if sys.platform == "win32" else _xdg_config() / "zed"


def _goose_dir() -> Path:
    if sys.platform == "win32":
        return _app_support() / "Block" / "goose" / "config"
    return _xdg_config() / "goose"


def _codex_home() -> Path:
    return _env_path("CODEX_HOME", _home() / ".codex")


def _hermes_home() -> Path:
    return _env_path("HERMES_HOME", _home() / ".hermes")


def _openclaw_home() -> Path:
    return _env_path("OPENCLAW_STATE_DIR", _home() / ".openclaw")


def _opencode_dir() -> Path:
    return _env_path("OPENCODE_CONFIG_DIR", _xdg_config() / "opencode")


# --------------------------------------------------------------------------- #
# installed checks


def _claude_code_installed() -> bool | None:
    settings = _claude_home() / "settings.json"
    if not _exists(settings):
        return False
    data = _read_json(settings)
    if data is None:
        return None
    enabled = _get(data, "enabledPlugins")
    return isinstance(enabled, dict) and any(k.startswith("honcho@") and v for k, v in enabled.items())


def _codex_installed() -> bool | None:
    hooks = _read_text(_codex_home() / "hooks.json") or ""
    config = _read_text(_codex_home() / "config.toml") or ""
    return "codex-honcho" in hooks or "# >>> codex-honcho" in config


def _opencode_files() -> list[Path]:
    d = _opencode_dir()
    return [d / "opencode.json", d / "opencode.jsonc", d / "config.json"]


def _opencode_installed() -> bool | None:
    files = _opencode_files()
    plugin = _any_list_has(files, "plugin", "@honcho-ai/opencode-honcho")
    plugins = _any_list_has(files, "plugins", "@honcho-ai/opencode-honcho")
    if plugin or plugins:
        return True
    return False if plugin is not None else None


def _pi_installed() -> bool | None:
    return _any_list_has([_home() / ".pi" / "agent" / "settings.json"], "packages", "@honcho-ai/pi-honcho")


def _kilo_installed() -> bool | None:
    # Server-side config only: tui.json holds just the /honcho commands, and
    # Kilo doesn't load the memory plugin from it.
    d = _xdg_config() / "kilo"
    files = [d / "kilo.jsonc", d / "kilo.json", d / "opencode.json"]
    return _any_list_has(files, "plugin", "@honcho-ai/kilo-honcho")


# `hermes memory setup` appends `provider:` to the memory block, sometimes after
# the next section's column-0 comment banner, so comments don't end the block.
_HERMES_PROVIDER = re.compile(
    r"^memory:[^\n]*\n(?:[ \t]+[^\n]*\n|[ \t]*#[^\n]*\n|\s*\n)*?[ \t]+provider:\s*['\"]?honcho\b",
    re.M,
)


def _hermes_install() -> list[list[str]]:
    setup = ["hermes", "memory", "setup", "honcho"]
    if _exists(_hermes_home() / "plugins" / "honcho"):
        # plugin present but not the active provider; reinstalling would fail
        return [setup]
    return [["hermes", "plugins", "install", "honcho"], setup]


def _hermes_installed() -> bool | None:
    text = _read_text(_hermes_home() / "config.yaml")
    if text is None:
        return False
    return bool(_HERMES_PROVIDER.search(text + "\n"))


def _openclaw_entry() -> object | None:
    return _get(_read_json(_openclaw_home() / "openclaw.json"), "plugins", "entries", "openclaw-honcho")


def _openclaw_installed() -> bool | None:
    config = _openclaw_home() / "openclaw.json"
    if not _exists(config):
        return False
    data = _read_json(config)
    if data is None:
        return None
    entry = _get(data, "plugins", "entries", "openclaw-honcho")
    # `openclaw plugins install` alone writes {"enabled": true}; only
    # `openclaw honcho setup` adds the config block with the key.
    return isinstance(entry, dict) and isinstance(entry.get("config"), dict)


def _openclaw_install() -> list[list[str]]:
    setup = ["openclaw", "honcho", "setup"]
    if isinstance(_openclaw_entry(), dict):
        # plugin already added (setup was interrupted); reinstalling would fail
        return [setup]
    return [["openclaw", "plugins", "install", "@honcho-ai/openclaw-honcho"], setup]


DSH_PLUGIN = "@honcho-ai/dsh-honcho"


def _dsh_home() -> Path:
    return _env_path("DSH_HOME", _home() / ".dsh")


def dsh_profiles() -> dict[str, bool | None]:
    """dsh profile name -> Honcho in its bundle list (None = unreadable)."""
    out: dict[str, bool | None] = {}
    try:
        manifests = sorted((_dsh_home() / "profiles").glob("*/package.json"))
    except OSError:
        return out
    for m in manifests:
        data = _read_json(m)
        if data is None:
            out[m.parent.name] = None
            continue
        bundles = _get(data, "dsh", "profile", "bundles")
        out[m.parent.name] = isinstance(bundles, list) and DSH_PLUGIN in bundles
    return out


def _dsh_installed() -> bool | None:
    """Honcho is in some profile's ``dsh.profile.bundles`` (a disabled plugin stays in dependencies)."""
    profiles = dsh_profiles()
    return _either(*profiles.values()) if profiles else False


def _dsh_present() -> bool:
    """``dsh`` is also Debian's distributed shell; only count DeepSeek's npm one."""
    exe = shutil.which("dsh")
    if not exe:
        return False
    try:
        return "deepseek" in os.path.realpath(exe).lower()
    except OSError:
        return False


def _cline_settings() -> Path:
    return _vscode_user() / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"


def _claude_desktop_dir() -> Path:
    return _app_support() / "Claude"


# --------------------------------------------------------------------------- #
# install commands


def _opencode_major() -> int | None:
    exe = shutil.which("opencode")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.\d+", out)
    return int(m.group(1)) if m else None


def _opencode_install() -> list[list[str]]:
    if _opencode_major() == 1:
        return [["opencode", "plugin", "@honcho-ai/opencode-honcho", "--global"]]
    return [["opencode", "plugin", "add", "@honcho-ai/opencode-honcho"]]


def _claude_code_install() -> list[list[str]]:
    known = _read_json(_claude_home() / "plugins" / "known_marketplaces.json")
    cmds = []
    if not (isinstance(known, dict) and "honcho" in known):
        cmds.append(["claude", "plugin", "marketplace", "add", "plastic-labs/claude-honcho"])
    cmds.append(["claude", "plugin", "install", "honcho@honcho"])
    return cmds


# --------------------------------------------------------------------------- #
# registry


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    kind: Literal["plugin", "mcp"]
    docs_url: str
    bins: tuple[str, ...] = ()
    paths: Callable[[], list[Path]] = field(default=lambda: [])
    installed: Callable[[], bool | None] = field(default=lambda: None)
    install: Callable[[], list[list[str]]] = field(default=lambda: [])
    requires: tuple[str, ...] = ()
    install_interactive: bool = False
    present: Callable[[], bool] | None = None  # overrides bins/paths when set
    activate: str | None = None  # extra step after install, beyond restarting the agent


AGENTS: tuple[Agent, ...] = (
    # ---- coding agents with a Honcho plugin
    Agent(
        id="claude-code", name="Claude Code", kind="plugin",
        docs_url=f"{DOCS}/integrations/claude-code",
        bins=("claude",), paths=lambda: [_claude_home()],
        installed=_claude_code_installed, install=_claude_code_install,
        requires=("claude", "bun"),
    ),
    Agent(
        id="codex", name="Codex", kind="plugin",
        docs_url=f"{DOCS}/integrations/codex",
        bins=("codex",), paths=lambda: [_codex_home()],
        installed=_codex_installed,
        install=lambda: [["npx", "-y", "@honcho-ai/codex-honcho", "install"]],
        requires=("npx",),
    ),
    Agent(
        id="opencode", name="OpenCode", kind="plugin",
        docs_url=f"{DOCS}/integrations/opencode",
        bins=("opencode",), paths=lambda: [_opencode_dir()],
        installed=_opencode_installed, install=_opencode_install,
        requires=("opencode",),
    ),
    Agent(
        id="pi", name="Pi", kind="plugin",
        docs_url=f"{DOCS}/community/pi-honcho-memory",
        bins=("pi",), paths=lambda: [_home() / ".pi" / "agent"],
        installed=_pi_installed,
        install=lambda: [["pi", "install", "npm:@honcho-ai/pi-honcho"]],
        requires=("pi",),
    ),
    Agent(
        id="kilo", name="Kilo Code", kind="plugin",
        docs_url=f"{DOCS}/integrations/kilo",
        bins=("kilo",), paths=lambda: [_xdg_config() / "kilo"],
        installed=_kilo_installed,
        install=lambda: [["kilo", "plugin", "@honcho-ai/kilo-honcho", "--global"]],
        requires=("kilo",),
    ),
    Agent(
        id="dsh", name="DeepSeek Harness", kind="plugin",
        docs_url=f"{DOCS}/integrations/deepseek-harness",
        present=_dsh_present,
        installed=_dsh_installed,
        install=lambda: [["dsh", "plugin", "--profile", "<profile>", "add", DSH_PLUGIN]],
        requires=("dsh", "pnpm"),  # `dsh plugin` hands off to pnpm
    ),
    # ---- general agents with a Honcho plugin
    Agent(
        id="hermes", name="Hermes Agent", kind="plugin",
        docs_url=f"{DOCS}/integrations/hermes",
        bins=("hermes",), paths=lambda: [_hermes_home()],
        installed=_hermes_installed, install=_hermes_install,
        requires=("hermes",), install_interactive=True,
    ),
    Agent(
        id="openclaw", name="OpenClaw", kind="plugin",
        docs_url=f"{DOCS}/integrations/openclaw",
        bins=("openclaw",), paths=lambda: [_openclaw_home()],
        installed=_openclaw_installed, install=_openclaw_install,
        requires=("openclaw",), install_interactive=True,
        activate="openclaw gateway --force",
    ),
    # ---- MCP clients
    Agent(
        id="cursor", name="Cursor", kind="mcp", docs_url=f"{MCP_DOCS}#cursor",
        bins=("cursor", "cursor-agent"), paths=lambda: [_home() / ".cursor"],
        # the MCP server, or the older cursor-honcho hooks plugin
        installed=lambda: _either(
            _mcp_file_has_honcho([_home() / ".cursor" / "mcp.json"], "mcpServers"),
            "honcho" in (_read_text(_home() / ".cursor" / "hooks.json") or ""),
        ),
    ),
    Agent(
        id="claude-desktop", name="Claude Desktop", kind="mcp", docs_url=f"{MCP_DOCS}#claude-desktop",
        paths=lambda: [_claude_desktop_dir()] if sys.platform in ("darwin", "win32") else [],
        installed=lambda: _mcp_file_has_honcho(
            [_claude_desktop_dir() / "claude_desktop_config.json"], "mcpServers"),
    ),
    Agent(
        id="windsurf", name="Windsurf", kind="mcp", docs_url=f"{MCP_DOCS}#windsurf",
        bins=("windsurf",), paths=lambda: [_home() / ".codeium" / "windsurf"],
        installed=lambda: _mcp_file_has_honcho(
            [_home() / ".codeium" / "windsurf" / "mcp_config.json"], "mcpServers"),
    ),
    Agent(
        id="cline", name="Cline", kind="mcp", docs_url=f"{MCP_DOCS}#cline",
        paths=lambda: [_cline_settings().parent.parent],
        installed=lambda: _mcp_file_has_honcho([_cline_settings()], "mcpServers"),
    ),
    Agent(
        id="vscode", name="VS Code", kind="mcp", docs_url=f"{MCP_DOCS}#vs-code-copilot-chat",
        bins=("code",), paths=lambda: [_vscode_user()],
        # user-level mcp.json, or the older "mcp" block in settings.json
        installed=lambda: _either(
            _mcp_file_has_honcho([_vscode_user() / "mcp.json"], "servers"),
            _mcp_file_has_honcho([_vscode_user() / "settings.json"], "mcp", "servers"),
        ),
    ),
    Agent(
        id="zed", name="Zed", kind="mcp", docs_url=f"{MCP_DOCS}#zed",
        bins=("zed", "zeditor"), paths=lambda: [_zed_dir()],
        installed=lambda: _mcp_file_has_honcho([_zed_dir() / "settings.json"], "context_servers"),
    ),
    Agent(
        id="goose", name="Goose", kind="mcp", docs_url=f"{MCP_DOCS}#goose",
        # config dir only: "goose" is also the binary of a common Go migration tool
        paths=lambda: [_goose_dir()],
        installed=lambda: "mcp.honcho.dev" in (_read_text(_goose_dir() / "config.yaml") or ""),
    ),
)


@dataclass
class Detection:
    agent: Agent
    found: bool
    installed: bool | None = None  # None = can't tell
    blocked: str | None = None     # why the install can't run here
    commands: list[list[str]] = field(default_factory=list)


def detect(agent: Agent) -> Detection:
    if agent.present is not None:
        found = agent.present()
    else:
        found = any(shutil.which(b) for b in agent.bins) or any(_exists(p) for p in agent.paths())
    if not found:
        return Detection(agent, found=False)
    try:
        installed = agent.installed()
    except Exception:
        installed = None
    missing = [b for b in agent.requires if not shutil.which(b)]
    blocked = f"needs {', '.join(missing)} on PATH" if missing and not installed else None
    commands = agent.install() if agent.kind == "plugin" and not installed else []
    return Detection(agent, found=True, installed=installed, blocked=blocked, commands=commands)


def detect_all() -> list[Detection]:
    return [detect(a) for a in AGENTS]
