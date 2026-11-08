"""Which AI agents are on this machine, and how ``honcho install`` adds Honcho to each.

Mirrors the "General agents" and "Coding agents" sections of the docs
integrations catalog (``docs/snippets/integrations-catalog.jsx``).

Finding an agent works like vercel-labs/skills: a plain filesystem check, its
binary on PATH or its config directory on disk, plus the variables an agent
sets when it is the one running ``honcho install``. Nothing is parsed.
``honcho install`` never edits an agent's config files either; it only runs
the agent's own commands. Three kinds:

- ``plugin``: the agent has an official Honcho plugin, installed with the
  agent's own plugin commands.
- ``mcp``: the hosted Honcho MCP server, registered with the agent's own
  ``mcp add`` command, plus the honcho-memory skill.
- ``suggest``: ``honcho install`` changes nothing and points at the setup guide.
  Cloud agents (Grok Bot) get a setup prompt to paste instead.

Whether Honcho is already set up comes from asking the agent itself (its
``plugin list`` / ``mcp get`` / ``config get`` command), never from reading
its files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DOCS = "https://honcho.dev/docs/v3/guides"
CATALOG_URL = f"{DOCS}/overview"
MCP_DOCS = f"{DOCS}/integrations/mcp"
MCP_URL = "https://mcp.honcho.dev"

GROK_BOT_PROMPT = (
    "Set up Honcho memory using the plugin at "
    "https://github.com/plastic-labs/honcho/tree/main/integrations/cursor: "
    "connect its MCP server (https://mcp.honcho.dev, sign in with OAuth), "
    "install its skills, then run honcho-setup."
)


@dataclass(frozen=True)
class Credentials:
    """What install commands need from ``~/.honcho/config.json``."""

    api_key: str
    base_url: str

    @property
    def hosted(self) -> bool:
        return self.base_url.rstrip("/") == "https://api.honcho.dev"


# --------------------------------------------------------------------------- #
# locations


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


def _hermes_home() -> Path:
    return _env_path("HERMES_HOME", _home() / ".hermes")


def _zed_dir() -> Path:
    return _app_support() / "Zed" if sys.platform == "win32" else _xdg_config() / "zed"


def _goose_dir() -> Path:
    if sys.platform == "win32":
        return _app_support() / "Block" / "goose" / "config"
    return _xdg_config() / "goose"


def _dsh_present() -> bool:
    """``dsh`` is also Debian's distributed shell; only count DeepSeek's npm one."""
    exe = shutil.which("dsh")
    if not exe:
        return False
    try:
        return "deepseek" in os.path.realpath(exe).lower()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# asking an agent whether Honcho is set up


def _output(argv: list[str]) -> tuple[int, str] | None:
    """Run one of the agent's own status commands. ``None`` when it couldn't run."""
    exe = shutil.which(argv[0])
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, *argv[1:]], capture_output=True, text=True, errors="replace", timeout=60, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout


def _json_list(out: tuple[int, str] | None, key: str | None = None) -> list[dict[str, object]] | None:
    """The list of objects a ``--json`` status command printed, or None if it didn't."""
    if out is None or out[0] != 0:
        return None
    try:
        data = json.loads(out[1])
    except ValueError:
        return None
    if key is not None:
        data = data.get(key) if isinstance(data, dict) else None
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else None


def _claude_code_has_honcho() -> bool | None:
    out = _output(["claude", "plugin", "list", "--json"])
    return None if out is None or out[0] != 0 else "honcho@honcho" in out[1]


def _codex_has_honcho() -> bool | None:
    out = _output(["codex", "mcp", "get", "honcho"])
    return None if out is None else out[0] == 0


def _hermes_has_honcho() -> bool | None:
    out = _output(["hermes", "config", "get", "memory.provider"])
    return None if out is None or out[0] != 0 else out[1].strip() == "honcho"


def _hermes_has_plugin() -> bool:
    plugins = _json_list(_output(["hermes", "plugins", "list", "--json", "--user"]))
    return any(p.get("name") == "honcho" for p in plugins or [])


def _openclaw_has_honcho() -> bool | None:
    plugins = _json_list(_output(["openclaw", "plugins", "list", "--json"]), key="plugins")
    return None if plugins is None else any(p.get("id") == "openclaw-honcho" for p in plugins)


def _pi_has_honcho() -> bool | None:
    out = _output(["pi", "list"])
    return None if out is None or out[0] != 0 else "@honcho-ai/pi-honcho" in out[1]


# --------------------------------------------------------------------------- #
# install commands — each is the agent's own CLI


class Quiet(list[str]):
    """An install step that isn't the agent's own CLI; its output shows only if it fails."""


def _claude_code_install(_: Credentials) -> list[list[str]]:
    return [
        ["claude", "plugin", "marketplace", "add", "plastic-labs/claude-honcho"],
        ["claude", "plugin", "install", "honcho@honcho"],
    ]


def _codex_install(creds: Credentials) -> list[list[str]]:
    # `codex mcp add --url` reads a bearer token only from an env var, so the
    # key from the config goes in mcp-remote's header instead. This entry is the
    # one copy of the key outside ~/.honcho/config.json.
    return [
        [
            "codex", "mcp", "add", "honcho", "--",
            "npx", "-y", "mcp-remote", MCP_URL,
            "--header", f"Authorization:Bearer {creds.api_key}",
            "--header", "X-Honcho-Plugin:codex",
        ],
        # the skill that tells Codex when to recall and record, installed by vercel-labs/skills
        Quiet(["npx", "-y", "skills", "add", "plastic-labs/honcho", "--skill", "honcho-memory",
               "--agent", "codex", "--global", "--yes"]),
    ]


def _hermes_install(_: Credentials) -> list[list[str]]:
    # The plugin reads the key from ~/.honcho/config.json, so switching the
    # provider on replaces Hermes' interactive `memory setup`.
    enable = ["hermes", "config", "set", "memory.provider", "honcho"]
    if _hermes_has_plugin():
        return [enable]
    return [["hermes", "plugins", "install", "honcho", "--enable", "--yes-deps"], enable]


def _hermes_notes() -> list[str]:
    # Hermes reads these before ~/.honcho/config.json, so a key in one of them wins.
    files = dict.fromkeys([_hermes_home() / "honcho.json", _home() / ".hermes" / "honcho.json"])
    return [f"Hermes reads {f} before ~/.honcho/config.json, so a key there wins." for f in files if _exists(f)]


def _openclaw_install(_: Credentials) -> list[list[str]]:
    # The plugin reads the key from ~/.honcho/config.json. From ClawHub rather
    # than npm: OpenClaw cancels an npm-sourced install without a terminal, and
    # without --accept-capabilities it stops to ask for consent either way.
    # `plugins uninstall` leaves the entry behind as enabled: false, so a
    # reinstall would stay disabled without the explicit enable. Message capture
    # also needs conversation access, which the plugin's own warning says to
    # turn on this way.
    return [
        ["openclaw", "plugins", "install", "clawhub:@honcho-ai/openclaw-honcho", "--accept-capabilities"],
        ["openclaw", "plugins", "enable", "openclaw-honcho"],
        ["openclaw", "config", "set", "plugins.entries.openclaw-honcho.hooks.allowConversationAccess", "true"],
    ]


# --------------------------------------------------------------------------- #
# registry


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    kind: Literal["plugin", "mcp", "suggest"]
    docs_url: str
    bins: tuple[str, ...] = ()
    paths: Callable[[], list[Path]] = field(default=lambda: [])
    present: Callable[[], bool] | None = None  # overrides bins/paths when set
    cloud: bool = False  # runs elsewhere: never found here, always offered
    caller_env: tuple[str, ...] = ()  # set in the shell of this agent's tool calls
    has_honcho: Callable[[], bool | None] = field(default=lambda: None)
    install: Callable[[Credentials], list[list[str]]] | None = None
    requires: tuple[str, ...] = ()
    hosted_only: bool = False  # the hosted MCP server can't reach a self-hosted Honcho
    activate: str | None = None  # extra step after install, beyond restarting the agent
    notes: Callable[[], list[str]] = field(default=lambda: [])  # warnings to show once installed


AGENTS: tuple[Agent, ...] = (
    # ---- installed by `honcho install`
    Agent(
        id="claude-code", name="Claude Code", kind="plugin",
        docs_url=f"{DOCS}/integrations/claude-code",
        bins=("claude",), paths=lambda: [_env_path("CLAUDE_CONFIG_DIR", _home() / ".claude")],
        caller_env=("CLAUDECODE", "CLAUDE_CODE"),
        has_honcho=_claude_code_has_honcho, install=_claude_code_install,
        requires=("claude", "node"),  # the plugin's hooks run on node
    ),
    Agent(
        id="codex", name="Codex", kind="mcp", docs_url=f"{MCP_DOCS}#codex",
        bins=("codex",), paths=lambda: [_env_path("CODEX_HOME", _home() / ".codex")],
        caller_env=("CODEX_THREAD_ID", "CODEX_SANDBOX", "CODEX_CI"),
        has_honcho=_codex_has_honcho, install=_codex_install,
        requires=("codex", "npx"), hosted_only=True,
    ),
    Agent(
        id="hermes", name="Hermes Agent", kind="plugin",
        docs_url=f"{DOCS}/integrations/hermes",
        bins=("hermes",), paths=lambda: [_hermes_home()],
        has_honcho=_hermes_has_honcho, install=_hermes_install,
        requires=("hermes",), notes=_hermes_notes,
    ),
    Agent(
        id="openclaw", name="OpenClaw", kind="plugin",
        docs_url=f"{DOCS}/integrations/openclaw",
        bins=("openclaw",), paths=lambda: [_env_path("OPENCLAW_STATE_DIR", _home() / ".openclaw")],
        has_honcho=_openclaw_has_honcho, install=_openclaw_install,
        requires=("openclaw",), activate="openclaw gateway restart",
    ),
    Agent(
        id="pi", name="Pi", kind="plugin", docs_url=f"{DOCS}/integrations/pi",
        bins=("pi",), paths=lambda: [_home() / ".pi" / "agent"],
        has_honcho=_pi_has_honcho, install=lambda _: [["pi", "install", "npm:@honcho-ai/pi-honcho"]],
        requires=("pi",),
    ),
    # ---- set up from the guide
    Agent(
        id="grok-bot", name="Grok Bot", kind="suggest",
        docs_url=f"{DOCS}/integrations/grok-bot", cloud=True,
    ),
    Agent(
        id="opencode", name="OpenCode", kind="suggest", docs_url=f"{DOCS}/integrations/opencode",
        bins=("opencode",), paths=lambda: [_env_path("OPENCODE_CONFIG_DIR", _xdg_config() / "opencode")],
    ),
    Agent(
        id="kilo", name="Kilo Code", kind="suggest", docs_url=f"{DOCS}/integrations/kilo",
        bins=("kilo",), paths=lambda: [_xdg_config() / "kilo"],
    ),
    Agent(
        id="dsh", name="DeepSeek Harness", kind="suggest",
        docs_url=f"{DOCS}/integrations/deepseek-harness", present=_dsh_present,
    ),
    Agent(
        id="cursor", name="Cursor", kind="suggest", docs_url=f"{DOCS}/integrations/cursor",
        bins=("cursor", "cursor-agent"), paths=lambda: [_home() / ".cursor"],
    ),
    Agent(
        id="claude-desktop", name="Claude Desktop", kind="suggest", docs_url=f"{MCP_DOCS}#claude-desktop",
        paths=lambda: [_app_support() / "Claude"] if sys.platform in ("darwin", "win32") else [],
    ),
    Agent(
        id="windsurf", name="Windsurf", kind="suggest", docs_url=f"{MCP_DOCS}#windsurf",
        bins=("windsurf",), paths=lambda: [_home() / ".codeium" / "windsurf"],
    ),
    Agent(
        id="cline", name="Cline", kind="suggest", docs_url=f"{MCP_DOCS}#cline",
        paths=lambda: [_vscode_user() / "globalStorage" / "saoudrizwan.claude-dev"],
    ),
    Agent(
        id="vscode", name="VS Code", kind="suggest", docs_url=f"{MCP_DOCS}#vs-code-copilot-chat",
        bins=("code",), paths=lambda: [_vscode_user()],
    ),
    Agent(
        id="zed", name="Zed", kind="suggest", docs_url=f"{MCP_DOCS}#zed",
        bins=("zed", "zeditor"), paths=lambda: [_zed_dir()],
    ),
    Agent(
        id="goose", name="Goose", kind="suggest", docs_url=f"{MCP_DOCS}#goose",
        # config dir only: "goose" is also the binary of a common Go migration tool
        paths=lambda: [_goose_dir()],
    ),
)


@dataclass
class Detection:
    agent: Agent
    found: bool
    installed: bool | None = None  # None = can't tell
    blocked: str | None = None     # why the install can't run here


def calling_agent() -> Agent | None:
    """The agent whose tool call is running this command, from env vars it sets."""
    named = os.environ.get("AI_AGENT", "").strip().lower()
    for agent in AGENTS:
        if agent.id == named or any(os.environ.get(v) for v in agent.caller_env):
            return agent
    return None


def _found(agent: Agent) -> bool:
    if agent.present is not None:
        return agent.present()
    return any(shutil.which(b) for b in agent.bins) or any(_exists(p) for p in agent.paths())


def detect(agent: Agent, creds: Credentials | None = None) -> Detection:
    if agent.cloud or not _found(agent):
        return Detection(agent, found=False)
    if agent.install is None:
        return Detection(agent, found=True)
    missing = [b for b in agent.requires if not shutil.which(b)]
    if missing:
        return Detection(agent, found=True, blocked=f"needs {', '.join(missing)} on PATH")
    try:
        installed = agent.has_honcho()
    except Exception:
        installed = None
    blocked = None
    if not installed and agent.hosted_only and creds is not None and not creds.hosted:
        blocked = f"only works with api.honcho.dev; see {agent.docs_url}"
    return Detection(agent, found=True, installed=installed, blocked=blocked)


def detect_all(creds: Credentials | None = None) -> list[Detection]:
    """Detect every agent. Status commands run in parallel; some take seconds."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda a: detect(a, creds), AGENTS))
