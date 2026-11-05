"""Detect which Honcho integrations are wired up on this machine.

Each detector looks at the integration's own files — plugin manifests, hook
registrations, MCP client entries — plus its ``hosts.<name>`` block in the
shared ``config.json``. Anything lifted from those files goes through
:func:`honcho_cli.redact.redact_tree`, so endpoints are shown and keys appear
only as a prefix, last four and fingerprint.

The raw keys an integration holds are returned separately from the report, so
``honcho config --deep`` can test each one against the API without the value
ever being part of what gets printed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from honcho_cli import config as cfg
from honcho_cli.redact import redact_tree, redact_value, safe_url

# Patched by tests; read through _home() so the override is honoured.
HOME = Path.home()

_MCP_CONTAINERS = ("mcpServers", "mcp_servers", "servers", "context_servers", "extensions")
_MCP_FIELDS = ("type", "url", "serverUrl", "uri", "command", "args", "headers", "http_headers", "env")


def detect_integrations(raw_config: dict | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(report, keys)``.

    ``report`` is ``{"detected": {name: entry}, "mcp_clients": {...},
    "checked": [...]}``, fully redacted. ``keys`` lists the raw keys those
    entries hold — ``{group, name, section, where, target, value}`` — and must
    never be printed.
    """
    hosts = raw_config.get("hosts") if isinstance(raw_config, dict) else None
    hosts = hosts if isinstance(hosts, dict) else {}
    keys: list[dict[str, Any]] = []
    detected: dict[str, Any] = {}
    for name, detector in _DETECTORS:
        entry = detector(hosts)
        if entry is not None:
            keys.extend({"group": "detected", "name": name, **k} for k in entry.pop("_keys", []))
            detected[name] = entry
    clients: dict[str, Any] = {}
    for name, path in _mcp_client_files():
        entry = _mcp_client_file(path)
        if entry is not None:
            keys.extend({"group": "mcp_clients", "name": name, **k} for k in entry.pop("_keys", []))
            clients[name] = entry
    report = {
        "detected": detected,
        "mcp_clients": clients,
        "checked": [name for name, _ in _DETECTORS] + [name for name, _ in _mcp_client_files()],
    }
    return report, keys


# --------------------------------------------------------------------------- #
# detectors — each returns an entry or None


def _claude_code(hosts: dict) -> dict | None:
    evidence: list[str] = []
    version = None
    plugins: list[dict[str, Any]] = []
    installed = _read_json(_home() / ".claude" / "plugins" / "installed_plugins.json")
    registry = installed.get("plugins") if isinstance(installed, dict) else None
    if isinstance(registry, dict):
        for key, value in registry.items():
            if "honcho" not in key.lower():
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict):
                    # The user-scope install is the one that runs everywhere.
                    if version is None or item.get("scope") == "user":
                        version = item.get("version") or version
                    project = item.get("projectPath")
                    plugins.append(
                        {
                            "name": key,
                            "version": item.get("version"),
                            "scope": item.get("scope"),
                            "project": _rel(Path(project)) if isinstance(project, str) else None,
                        }
                    )
        if plugins:
            evidence.append("~/.claude/plugins/installed_plugins.json")
    statusline = None
    settings = _read_json(_home() / ".claude" / "settings.json")
    if isinstance(settings, dict):
        enabled = settings.get("enabledPlugins")
        on = sorted(k for k, v in enabled.items() if "honcho" in k.lower() and v) if isinstance(enabled, dict) else []
        for plugin in plugins:
            plugin["enabled_globally"] = plugin["name"] in on
        line = settings.get("statusLine")
        if isinstance(line, dict) and "honcho" in str(line.get("command", "")):
            # Only the script's path: the command line itself can carry an
            # env-assignment such as HONCHO_API_KEY=… .
            script = next(
                (t for t in str(line["command"]).split()
                 if t.startswith(("~/", "/", "./")) and "=" not in t and "://" not in t and "@" not in t
                 and t.endswith((".sh", ".js", ".mjs", ".cjs", ".ts", ".py"))),
                None,
            )
            if script is None:
                statusline = "custom command"
            else:
                statusline = script if script.startswith("~/") else _rel(Path(script))
        if on or statusline:
            evidence.append("~/.claude/settings.json")
    claude_json = _read_json(_home() / ".claude.json")
    mcp = _mcp_entry(claude_json, "~/.claude.json")
    # `claude mcp add` defaults to local scope: projects["<dir>"].mcpServers.
    project_mcp: list[dict[str, Any]] = []
    projects = claude_json.get("projects") if isinstance(claude_json, dict) else None
    if isinstance(projects, dict):
        for project, data in projects.items():
            entry = _mcp_entry(data, "~/.claude.json", label_prefix=f'projects["{_rel(Path(project))}"].')
            if entry is not None:
                project_mcp.append({"project": _rel(Path(project)), **entry})
    if mcp is not None or project_mcp:
        evidence.append("~/.claude.json")
    blocks = _host_blocks(hosts, "claude_code", "claude-code")
    return _finish(
        evidence, blocks, hosts, version=version, plugins=plugins or None,
        statusline=statusline, mcp=mcp, project_mcp=project_mcp or None,
    )


def _codex(hosts: dict) -> dict | None:
    codex_home = Path(os.environ.get("CODEX_HOME") or _home() / ".codex")
    evidence: list[str] = []
    data = _read_toml(codex_home / "config.toml")
    mcp = _mcp_entry(data, f"{_rel(codex_home)}/config.toml")
    hooks_enabled = None
    if mcp is not None:
        evidence.append(f"{_rel(codex_home)}/config.toml: [mcp_servers.honcho]")
    if isinstance(data, dict) and isinstance(data.get("features"), dict):
        hooks_enabled = data["features"].get("hooks")
    if _text_mentions(codex_home / "hooks.json", "honcho"):
        evidence.append(f"{_rel(codex_home)}/hooks.json: honcho hooks registered")
    bundle = codex_home / "honcho" / "codex-honcho.mjs"
    if bundle.exists():
        evidence.append(f"{_rel(bundle)}: staged hook bundle")
    blocks = _host_blocks(hosts, "codex")
    return _finish(evidence, blocks, hosts, mcp=mcp, hooks_enabled=hooks_enabled)


def _cursor(hosts: dict) -> dict | None:
    evidence: list[str] = []
    if _text_mentions(_home() / ".cursor" / "hooks.json", "honcho"):
        evidence.append("~/.cursor/hooks.json: honcho hooks registered")
    mcp = _mcp_entry(_read_json(_home() / ".cursor" / "mcp.json"), "~/.cursor/mcp.json")
    if mcp is not None:
        evidence.append("~/.cursor/mcp.json: mcpServers.honcho")
    if (cfg.CONFIG_DIR / "plugins" / "cursor-honcho").is_dir():
        evidence.append(f"{_rel(cfg.CONFIG_DIR)}/plugins/cursor-honcho: plugin checkout")
    blocks = _host_blocks(hosts, "cursor")
    return _finish(evidence, blocks, hosts, mcp=mcp)


def _opencode(hosts: dict) -> dict | None:
    config_dir = Path(os.environ.get("OPENCODE_CONFIG_DIR") or _home() / ".config" / "opencode")
    evidence: list[str] = []
    version = None
    for name in ("opencode.json", "opencode.jsonc"):
        text = _read_text(config_dir / name)
        if text and "opencode-honcho" in text:
            # Version-shaped specs only: a git/tarball spec can carry a token.
            match = re.search(r"opencode-honcho@(\d[\w.+-]*)", text)
            version = version or (match.group(1) if match else None)
            evidence.append(f"{_rel(config_dir / name)}: plugin entry" + (f" @{match.group(1)}" if match else ""))
    if (cfg.CONFIG_DIR / "opencode-import-state.json").exists():
        evidence.append(f"{_rel(cfg.CONFIG_DIR)}/opencode-import-state.json")
    blocks = _host_blocks(hosts, "opencode")
    return _finish(evidence, blocks, hosts, version=version)


def _openclaw(hosts: dict) -> dict | None:
    root = _home() / ".openclaw"
    evidence: list[str] = []
    plugin = None
    workspace = base_url = None
    data = _read_json(root / "openclaw.json")
    plugins = data.get("plugins") if isinstance(data, dict) else None
    entries = plugins.get("entries") if isinstance(plugins, dict) else None
    entry = entries.get("openclaw-honcho") if isinstance(entries, dict) else None
    keys: list[dict[str, Any]] = []
    if isinstance(entry, dict):
        evidence.append("~/.openclaw/openclaw.json: plugins.entries.openclaw-honcho")
        config = entry.get("config") if isinstance(entry.get("config"), dict) else {}
        api_key = config.get("apiKey")
        if isinstance(api_key, str) and api_key.strip() and "${" not in api_key:
            keys = [
                {
                    "section": "plugin",
                    "where": "~/.openclaw/openclaw.json plugins.entries.openclaw-honcho.config.apiKey",
                    "target": _openclaw_target(config),
                    "value": api_key.strip(),
                    "format_error": None,
                }
            ]
        plugin = {
            "path": "~/.openclaw/openclaw.json",
            "key": "plugins.entries.openclaw-honcho",
            **redact_tree({k: entry.get(k) for k in ("enabled", "config", "hooks") if k in entry}),
        }
        config = entry.get("config")
        if isinstance(config, dict):
            workspace = config.get("workspaceId") if isinstance(config.get("workspaceId"), str) else None
            base_url = config.get("baseUrl") if isinstance(config.get("baseUrl"), str) else None
    package = _read_json(root / "extensions" / "openclaw-honcho" / "package.json")
    version = package.get("version") if isinstance(package, dict) else None
    if version:
        evidence.append(f"~/.openclaw/extensions/openclaw-honcho: {version}")
    if (cfg.CONFIG_DIR / "openclaw-peers.json").exists():
        evidence.append(f"{_rel(cfg.CONFIG_DIR)}/openclaw-peers.json")
    blocks = _host_blocks(hosts, "openclaw")
    return _finish(
        evidence, blocks, hosts, version=version, plugin=plugin, workspace=workspace,
        base_url=base_url, _keys=keys,
    )


def _openclaw_target(config: dict) -> str | None:
    """The API the plugin sends its key to, when its config says so literally.

    Without ``baseUrl`` the plugin falls back to ``HONCHO_BASE_URL`` from the
    gateway's environment (process env, ``~/.openclaw/.env``, the config's
    ``env`` block) and then to hosted — not something this shell can see, so
    the key is left untested rather than sent to a guess.
    """
    base = config.get("baseUrl")
    return base if _literal_url(base) else None


def _dsh(hosts: dict) -> dict | None:
    evidence: list[str] = []
    if (cfg.CONFIG_DIR / "dsh").is_dir():
        evidence.append(f"{_rel(cfg.CONFIG_DIR)}/dsh: capture cursors")
    return _finish(evidence, _host_blocks(hosts, "dsh"), hosts)


def _hermes(hosts: dict) -> dict | None:
    hermes_home = Path(os.environ.get("HERMES_HOME") or _home() / ".hermes")
    evidence: list[str] = []
    own_file = None
    data = _read_json(hermes_home / "honcho.json")
    if isinstance(data, dict):
        own_hosts = data.get("hosts")
        own_file = {
            "path": f"{_rel(hermes_home)}/honcho.json",
            "baseUrl": redact_tree(data.get("baseUrl"), key="baseUrl"),
            "hosts": sorted(own_hosts) if isinstance(own_hosts, dict) else [],
        }
        evidence.append(f"{_rel(hermes_home)}/honcho.json")
    manifest = _read_text(hermes_home / "plugins" / "honcho" / "plugin.yaml")
    version = None
    if manifest:
        match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", manifest, re.MULTILINE)
        version = match.group(1) if match else None
        evidence.append(f"{_rel(hermes_home)}/plugins/honcho: {version or 'installed'}")
    config_yaml = _read_text(hermes_home / "config.yaml")
    if config_yaml and re.search(r"provider:\s*honcho\b", config_yaml):
        evidence.append(f"{_rel(hermes_home)}/config.yaml: memory.provider = honcho")
    blocks = [k for k in hosts if k == "hermes" or k.startswith(("hermes_", "hermes."))]
    return _finish(evidence, blocks, hosts, version=version, own_file=own_file)


def _pi(hosts: dict) -> dict | None:
    return _finish([], _host_blocks(hosts, "pi"), hosts)


def _sillytavern(hosts: dict) -> dict | None:
    return _finish([], _host_blocks(hosts, "sillytavern"), hosts)


def _nanobot(hosts: dict) -> dict | None:
    data = _read_json(_home() / ".nanobot" / "config.json")
    block = data.get("honcho") if isinstance(data, dict) else None
    if not isinstance(block, dict):
        return None
    workspace = block.get("workspaceId") if isinstance(block.get("workspaceId"), str) else None
    return _finish(
        ["~/.nanobot/config.json: honcho"], [], hosts, config=redact_tree(block), workspace=workspace
    )


_DETECTORS: tuple[tuple[str, Callable[[dict], dict | None]], ...] = (
    ("claude_code", _claude_code),
    ("codex", _codex),
    ("cursor", _cursor),
    ("opencode", _opencode),
    ("openclaw", _openclaw),
    ("dsh", _dsh),
    ("hermes", _hermes),
    ("pi", _pi),
    ("sillytavern", _sillytavern),
    ("nanobot", _nanobot),
)


# --------------------------------------------------------------------------- #
# generic MCP clients (no harness plugin, just a server entry)


def _mcp_client_files() -> list[tuple[str, Path]]:
    home = _home()
    if sys.platform == "darwin":
        app_support = home / "Library" / "Application Support"
        claude_desktop = app_support / "Claude" / "claude_desktop_config.json"
        cline = app_support / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
        vscode = app_support / "Code" / "User" / "settings.json"
    elif sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
        claude_desktop = appdata / "Claude" / "claude_desktop_config.json"
        cline = appdata / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
        vscode = appdata / "Code" / "User" / "settings.json"
    else:
        claude_desktop = home / ".config" / "Claude" / "claude_desktop_config.json"
        cline = home / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
        vscode = home / ".config" / "Code" / "User" / "settings.json"
    return [
        ("claude_desktop", claude_desktop),
        ("windsurf", home / ".codeium" / "windsurf" / "mcp_config.json"),
        ("zed", home / ".config" / "zed" / "settings.json"),
        ("vscode", vscode),
        ("cline", cline),
        ("goose", home / ".config" / "goose" / "config.yaml"),
    ]


def _mcp_client_file(path: Path) -> dict | None:
    if path.suffix in (".yaml", ".yml"):
        text = _read_text(path)
        if text and "honcho" in text.lower():
            return {"path": _rel(path), "note": "YAML not parsed; a honcho entry is present"}
        return None
    data = _read_json(path)
    entry = _mcp_entry(data, _rel(path))
    if entry is None and isinstance(data, dict):
        entry = _mcp_entry(data.get("mcp"), _rel(path), label_prefix="mcp.")  # VS Code user settings
    return entry


def _mcp_entry(data: Any, path: str, *, label_prefix: str = "") -> dict | None:
    """The redacted ``honcho`` server entry from any MCP client config shape.

    Raw keys ride along under ``_keys`` for the caller to lift out.
    """
    if not isinstance(data, dict):
        return None
    for container in _MCP_CONTAINERS:
        block = data.get(container)
        if isinstance(block, dict) and isinstance(block.get("honcho"), dict):
            fields = {k: block["honcho"][k] for k in _MCP_FIELDS if k in block["honcho"]}
            label = f"{label_prefix}{container}.honcho"
            return {
                "path": path,
                "key": label,
                **redact_tree(fields),
                "_keys": _honcho_credentials(fields, f"{path} {label}"),
            }
    return None


_HOSTED_MCP = "mcp.honcho.dev"


def _mcp_target(fields: dict[str, Any]) -> str | None:
    """The Honcho API an MCP entry's key goes to — only for the shape the
    Honcho installers and docs write: a ``url`` (or ``serverUrl``/``uri``) on
    exactly the hosted MCP host, which forwards to the hosted API.

    Everything else is None and its key is not tested: a self-hosted MCP,
    ``mcp-remote`` (the URL and key are command-line args, which can't be
    read reliably), a stdio server, or a URL given as a variable.
    """
    direct = [fields.get(k) for k in ("url", "serverUrl", "uri") if isinstance(fields.get(k), str)]
    if direct and all(_literal_url(u) and _host(u) == _HOSTED_MCP for u in direct):
        return cfg.DEFAULT_BASE_URL
    return None


def _honcho_credentials(fields: dict[str, Any], where: str) -> list[dict[str, Any]]:
    """The Honcho key in an MCP entry: its ``Authorization`` header.

    Any other secret in the entry — env vars, args, other headers — is
    redacted but never sent anywhere. A templated value (``${…}``) can't be
    resolved here and is skipped. The hosted MCP server only accepts
    ``Bearer <key>``, so a bare key is recorded as malformed, not tested.
    """
    target = _mcp_target(fields)
    found: list[dict[str, Any]] = []
    for group in ("headers", "http_headers"):
        headers = fields.get(group)
        if not isinstance(headers, dict):
            continue
        for name, raw in headers.items():
            if str(name).lower() != "authorization" or not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text or "${" in text:
                continue
            bearer = re.fullmatch(r"(?i)bearer\s+(\S+)", text)
            found.append(
                {
                    "section": "mcp",
                    "where": f"{where}.{group}.{name}",
                    "target": target,
                    "value": bearer.group(1) if bearer else text,
                    "format_error": None if bearer else "missing 'Bearer ' prefix",
                }
            )
    return found


def _literal_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://")) and "${" not in value


def _host(url: str) -> str | None:
    """Hostname, or None when the authority holds characters URL parsers
    disagree on (a backslash reads as a path separator to WHATWG parsers)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if "\\" in parts.netloc or any(c.isspace() for c in parts.netloc):
        return None
    return (parts.hostname or "").lower() or None


# --------------------------------------------------------------------------- #
# helpers


def _home() -> Path:
    return HOME


def _rel(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(_home()))
    except ValueError:
        return str(path)


def _host_blocks(hosts: dict, *names: str) -> list[str]:
    return [name for name in names if isinstance(hosts.get(name), dict)]


def _block_value(hosts: dict, blocks: list[str], *keys: str) -> str | None:
    for block in blocks:
        data = hosts.get(block)
        for key in keys:
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, dict):
                value = value.get("baseUrl")  # endpoint: {baseUrl}
            if isinstance(value, str) and value:
                return value
    return None


def _finish(evidence: list[str], blocks: list[str], hosts: dict, **extra: Any) -> dict | None:
    if not evidence and not blocks:
        return None
    keys = list(extra.pop("_keys", None) or [])
    for section in ("mcp", "plugin", "config", "project_mcp"):
        block = extra.get(section)
        for item in block if isinstance(block, list) else [block]:
            if isinstance(item, dict):
                keys.extend(item.pop("_keys", []))
    workspace = extra.pop("workspace", None) or _block_value(hosts, blocks, "workspace", "workspaceId")
    base_url = extra.pop("base_url", None) or _block_value(hosts, blocks, "baseUrl", "endpoint")
    entry: dict[str, Any] = {
        "detected": True,
        "version": extra.pop("version", None),
        "workspace": redact_value("workspace", workspace) if workspace else None,
        "base_url": safe_url(base_url) if base_url else None,
        "host_blocks": blocks,
        "evidence": evidence,
    }
    entry.update({k: v for k, v in extra.items() if v is not None})
    if keys:
        entry["_keys"] = keys
    return entry


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _text_mentions(path: Path, needle: str) -> bool:
    text = _read_text(path)
    return bool(text) and needle in text.lower()


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_strip_jsonc(text))  # Zed and VS Code settings allow comments
    except json.JSONDecodeError:
        return None


def _strip_jsonc(text: str) -> str:
    """Drop ``//`` and ``/* */`` comments and trailing commas outside strings."""
    out: list[str] = []
    i, n, in_string = 0, len(text), False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            in_string = c != '"'
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # trailing comma
                continue
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _read_toml(path: Path) -> Any:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
