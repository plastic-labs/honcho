#!/usr/bin/env python3
"""Validate the Cursor/Grok Bot plugin against the Cursor Marketplace checklist.

Runs with the standard library only, from any directory:

    python3 integrations/cursor/scripts/validate.py

Checks the per-plugin manifest, the MCP config, every skill, rule and command
file, the logo, and the root marketplace manifest. Exits non-zero on the first
group of failures so CI can gate on it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
def find_repo_root(start: Path) -> Path:
    """Walk up from the plugin until the directory holding the root marketplace manifest."""
    for candidate in (start, *start.parents):
        if (candidate / ".cursor-plugin" / "marketplace.json").exists() or (candidate / ".git").exists():
            return candidate
    raise SystemExit("validate.py: no .cursor-plugin/marketplace.json or .git found above the plugin directory")


REPO_ROOT = find_repo_root(PLUGIN_DIR.parent)
PLUGIN_REL = PLUGIN_DIR.relative_to(REPO_ROOT).as_posix()
KEBAB = re.compile(r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        fail(f"{path.relative_to(REPO_ROOT)}: missing")
    except json.JSONDecodeError as e:
        fail(f"{path.relative_to(REPO_ROOT)}: invalid JSON ({e})")
    return None


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    m = FRONTMATTER.match(text)
    if not m:
        fail(f"{path.relative_to(REPO_ROOT)}: no YAML frontmatter")
        return {}
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def check_relative(path_value: str, where: str) -> None:
    if path_value.startswith("/") or ".." in Path(path_value).parts:
        fail(f"{where}: path {path_value!r} must be relative and must not use '..'")


def check_manifest() -> dict:
    manifest = load_json(PLUGIN_DIR / ".cursor-plugin" / "plugin.json") or {}
    where = f"{PLUGIN_REL}/.cursor-plugin/plugin.json"
    for key in ("name", "version", "description", "author", "license", "logo"):
        if not manifest.get(key):
            fail(f"{where}: missing {key!r}")
    name = manifest.get("name", "")
    if name and not KEBAB.match(name):
        fail(f"{where}: name {name!r} is not lowercase kebab-case")
    if not (manifest.get("author") or {}).get("name"):
        fail(f"{where}: author.name is required")
    for key in ("logo", "mcpServers", "skills", "rules", "commands", "agents", "hooks"):
        value = manifest.get(key)
        if isinstance(value, str):
            check_relative(value, where)
            if not (PLUGIN_DIR / value).exists():
                fail(f"{where}: {key} points at {value!r}, which does not exist")
    if "hooks" in manifest:
        fail(f"{where}: hooks are declared; Grok Bot never runs them and marketplace clones run no install step")
    return manifest


def check_mcp(manifest: dict) -> None:
    rel = manifest.get("mcpServers", "./mcp.json")
    path = PLUGIN_DIR / rel
    config = load_json(path) or {}
    where = f"{PLUGIN_REL}/{rel.lstrip('./')}"
    servers = config.get("mcpServers") or {}
    if not servers:
        fail(f"{where}: no mcpServers declared")
    declared = set((manifest.get("variables") or {}).get("properties", {}).keys())
    for server_name, server in servers.items():
        if "command" in server:
            fail(f"{where}: server {server_name!r} is stdio; use the hosted server over HTTP")
        if not str(server.get("url", "")).startswith("https://"):
            fail(f"{where}: server {server_name!r} must use an https:// url")
        for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", json.dumps(server)):
            if var not in declared:
                fail(f"{where}: ${{{var}}} is used but not declared under manifest 'variables'")
        plugin_header = (server.get("headers") or {}).get("X-Honcho-Plugin", "")
        expected = f"cursor-honcho/{manifest.get('version')}"
        if plugin_header != expected:
            fail(f"{where}: X-Honcho-Plugin header is {plugin_header!r}, expected {expected!r}")


def check_components(manifest: dict) -> None:
    skills_dir = PLUGIN_DIR / manifest.get("skills", "./skills/")
    skill_files = sorted(skills_dir.glob("*/SKILL.md")) if skills_dir.is_dir() else []
    if not skill_files:
        fail(f"{PLUGIN_REL}/skills: no SKILL.md files found")
    for path in skill_files:
        fm = frontmatter(path)
        rel = path.relative_to(REPO_ROOT)
        if fm.get("name") != path.parent.name:
            fail(f"{rel}: frontmatter name {fm.get('name')!r} must equal the folder name {path.parent.name!r}")
        if len(fm.get("description", "")) < 40:
            fail(f"{rel}: description is missing or too short to trigger reliably")
    for key, pattern in (("rules", "*.mdc"), ("commands", "*.md"), ("agents", "*.md")):
        folder = manifest.get(key)
        if not folder:
            continue
        for path in sorted((PLUGIN_DIR / folder).glob(pattern)):
            fm = frontmatter(path)
            if not fm.get("description"):
                fail(f"{path.relative_to(REPO_ROOT)}: frontmatter needs a description")
            if key != "rules" and fm.get("name") != path.stem:
                fail(f"{path.relative_to(REPO_ROOT)}: frontmatter name must equal {path.stem!r}")
    logo = PLUGIN_DIR / manifest.get("logo", "")
    if logo.suffix.lower() not in {".svg", ".png"}:
        fail(f"{PLUGIN_REL}: logo must be an .svg or .png committed in the repo")
    for required in ("README.md", "LICENSE", "CHANGELOG.md"):
        if not (PLUGIN_DIR / required).exists():
            fail(f"{PLUGIN_REL}/{required}: missing")


def check_marketplace(manifest: dict) -> None:
    path = REPO_ROOT / ".cursor-plugin" / "marketplace.json"
    market = load_json(path) or {}
    where = ".cursor-plugin/marketplace.json"
    for key in ("name", "owner", "plugins"):
        if not market.get(key):
            fail(f"{where}: missing {key!r}")
    names: list[str] = []
    for entry in market.get("plugins") or []:
        names.append(entry.get("name", ""))
        source = entry.get("source")
        source_path = source if isinstance(source, str) else (source or {}).get("path", "")
        check_relative(source_path, where)
        plugin_dir = REPO_ROOT / source_path
        if not (plugin_dir / ".cursor-plugin" / "plugin.json").exists():
            fail(f"{where}: entry {entry.get('name')!r} source {source_path!r} has no .cursor-plugin/plugin.json")
        if plugin_dir.resolve() == PLUGIN_DIR and entry.get("name") != manifest.get("name"):
            fail(f"{where}: entry name {entry.get('name')!r} differs from the plugin manifest name {manifest.get('name')!r}")
    if len(names) != len(set(names)):
        fail(f"{where}: plugin names must be unique")


def main() -> int:
    manifest = check_manifest()
    check_mcp(manifest)
    check_components(manifest)
    check_marketplace(manifest)
    if errors:
        print(f"{PLUGIN_REL} validation failed:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"{PLUGIN_REL} validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
