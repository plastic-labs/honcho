"""Where a profile's Honcho settings live, and the one-time move that puts them there.

A Hermes profile reads one file, ``$HERMES_HOME/honcho.json``, and in it one block: ``hosts.<host>``
(``hermes``, or ``hermes_<profile>``). Once that block carries ``schemaVersion: 2`` it is all the
plugin reads: never the file's top level, another host's block or another file.

Until then the block is read as ``migrate_block`` will write it: exactly what hermes-plugin-honcho
1.1.0 resolved for the profile. 1.1.0 read the first of ``$HERMES_HOME/honcho.json``,
``~/.hermes/honcho.json`` and ``~/.honcho/config.json`` that existed, and in it each setting from the
block when set there, else from the top level. Reading before the migration therefore gives the same
answer as after it, and a later write to the top level or to another file changes nothing once it ran.

Ported from @honcho-ai/harness-plugin-core (``hostMigration``, ``effectiveBlock``, ``migrateHost``,
``saveHost``); the config lock is ``oauth._config_refresh_lock``.
"""

from __future__ import annotations

import copy
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from . import oauth

SCHEMA_VERSION = 2
CLOUD = "https://api.honcho.dev"
_HOST = "hermes"
_CONNECTION = ("apiKey", "baseUrl", "peerName", "workspace", "aiPeer")

# The rest of the block: Hermes's gateway identity settings, ``enabled`` and ``logging``, which 1.1.0 read from
# the block, else the top level. Aliases share one lookup.
_SETTINGS = (
    ("enabled",), ("pinUserPeer", "pinPeerName"), ("userPeerAliases",), ("runtimePeerPrefix",), ("a2aSessions",),
    ("logging",),
)
# How 1.1.0 decided the block's value counted: when truthy, when present; otherwise when not None.
_TRUTHY = set(_CONNECTION)
_PRESENT = {"userPeerAliases", "runtimePeerPrefix"}
# Settings 1.1.0 had and this release fixes (sessions, recall, injection, capture): the migration removes them.
DEAD_KEYS = (
    "observationMode", "observation", "recallMode", "recallSync", "initOnSessionStart", "injectionFrequency",
    "contextCadence", "dialecticCadence", "injection", "contextTokens", "dialecticReasoningLevel", "dialecticDynamic",
    "dialecticMaxChars", "dialecticDepth", "dialecticDepthLevels", "dialecticMaxInputChars", "reasoningHeuristic",
    "reasoningLevelCap", "queryRewrite", "firstTurnBaseWait", "firstTurnDialecticWait", "saveMessages",
    "writeFrequency", "messageMaxChars", "sessionStrategy", "sessionPeerPrefix", "sessionAiPeerPrefix", "sessions",
    "environment", "base_url", "timeout", "requestTimeout",
)


def _hosts(file: dict) -> dict:
    hosts = file.get("hosts")
    return hosts if isinstance(hosts, dict) else {}


def _legacy_block(file: dict, host: str) -> tuple[str, dict]:
    """(key, block) 1.1.0 read for ``host``: its own, or for an empty profile block the older ``hermes.<profile>``."""
    hosts = _hosts(file)
    block = hosts.get(host)
    if not (isinstance(block, dict) and block) and host.startswith(f"{_HOST}_"):
        dotted = f"{_HOST}.{host[len(_HOST) + 1:]}"
        if isinstance(hosts.get(dotted), dict):
            return dotted, hosts[dotted]
    return host, block if isinstance(block, dict) else {}


def _text(value: Any) -> str | None:
    """A value 1.1.0 would use (Python truthiness: its lookups were ``block.get(k) or top.get(k)``), as text."""
    return str(value) if value else None


def legacy_settings(file: dict, host: str) -> dict[str, str | None]:
    """How hermes-plugin-honcho 1.1.0 (``client.py``) resolved its connection from ``file``:

    - Its block is ``hosts.<host>``; an empty profile block falls back to ``hermes.<profile>``.
    - Each field is the block's value if set, else the top level's.
    - The server: the block's ``baseUrl`` or ``base_url``, then the top level's ``endpoint.baseUrl``,
      ``baseUrl``, ``base_url``, with a trailing ``/v3`` dropped; a URL with a control character was
      ignored. With none, ``environment: "local"`` meant ``http://localhost:8000``. The block's own
      ``endpoint`` was never read.

    None where it used a default: Honcho cloud, or the host name for ``workspace`` and ``aiPeer``.
    """
    block = _legacy_block(file, host)[1]

    def pick(key: str) -> str | None:
        return _text(block.get(key)) or _text(file.get(key))

    endpoint = file.get("endpoint")
    url = (_text(block.get("baseUrl")) or _text(block.get("base_url"))
           or (_text(endpoint.get("baseUrl")) if isinstance(endpoint, dict) else None)
           or _text(file.get("baseUrl")) or _text(file.get("base_url")))
    if url and not all(0x20 <= ord(c) < 0x7F for c in url):
        url = None
    if url:
        base_url: str | None = re.sub(r"/v\d+/*$", "", url).rstrip("/")
    else:
        base_url = "http://localhost:8000" if pick("environment") == "local" else None
    return {"apiKey": pick("apiKey"), "baseUrl": base_url, "peerName": pick("peerName"),
            "workspace": pick("workspace"), "aiPeer": pick("aiPeer")}


def normalize_url(url: str | None) -> str:
    """``url`` (Honcho cloud when empty) with a scheme, a lowercase host and no trailing slash."""
    text = (url or CLOUD).strip() or CLOUD
    if not text.startswith(("http://", "https://")):
        name = text.split("/")[0].split(":")[0].lower()
        text = f"{'http' if name in {'localhost', '127.0.0.1', '::1'} else 'https'}://{text}"
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    return f"{parts.scheme}://{parts.netloc.lower()}{parts.path.rstrip('/')}"


def same_server(a: str | None, b: str | None) -> bool:
    """Two URLs name the same server; empty means Honcho cloud."""
    return normalize_url(a) == normalize_url(b)


def is_marked(block: Any) -> bool:
    """The block is schema 2: the plugin reads only it."""
    version = block.get("schemaVersion") if isinstance(block, dict) else None
    return isinstance(version, (int, float)) and not isinstance(version, bool) and version >= SCHEMA_VERSION


def _block_counts(block: dict, keys: tuple[str, ...]) -> bool:
    """1.1.0 took this setting from the block rather than the top level."""
    key = keys[0]
    if key in _PRESENT:
        return key in block
    if key in _TRUTHY:
        return bool(block.get(key))
    return any(block.get(k) is not None for k in keys)


def legacy_path(path: Path) -> Path | None:
    """The file 1.1.0 read for the profile whose file is ``path``: the first of it, the default
    profile's and ``~/.honcho/config.json`` that exists."""
    from hermes_cli.profiles import _get_default_hermes_home

    for candidate in (path, _get_default_hermes_home() / "honcho.json", Path.home() / ".honcho" / "config.json"):
        if candidate.exists():
            return candidate
    return None


def _read(path: Path) -> dict:
    """The parsed file, {} when missing. A file that does not parse raises ValueError."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def _source(path: Path, file: dict) -> dict:
    """What 1.1.0 read for the profile: ``file`` (the parsed ``path``) when ``path`` exists, else the next file."""
    source = legacy_path(path)
    return file if source is None or source == path else _read(source)


def migrate_block(file: dict, host: str, source: dict) -> list[str]:
    """Bring ``hosts.<host>`` of ``file`` (the profile's own file) to schema 2, once: exactly what 1.1.0
    resolved from ``source`` (the file it read: ``file`` itself, or another when the profile had none) for
    the settings this release still has, then the mark; the ones it fixes (``DEAD_KEYS``) go. A block
    already signed in keeps its grant, and with it Honcho cloud, rather than a key or server from
    elsewhere. Returns the settings it set or removed (``-key``), not the mark; [] for a marked block."""
    if not isinstance(file.get("hosts"), dict):
        file["hosts"] = {}
    hosts = file["hosts"]
    if is_marked(hosts.get(host)):
        return []
    # Everything 1.1.0 read, taken before the block changes.
    old = legacy_settings(source, host)
    source_key, source_block = _legacy_block(source, host)
    top = {k: copy.deepcopy(v) for k, v in source.items() if k != "hosts"}

    if source is file:
        block = hosts[host] if isinstance(hosts.get(host), dict) else {}
        if source_key != host:
            block.update(copy.deepcopy(source_block))
            hosts.pop(source_key, None)
    else:
        block = copy.deepcopy(source_block)
        if "defaultHost" in top:  # the default profile picked its block by it
            file.setdefault("defaultHost", top["defaultHost"])
    hosts[host] = block
    changed: list[str] = []

    def put(key: str, value: Any) -> None:
        if key not in block or block[key] != value:
            block[key] = value
            changed.append(key)

    def drop(key: str) -> None:
        if key in block:
            del block[key]
            changed.append(f"-{key}")

    if old["baseUrl"] and same_server(old["baseUrl"], None):
        old["baseUrl"] = None
    signed_in = oauth.OAuthCredential.from_host_block(block) is not None
    for key in _CONNECTION:
        if signed_in and key in ("apiKey", "baseUrl"):
            continue
        if old[key] is not None:
            put(key, old[key])
        else:
            drop(key)
    for keys in _SETTINGS:
        if not _block_counts(block, keys):
            for key in keys:
                if key in top:
                    put(key, top[key])
    for key in DEAD_KEYS:  # `base_url` and `environment` were folded into baseUrl above
        drop(key)
    block["schemaVersion"] = SCHEMA_VERSION
    return changed


def migrated_block(file: dict, host: str, path: Path) -> dict:
    """``hosts.<host>`` of ``file`` (the parsed ``path``), migrated in place first when unmarked."""
    migrate_block(file, host, _source(path, file))
    return file["hosts"][host]


def effective_block(path: Path, host: str, file: dict | None = None) -> dict:
    """``hosts.<host>`` as the plugin reads it: as stored once marked, else as the migration will
    leave it. ``file`` is the parsed ``path`` when the caller has it. Raises ValueError when a file
    it needs does not parse."""
    view = copy.deepcopy(_read(path) if file is None else file)
    return copy.deepcopy(migrated_block(view, host, path))


@contextmanager
def locked_config(path: Path) -> Iterator[dict]:
    """Read-modify-write ``path`` under the config lock: yields the parsed file ({} when missing) and
    writes it back by temp file + rename, mode 0600. A file that does not parse raises ValueError
    before anything is written, since writing back a partial file would drop every other host."""
    from hermes_constants import mkdir_under_hermes_home
    from utils import atomic_json_write

    mkdir_under_hermes_home(path.parent)
    with oauth._refresh_lock, oauth._config_refresh_lock(path):
        file = oauth._read_config_strict(path)
        if not isinstance(file, dict):
            raise ValueError(f"{path} does not hold a JSON object")
        yield file
        atomic_json_write(path, file, mode=0o600)


def migrate_host(path: Path, host: str) -> list[str]:
    """Write the migration of ``hosts.<host>`` to ``path``, under the config lock. Call it when the
    profile starts. A marked block, or a profile that never had a file, takes no lock and writes
    nothing. Returns the settings it moved or removed; [] when there were none."""
    try:
        peek = _read(path)
    except (OSError, ValueError):
        return []  # 1.1.0 ignored a file it could not parse; leave it for the user to fix
    if is_marked(_hosts(peek).get(host)) or legacy_path(path) is None:
        return []
    with locked_config(path) as file:
        return migrate_block(file, host, _source(path, file))


def apply_patch(block: dict, patch: dict[str, Any]) -> None:
    """Merge ``patch`` into a migrated block (None removes a key) and mark it. A key belongs to the
    server it was saved for: a patch that moves the block to another server drops its key and grant
    unless it sets ``apiKey``. A new ``apiKey`` replaces a grant, whose access token held that slot."""
    if "baseUrl" in patch and "apiKey" not in patch and not same_server(patch["baseUrl"], block.get("baseUrl")):
        block.pop("apiKey", None)
        block.pop("oauth", None)
    if "apiKey" in patch and "oauth" not in patch:
        block.pop("oauth", None)
    for key, value in patch.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    block["schemaVersion"] = SCHEMA_VERSION


def save_host(path: Path, host: str, patch: dict[str, Any]) -> None:
    """Merge ``patch`` into ``hosts.<host>`` of ``path`` under the config lock, migrating the block first."""
    with locked_config(path) as file:
        apply_patch(migrated_block(file, host, path), patch)


def apply_granted_config(block: dict, granted: dict, host: str) -> None:
    """A sign-in's ``config`` (shaped like a whole file) applied to one block, as 1.1.0 merged it into the
    file: its top-level settings where the block's own did not count, then its ``hosts.<host>`` on top.
    The credential and server are the sign-in's, never the config's."""
    skip = {"hosts", "apiKey", "oauth", "baseUrl", "base_url", "environment", "endpoint", "schemaVersion"}
    for key, value in granted.items():
        if key not in skip and not _block_counts(block, (key,)):
            block[key] = copy.deepcopy(value)
    own = _hosts(granted).get(host)
    if isinstance(own, dict):
        _merge(block, {k: v for k, v in own.items() if k not in skip})


def _merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
