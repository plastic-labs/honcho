"""``hermes honcho`` subcommands: setup wizard, status, peers, sessions, identity, migrate."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.profiles import _get_default_hermes_home
from .attribution import HONCHO_APP_URL, attribution_url
from .client import _host_block, profile_host_key, resolve_active_host, resolve_config_path, HOST
from .client import resolve_global_config_path, session_scopes, spawn_context_thread
from .session_peers import sanitize_peer_id
from hermes_cli.config import cfg_get
from utils import read_json_or_empty

RULE = "─" * 40
_RETRY_HINT = "  Re-run 'hermes honcho setup' to retry, or choose an API key instead.\n"

# A cloned profile's block carries the operator's runtime-to-peer routing intent from the default block.
_CLONE_KEYS = ("pinUserPeer", "userPeerAliases", "runtimePeerPrefix")
_IDENTITY_MAPPING_KEYS = ("pinPeerName", "pinUserPeer", "userPeerAliases", "runtimePeerPrefix")
# Setup-wizard answer -> identity-mapping shape ("2"/pooled answers are handled inline).
_SHAPE_CHOICES = {"1": "single", "me": "single", "just-me": "single", "3": "multi", "others": "multi",
                  "e": "raw", "edit": "raw", "raw": "raw"}
# ── config access ──────────────────────────────────────────────────────────

_profile_override: str | None = None


def _host_key() -> str:
    """Active Honcho host key (``--target-profile`` override, else the active profile)."""
    if _profile_override:
        return HOST if _profile_override in {"default", "custom"} else profile_host_key(_profile_override)
    return resolve_active_host()


def _config_path() -> Path:
    """The honcho.json of the active profile (``--target-profile``'s when given): the only file its settings live in."""
    if _profile_override and _profile_override != "custom":
        try:
            from hermes_cli.profiles import get_profile_dir
            return get_profile_dir(_profile_override) / "honcho.json"
        except Exception:
            pass
    return resolve_config_path()


class _ReadConfig(dict):
    """A command's config, with the ``snapshot`` and ``path`` _write_config() needs to apply only its edits."""

    def __init__(self, raw: dict, path: Path):
        super().__init__(raw)
        self.snapshot, self.path = copy.deepcopy(raw), path


def _profile_block(path: Path, host: str) -> dict:
    """``hosts.<host>`` of a profile's honcho.json as the plugin reads it; {} when it holds no settings."""
    from .host_config import effective_block, legacy_path
    if legacy_path(path) is None:
        return {}
    try:
        block = effective_block(path, host)
    except (OSError, ValueError):
        return {}
    return block if any(k != "schemaVersion" for k in block) else {}


def _read_config() -> dict:
    """The active profile's honcho.json as a command edits it: the other host blocks as stored, the active
    host's block as the plugin reads it (migrated, when it is not yet), and no top level, which the plugin
    never reads. {} when the profile has no config at all."""
    from .host_config import effective_block, legacy_path
    path = _config_path()
    if legacy_path(path) is None:
        return _ReadConfig({}, path)
    try:
        block = effective_block(path, _host_key())
    except (OSError, ValueError):
        return _ReadConfig({}, path)  # a save refuses the file (ConfigWriteRefused) rather than replace it
    raw = read_json_or_empty(path)
    hosts = raw.get("hosts") if isinstance(raw.get("hosts"), dict) else {}
    view: dict = {"hosts": {**hosts, _host_key(): block}}
    if "defaultHost" in raw:
        view["defaultHost"] = raw["defaultHost"]
    return _ReadConfig(view, path)


class ConfigWriteRefused(ValueError):
    """honcho.json exists on disk but does not parse, so no command may overwrite it. A ValueError so
    Hermes' dashboard reports a refused save as a 400, also across ``plugins.isolation: host``."""


def _refuse_unparseable(path: Path) -> dict:
    """Return ``path``'s parsed content ({} when absent); raise ConfigWriteRefused when it exists but cannot
    be parsed, since writing back ``{}`` would drop every host."""
    from .oauth import _read_config_strict
    try:
        return _read_config_strict(path)
    except (OSError, ValueError) as e:
        raise ConfigWriteRefused(f"{path} exists but could not be read as JSON ({e}). Nothing was written. "
                                 "Fix or move the file, then re-run.") from e


def _apply_edits(base: dict, edited: dict, current: dict) -> dict:
    """Return ``current`` with the root keys and ``hosts.<h>.<k>`` the command changed (``base`` to ``edited``)
    applied. An untouched key keeps its on-disk value, so a rotation that landed while the command ran survives."""
    out = copy.deepcopy(current)
    for key in (set(base) | set(edited)) - {"hosts"}:
        if key in edited and edited[key] != base.get(key):
            out[key] = copy.deepcopy(edited[key])
        elif key not in edited and key in base:
            out.pop(key, None)
    base_hosts, edited_hosts = base.get("hosts") or {}, edited.get("hosts") or {}
    out_hosts = out.setdefault("hosts", {}) if (base_hosts or edited_hosts or "hosts" in current) else None
    for host in set(base_hosts) | set(edited_hosts):
        if host not in edited_hosts:
            out_hosts.pop(host, None)
            continue
        b, e = base_hosts.get(host) or {}, edited_hosts[host]
        block = out_hosts.setdefault(host, {})
        for key in set(b) | set(e):
            if key in e and e[key] != b.get(key):
                block[key] = copy.deepcopy(e[key])
            elif key not in e and key in b:
                block.pop(key, None)
    return out


def _write_config(cfg: dict, path: Path | None = None) -> None:
    """Persist ``cfg`` under the config lock. The object _read_config() returned has only its edits applied
    onto a fresh read of disk, whose active host block is migrated first and then marked; a plain dict is
    written whole."""
    from .host_config import SCHEMA_VERSION, locked_config, migrated_block
    path = path or getattr(cfg, "path", None) or _config_path()
    _refuse_unparseable(path)
    host = _host_key()
    with locked_config(path) as disk:
        migrated_block(disk, host, path)
        out = _apply_edits(cfg.snapshot, cfg, disk) if isinstance(cfg, _ReadConfig) else copy.deepcopy(cfg)
        disk.clear()
        disk.update(out)
        if isinstance(block := (disk.get("hosts") or {}).get(host), dict):
            block["schemaVersion"] = SCHEMA_VERSION
    if isinstance(cfg, _ReadConfig):  # a later write on the same object applies only edits made after this one
        cfg.snapshot, cfg.path = copy.deepcopy(dict(cfg)), path


def _label(host: str) -> str:
    return f"[{host}] " if host != "hermes" else ""


def _mask(key: str) -> str:
    return f"...{key[-8:]}" if len(key) > 8 else ("set" if key else "not set")


def _pref(block: dict, cfg: dict, key: str, default=None):
    """Host-block value, else ``default`` (a _read_config() view has no top level)."""
    return block.get(key) or cfg.get(key, default)


def _active_block(cfg: dict) -> dict:
    return (cfg.get("hosts") or {}).get(_host_key(), {})


def _set_field(cfg: dict, key: str, value, echo: str) -> None:
    """Write one key on the active host block and echo the change."""
    host = _host_key()
    cfg.setdefault("hosts", {}).setdefault(host, {})[key] = value
    print(f"  {_label(host)}{echo}")


def _save(cfg: dict) -> None:
    _write_config(cfg)
    print(f"  Saved to {_config_path()}\n")


def _default_block() -> dict:
    """The default profile's host block as the plugin reads it; {} when it holds no settings."""
    return _profile_block(_get_default_hermes_home() / "honcho.json", HOST)


def _resolve_api_key(cfg: dict, block: dict | None = None, *, env: bool = True) -> str:
    """API key for ``block`` (default: the active host's block), then env. A self-hosted
    http(s) or host:port ``baseUrl`` without a key yields "local" so credential guards accept it.
    ``env=False`` counts only what is on disk: a variable can vanish from the next process."""
    block = _host_block(cfg, _host_key()) if block is None else block
    key = block.get("apiKey") or (os.environ.get("HONCHO_API_KEY", "") if env else "")
    if key:
        return key
    base_url = (block.get("baseUrl") or (os.environ.get("HONCHO_BASE_URL", "") if env else "") or "").strip()
    if not base_url:
        return key
    from urllib.parse import urlparse
    try:
        parsed = urlparse(base_url)
    except (TypeError, ValueError):
        parsed = None
    if parsed and parsed.scheme in {"http", "https"} and parsed.netloc:
        return "local"
    lowered = base_url.lower()
    if lowered not in {"true", "false", "none", "null"} and any(c in base_url for c in ".:") and not base_url.isdigit():
        return "local"
    return key


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    sys.stdout.write(f"  {label}{f' [{default}]' if default else ''}: ")
    sys.stdout.flush()
    if secret and sys.stdin.isatty():
        from hermes_cli.secret_prompt import masked_secret_prompt
        val = masked_secret_prompt("")
    else:  # non-TTY (piped input, test runners) reads plaintext
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _yes(answer: str) -> bool:
    return answer.strip().lower() in {"y", "yes"}


# ── Honcho connection ──────────────────────────────────────────────────────

def _connect(host: str | None, *, reset: bool = False):
    """(hcfg, client) for ``host``; lazy imports so tests can patch client.*."""
    from .client import HonchoClientConfig, get_honcho_client, reset_honcho_client
    if reset:
        reset_honcho_client()
    hcfg = HonchoClientConfig.from_global_config(host=host, config_path=_config_path())
    return hcfg, get_honcho_client(hcfg)


# The Honcho session the CLI's own reads and uploads (identity seeding, file migration) go through.
_CLI_SESSION = "hermes-cli"


def _session_manager(hcfg, client):
    """(manager, session_key) with the CLI's session ensured (get_or_create is idempotent)."""
    from .session import HonchoSessionManager
    mgr = HonchoSessionManager(honcho=client, config=hcfg)
    mgr.get_or_create(_CLI_SESSION)
    return mgr, _CLI_SESSION


def _ensure_peer_exists(host_key: str | None = None, config_path: Path | None = None) -> bool:
    """Create the AI (and user) peer in Honcho if missing. Idempotent; False on failure."""
    try:
        from .client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=host_key, config_path=config_path or _config_path())
        if not hcfg.enabled or not (hcfg.api_key or hcfg.base_url):
            return False
        client = get_honcho_client(hcfg)
        client.peer(hcfg.ai_peer)
        if hcfg.peer_name:
            client.peer(hcfg.peer_name)
        return True
    except Exception:
        return False


# ── profile sync ───────────────────────────────────────────────────────────

def _inherit_defaults(block: dict, default_block: dict, cfg: dict, keys: tuple[str, ...]) -> None:
    """Copy ``keys`` (and peerName) from the default host block into ``block`` where unset."""
    for key in keys:
        if (val := default_block.get(key)) is not None and key not in block:
            block[key] = val
    if (peer_name := _pref(default_block, cfg, "peerName")) and "peerName" not in block:
        block["peerName"] = peer_name


def clone_honcho_for_profile(profile_name: str, *, in_hook: bool = False) -> bool:
    """Give a new profile a host block in its own honcho.json, cloned from the default profile's block
    (called during profile creation). False if Honcho isn't configured on the default profile or the
    profile already has a block.

    ``in_hook``: called from ``on_profile_created`` with the new profile's home bound; the peer is created
    off-thread (the hook runs inside the dashboard's create request)."""
    from .host_config import _hosts, legacy_path, locked_config, migrated_block
    default_block = _default_block()
    if not default_block and not os.environ.get("HONCHO_API_KEY"):
        return False
    new_host = profile_host_key(profile_name)
    if in_hook:
        path = get_hermes_home() / "honcho.json"
    else:
        from hermes_cli.profiles import get_profile_dir
        path = get_profile_dir(profile_name) / "honcho.json"
    source = legacy_path(path)
    if source is not None and new_host in _hosts(_refuse_unparseable(source)):
        return False

    new_block: dict = {}
    _inherit_defaults(new_block, default_block, {}, _CLONE_KEYS)
    # Carry a legacy default-block pinPeerName forward under the canonical key.
    if "pinUserPeer" not in new_block and default_block.get("pinPeerName") is not None:
        new_block["pinUserPeer"] = default_block["pinPeerName"]
    # AI peer is profile-specific (bare profile name: Honcho peer IDs allow no dots);
    # workspace is shared so all profiles see the same context.
    new_block.update(aiPeer=profile_name, workspace=default_block.get("workspace") or HOST)
    _refuse_unparseable(path)
    with locked_config(path) as file:
        # What the profile read until now (the top level of the file 1.1.0 read for it), then the clone.
        block = migrated_block(file, new_host, path)
        block.update(new_block)
        # The default host's apiKey is not inherited; the block stays unenabled until this profile signs in.
        if _resolve_api_key({}, block, env=False):
            block["enabled"] = default_block.get("enabled", True)
    if not in_hook:
        _ensure_peer_exists(new_host, path)  # eager so the peer exists before first message
        return True
    # Off the request thread, but not a daemon: `hermes profile create` exits right after the hook, and a
    # daemon thread would die before its request. The thread carries the bound profile home with it.
    spawn_context_thread(_ensure_peer_exists, name="honcho-profile-peer", args=(new_host, path), daemon=False).start()
    return True


def migrate_profile_host(old_name: str, new_name: str) -> list[str]:
    """Re-key a renamed profile's host block (``hermes_<old>`` or legacy ``hermes.<old>``) to
    ``hermes_<new>``, keeping its peer: a block without ``aiPeer`` pins the old name, which its peer was
    created under. The profile's own honcho.json first gets the block as the profile read it, so one not
    yet migrated keeps settings it read from another file; the default profile's file and ~/.honcho's are
    re-keyed too, so no later profile with the old name picks up a leftover block. Returns one line per
    file touched. Runs from ``on_profile_renamed`` with the renamed profile's home bound."""
    from .host_config import _legacy_block, legacy_path, locked_config, migrated_block
    old_host, legacy_old_host = profile_host_key(old_name), f"{HOST}.{old_name}"
    new_host = profile_host_key(new_name)
    own = _config_path()
    lines: list[str] = []
    try:
        source = legacy_path(own)
        if source is not None and source != own and _legacy_block(_refuse_unparseable(source), old_host)[1]:
            _refuse_unparseable(own)
            with locked_config(own) as file:
                migrated_block(file, old_host, own)
    except ConfigWriteRefused as e:
        return [f"Honcho host block not migrated: {e}"]
    seen: set[Path] = set()
    for path in (own, _get_default_hermes_home() / "honcho.json", resolve_global_config_path()):
        resolved = path.resolve() if path.exists() else path
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            hosts = _refuse_unparseable(path).get("hosts")
        except ConfigWriteRefused as e:
            lines.append(f"Honcho host block not migrated: {e}")
            continue
        if not isinstance(hosts, dict):
            continue
        source_key = old_host if old_host in hosts else legacy_old_host
        if source_key not in hosts:
            continue
        if new_host in hosts:
            lines.append(f"⚠ Honcho host block not migrated: {new_host} already exists in {path}")
            continue
        with locked_config(path) as cfg:
            block = cfg["hosts"].pop(source_key, {})
            if isinstance(block, dict) and "aiPeer" not in block:
                block = {**block, "aiPeer": old_name}
            cfg["hosts"][new_host] = block
        lines.append(f"✓ Honcho host updated: {source_key} → {new_host}")
    return lines


def _sync_profiles(verbose: bool) -> int:
    """Clone host blocks for profiles lacking one, each into its own honcho.json; returns the count created."""
    say = print if verbose else (lambda *a: None)
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception as e:
        return say(f"  Could not list profiles: {e}\n") or 0
    if not _default_block() and not os.environ.get("HONCHO_API_KEY"):
        return say("  Honcho not configured on default profile. Run 'hermes honcho setup' first.\n") or 0

    created = skipped = 0
    for p in (p for p in profiles if p.name != "default"):
        try:
            cloned = clone_honcho_for_profile(p.name)
        except ConfigWriteRefused as e:
            return say(f"  {e}\n") or created
        if cloned:
            say(f"  + {p.name} -> {profile_host_key(p.name)}")
            created += 1
        else:
            skipped += 1
    say(f"\n  {created} profile(s) synced." if created else "  All profiles already have Honcho config.")
    if skipped:
        say(f"  {skipped} profile(s) already configured (skipped).")
    say()
    return created


def cmd_sync(args) -> None:
    """Sync Honcho config to all existing profiles (inherits from the default block)."""
    _sync_profiles(verbose=True)


def sync_honcho_profiles_quiet() -> int:
    """Sync host blocks for all profiles from `hermes update`; no output, no exceptions."""
    return _sync_profiles(verbose=False)


def cmd_enable(args) -> None:
    """Enable Honcho for the active profile; refuses a block that cannot authenticate."""
    cfg = _read_config()
    host = _host_key()
    label = _label(host)
    block = cfg.setdefault("hosts", {}).setdefault(host, {})
    if not _resolve_api_key(cfg, block, env=False):
        profile = _active_profile_name()
        setup = "hermes honcho setup" + (f" --target-profile {profile}" if profile != "default" else "")
        return print(f"  {label}Honcho stays disabled: no API key or base URL is configured for this profile, and the default "
                     f"profile's key is not shared.\n  Run '{setup}' to sign in, or set apiKey on hosts.{host} in {_config_path()}.\n")
    if block.get("enabled") is True:
        return print(f"  {label}Honcho is already enabled.\n")
    block["enabled"] = True

    if not block.get("aiPeer"):  # fresh profile block: take the user peer and workspace from default
        default_block = _default_block()
        _inherit_defaults(block, default_block, {}, ())
        block.setdefault("aiPeer", host.split(".", 1)[1] if "." in host else host)
        block.setdefault("workspace", default_block.get("workspace") or HOST)
    _write_config(cfg)
    print(f"  {label}Honcho enabled.")  # before the (possibly slow) peer-creation round trip
    peer_state = f"Peer '{block.get('aiPeer', host)}' ready." if _ensure_peer_exists(host) else "Peer creation deferred (no connection)."
    print(f"  {label}{peer_state}\n  Saved to {_config_path()}\n")


def cmd_disable(args) -> None:
    """Disable Honcho for the active profile."""
    cfg = _read_config()
    host = _host_key()
    block = cfg_get(cfg, "hosts", host, default={})
    if not block or block.get("enabled") is False:
        return print(f"  {_label(host)}Honcho is already disabled.\n")
    block["enabled"] = False
    print(f"  {_label(host)}Honcho disabled.")
    _save(cfg)


# ── identity mapping (setup wizard) ────────────────────────────────────────

def _resolve_effective_identity_mapping(cfg: dict, hermes_host: dict) -> tuple[bool, dict, str, bool, bool]:
    """``(pin, aliases, prefix, aliases_from_root, prefix_from_root)`` for the active host,
    mirroring ``from_global_config`` precedence (host over root; ``pinUserPeer`` beats
    ``pinPeerName``) so setup classifies the shape the gateway actually runs with.
    ``*_from_root`` lets writes skip inherited values."""
    pin_sources = (hermes_host.get("pinUserPeer"), hermes_host.get("pinPeerName"),
                   cfg.get("pinUserPeer"), cfg.get("pinPeerName"))
    pin = bool(next((v for v in pin_sources if v is not None), False))

    def _inherit(key):
        if key in hermes_host:
            return hermes_host.get(key), False
        val = cfg.get(key)
        return val, val is not None

    aliases_src, aliases_from_root = _inherit("userPeerAliases")
    prefix_src, prefix_from_root = _inherit("runtimePeerPrefix")
    aliases = aliases_src if isinstance(aliases_src, dict) else {}
    return pin, aliases, str(prefix_src or ""), aliases_from_root, prefix_from_root


def _scrub_identity_mapping(hermes_host: dict) -> None:
    """Drop every peer-mapping key so a stale alias/prefix/pin can't bleed into the new shape."""
    for key in _IDENTITY_MAPPING_KEYS:
        hermes_host.pop(key, None)


def _migrate_pin_key(block: dict) -> bool:
    """Rewrite legacy ``pinPeerName`` to canonical ``pinUserPeer`` in place (the
    resolver prefers the canonical key). Returns True if the block changed."""
    if "pinPeerName" not in block:
        return False
    block.setdefault("pinUserPeer", block.pop("pinPeerName"))
    return True


def _gateway_platforms() -> list[str] | None:
    """Connected gateway platforms, or None if undetectable (lazy + guarded: the memory
    plugin must not hard-depend on the gateway package)."""
    try:
        from gateway.config import load_gateway_config
        return [p.value for p in load_gateway_config().get_connected_platforms()]
    except Exception:
        return None


def _collect_operator_aliases(existing: dict, peer_target: str) -> dict:
    """Prompt for the operator's per-platform runtime IDs, aliasing each to ``peer_target``."""
    aliases = dict(existing)
    print(f"\n  Add runtime IDs that should alias to peer '{peer_target}'.\n"
          "  Leave blank to skip a platform.  Existing aliases are preserved.")
    for platform_label, alias_hint in (("Telegram UID", "e.g. 7654321"), ("Discord snowflake", "e.g. 491827364"),
                                       ("Slack user ID", "e.g. U04ABCDEF"), ("Matrix MXID", "e.g. @you:matrix.org")):
        entered = _prompt(f"  {platform_label} ({alias_hint})", default="").strip()
        if entered:
            aliases[entered] = peer_target
    return aliases


def _apply_runtime_prefix(hermes_host: dict, current_prefix: str, prefix_from_root: bool, label: str) -> None:
    """Write a host-level runtimePeerPrefix only when it diverges from an
    inherited root value; otherwise let the root cascade stand."""
    new_prefix = _prompt(label, default=current_prefix or "").strip()
    if new_prefix and not (prefix_from_root and new_prefix == current_prefix):
        hermes_host["runtimePeerPrefix"] = new_prefix


def _echo_identity_mapping(hermes_host: dict) -> None:
    print(f"  resolved →\n    pinUserPeer       = {bool(hermes_host.get('pinUserPeer'))}\n"
          f"    userPeerAliases   = {hermes_host.get('userPeerAliases') or '{}'}\n"
          f"    runtimePeerPrefix = {hermes_host.get('runtimePeerPrefix') or '(none)'}")


def _configure_raw_identity_mapping(hermes_host, current_pin, current_aliases, current_prefix,
                                    aliases_from_root, prefix_from_root) -> None:
    """Power-user escape hatch: set the three resolver knobs directly."""
    print("\n  Raw identity-mapping keys (resolver tries them top-down):")
    pin_in = _prompt("pinUserPeer — pin all gateway users to your peer? (true/false)",
                     default=str(bool(current_pin)).lower()).strip().lower()
    pin = pin_in in {"true", "t", "yes", "y", "1"}
    _scrub_identity_mapping(hermes_host)
    hermes_host["pinUserPeer"] = pin
    if pin:
        return
    aliases = dict(current_aliases) if isinstance(current_aliases, dict) and not aliases_from_root else {}
    print("  userPeerAliases — 'runtime_id=peer' pairs (blank line to finish):")
    while entry := _prompt("    alias", default="").strip():
        rid, _, peer = (p.strip() for p in entry.partition("="))
        if rid and peer:
            aliases[rid] = peer
    if aliases:
        hermes_host["userPeerAliases"] = aliases
    _apply_runtime_prefix(hermes_host, current_prefix, prefix_from_root,
                          "runtimePeerPrefix — namespace for unknown IDs (blank for none)")


def _setup_identity_mapping(cfg: dict, hermes_host: dict, current_peer: str, new_host: bool) -> None:
    """Gateway identity mapping step. Only the gateway supplies a runtime user ID (CLI/TUI/
    desktop fall through to peerName), so the step is gated on gateway detection."""
    current_pin, current_aliases, current_prefix, aliases_from_root, prefix_from_root = (
        _resolve_effective_identity_mapping(cfg, hermes_host))
    current_shape = "single" if current_pin else "hybrid" if current_aliases else "multi"

    gw_platforms = _gateway_platforms()
    if gw_platforms:
        print(f"\n  Gateway platforms detected: {', '.join(gw_platforms)}")
    else:
        notice, question = (
            ("\n  Each gateway account (a Telegram user, a Discord user, ...)\n"
             "  resolves to a peer. Honcho builds one representation per peer.",
             "Running the Hermes gateway (Telegram/Discord/etc.)? (y/N)") if gw_platforms is None else
            ("\n  No gateway platforms connected — nothing to map.", "Configure anyway? (y/N)"))
        print(notice)
        if not _yes(_prompt(question, default="n")):
            return

    peer_target = hermes_host.get("peerName") or current_peer or "user"
    ai_peer_label = hermes_host.get("aiPeer") or cfg.get("aiPeer") or "hermes"
    # Fresh configs default to the personal shape; configured ones keep their detected shape.
    identity_configured = not new_host or any(k in cfg for k in _IDENTITY_MAPPING_KEYS)
    default_choice = {"single": "1", "hybrid": "2", "multi": "3"}[current_shape] if identity_configured else "1"
    print("\n  This step covers the HUMAN mapping only. Each account using the\n"
          "  gateway resolves to a peer — the entity Honcho reasons about over\n"
          f"  time. This agent is already its own peer ('{ai_peer_label}'), and each\n"
          "  Hermes profile brings its own AI peer to the gateway.\n"
          "\n  How should accounts resolve?\n"
          "    [1] single peer — one person uses this agent; every account\n"
          f"        resolves to '{peer_target}'. The common personal setup.\n"
          "        Never for a gateway serving other people — their memory\n"
          "        would merge into yours\n"
          "    [2] your peer + one per other account — your accounts are\n"
          f"        aliased to '{peer_target}'; each other account gets its own\n"
          "        peer until you alias it. For a gateway you share\n"
          "    [3] one peer per account — no aliases; every account is its\n"
          "        own peer. For agents serving other people\n"
          "    [s] skip (leave untouched)   [e] edit raw keys\n"
          f"\n  Tip: alias your Telegram UID and your Discord ID to '{peer_target}' —\n"
          "  both accounts then resolve to one peer.")
    choice = _prompt("Choice", default=default_choice).strip().lower()

    if choice in {"2", "me+others", "both"}:
        pooled = _prompt("  Resolve all YOUR accounts to one peer? (Y/n)", default="y").strip().lower()
        shape = "hybrid" if pooled in {"y", "yes", ""} else "multi"
    else:
        shape = _SHAPE_CHOICES.get(choice, "skip")

    # Un-pinning without aliasing strands the pooled peerName history; steer toward pooling.
    if current_pin and shape == "multi":
        print(f"\n  ⚠ The peer '{peer_target}' already has a representation built\n"
              f"    from your messages. One peer per account means your accounts\n"
              f"    resolve to new peers with no history.")
        if _prompt(f"  Keep your accounts resolving to '{peer_target}' instead? (Y/n)", default="y").strip().lower() in {"y", "yes", ""}:
            shape = "hybrid"

    if shape == "skip":
        return print("  Identity mapping left untouched.")
    if shape == "raw":
        _configure_raw_identity_mapping(hermes_host, current_pin, current_aliases, current_prefix,
                                        aliases_from_root, prefix_from_root)
    else:
        # Preserve operator-curated host-level aliases across multi → multi re-runs. Root-sourced
        # aliases cascade naturally and are NOT copied down — an empty host map would mask a root baseline.
        prior_aliases = dict(current_aliases) if isinstance(current_aliases, dict) else {}
        if shape == "multi" and aliases_from_root:
            prior_aliases = {}
        _scrub_identity_mapping(hermes_host)  # each shape starts from a clean slate
        hermes_host["pinUserPeer"] = shape == "single"
        if shape == "single":
            print(f"  Every gateway account resolves to peer '{peer_target}'.")
        else:
            aliases = prior_aliases if shape == "multi" else _collect_operator_aliases(prior_aliases, peer_target)
            if aliases:
                hermes_host["userPeerAliases"] = aliases
            _apply_runtime_prefix(hermes_host, current_prefix, prefix_from_root,
                                  "Runtime peer prefix (e.g. 'telegram_', blank for none)" if shape == "multi" else
                                  "Runtime peer prefix for unknown users (e.g. 'telegram_', blank for none)")
            print("  Each gateway account resolves to its own peer." if shape == "multi" else
                  f"  Your accounts resolve to '{peer_target}'; each other account to its own peer.")
    _echo_identity_mapping(hermes_host)


# ── setup wizard ───────────────────────────────────────────────────────────

def _ensure_sdk_installed() -> bool:
    """Check honcho-ai is importable; offer to install if not. Returns True if ready."""
    try:
        import honcho  # noqa: F401
        return True
    except ImportError:
        pass
    print("  honcho-ai is not installed.")
    if not _yes(_prompt("Install it now? (honcho-ai>=2.5.0,<3)", default="y")):
        print("  Skipping install. Run: pip install 'honcho-ai>=2.5.0,<3'\n")
        return False
    print("  Installing honcho-ai...", flush=True)
    from tools.lazy_deps import install_specs  # env-aware: sealed hosted venvs redirect to the data volume
    result = install_specs(["honcho-ai>=2.5.0,<3"])
    if result.ok:
        print("  Installed.\n")
        return True
    print(f"  Cannot install: {result.reason}\n" if result.blocked else
          f"  Install failed:\n{(result.stderr or '').strip()}\n  Run manually: uv pip install 'honcho-ai>=2.5.0,<3'\n")
    return False


def _device_login_available() -> bool:
    """Whether the resolved host offers the RFC 8628 device grant. Fails closed."""
    try:
        from .oauth_flow import resolve_endpoints, supports_device_login
        return supports_device_login(resolve_endpoints())
    except Exception:
        return False


def _headless() -> tuple[bool, bool]:
    """(is_remote, can_open_browser) — degrades safely if hermes_cli internals move."""
    try:
        from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session
        return _is_remote_session(), _can_open_graphical_browser()
    except Exception:
        return False, True


def _apply_grant_to_host(cfg: dict, hermes_host: dict, cred) -> None:
    """Store an OAuth grant on the host block and in ``cfg``'s snapshot. install_grant already wrote it to disk,
    so the final save must not copy it over a rotation that lands during the later prompts."""
    hermes_host["apiKey"] = cred.access_token
    hermes_host["oauth"] = cred.oauth_block()
    if (snapshot := getattr(cfg, "snapshot", None)) is not None:
        snapshot.setdefault("hosts", {}).setdefault(_host_key(), {}).update(apiKey=cred.access_token, oauth=cred.oauth_block())
    if cred.consent_peer_name:  # default the peer prompt to the consent name
        hermes_host["peerName"] = cred.consent_peer_name
    print("  Authorized — token saved. Let's finish configuring.\n")


def _setup_local_auth(cfg: dict, hermes_host: dict) -> None:
    """Self-hosted Honcho may run with AUTH_USE_AUTH=true; clients then send a JWT signed with
    the server's AUTH_JWT_SECRET as the bearer token, stored as the block's ``apiKey``. A key the
    block held for another server is dropped unless one is entered here."""
    from .host_config import same_server
    old_url = hermes_host.get("baseUrl")
    if new_url := _prompt("Base URL", default=old_url or "http://localhost:8000"):
        hermes_host["baseUrl"] = new_url
    if not same_server(hermes_host.get("baseUrl"), old_url):
        hermes_host.pop("apiKey", None)
        hermes_host.pop("oauth", None)
    current_host_key = hermes_host.get("apiKey", "")
    print("\n  Local Honcho auth (JWT signed with the server's AUTH_JWT_SECRET).\n"
          f"  Leave blank if your server runs with AUTH_USE_AUTH=false. Current: {_mask(current_host_key)}")
    new_local_key = _prompt("Local JWT / bearer token (blank to skip / keep current)", secret=True)
    if new_local_key:
        hermes_host["apiKey"] = new_local_key
    elif current_host_key:
        print("  Keeping existing local JWT.")
    else:
        print("\n  No local JWT set. Local no-auth ready.")


def _setup_device_login(cfg: dict, hermes_host: dict, write_path: Path, *, open_browser: bool) -> bool:
    """RFC 8628 device-code sign-in. Returns False if setup must abort."""
    from .oauth_flow import (
        AccessDenied, AuthorizationTimeout, DeviceCode, DeviceCodeExpired, DeviceFlowError, authorize_via_device_code,
    )

    def _show(device: DeviceCode) -> None:
        print(f"\n  To connect, on any device with a browser:\n\n    1. Open   {device.verification_uri}\n"
              f"    2. Enter  {device.user_code}\n\n  Or open directly:\n\n    {device.verification_uri_complete}\n")
        mins = max(1, device.expires_in // 60)
        print(f"  Waiting for approval (expires in {mins} min, Ctrl-C to cancel) ", end="", flush=True)

    print("\n  Requesting device code…")
    import webbrowser
    try:
        cred = authorize_via_device_code(
            config_path=write_path, host=_host_key(), source="hermes-cli", apply_config=False, display=_show,
            open_url=webbrowser.open if open_browser else None, on_poll=lambda: print(".", end="", flush=True),
        )
    except KeyboardInterrupt:
        print("\n  Cancelled. Re-run 'hermes honcho setup' to try again.\n")
    except (AuthorizationTimeout, DeviceCodeExpired):
        print("\n  Device code expired before approval.\n  Re-run 'hermes honcho setup' to get a new code.\n")
    except AccessDenied:
        print("\n  Sign-in was denied on the approval page.\n" + _RETRY_HINT)
    except Exception as e:
        print("\n  Too many device-code requests — wait a minute and re-run setup.\n"
              if isinstance(e, DeviceFlowError) and e.error == "http_429" else f"\n  Device sign-in failed: {e}\n" + _RETRY_HINT)
    else:
        print(" approved")
        _apply_grant_to_host(cfg, hermes_host, cred)
        return True
    return False


def _setup_browser_login(cfg: dict, hermes_host: dict, write_path: Path) -> bool:
    """Loopback OAuth sign-in. Tokens merge into the in-memory cfg so the wizard's final save
    keeps them; settings stay wizard-owned (apply_config=False). Returns False on abort."""
    from .oauth_flow import authorize_via_loopback
    import webbrowser

    def _open(url: str) -> None:
        print(f"\n  Open this link to authorize (waiting up to 5 minutes):\n\n    {url}\n")
        webbrowser.open(url)

    print("\n  Starting browser sign-in…")
    try:
        cred = authorize_via_loopback(config_path=write_path, host=_host_key(), source="hermes-cli",
                                      apply_config=False, open_url=_open)
    except Exception as e:
        print(f"  OAuth sign-in failed: {e}\n" + _RETRY_HINT)
        return False
    _apply_grant_to_host(cfg, hermes_host, cred)
    return True


def _setup_cloud_auth(cfg: dict, hermes_host: dict, write_path: Path) -> bool:
    """Cloud auth: OAuth (browser), device code, or API key. Returns False on abort. A key the block
    held for a self-hosted server is dropped."""
    from .host_config import same_server
    if not same_server(hermes_host.pop("baseUrl", None), None):  # cloud uses the SDK default
        hermes_host.pop("apiKey", None)
        hermes_host.pop("oauth", None)
    from .oauth import OAuthCredential, is_oauth_access_token
    existing_oauth = OAuthCredential.from_host_block(hermes_host)
    device_available = _device_login_available()
    is_remote, can_browse = _headless()

    print("\n  Auth method:")
    if existing_oauth is not None:
        print(f"    (currently connected via OAuth — client {existing_oauth.client_id})")
    print("    oauth  -- sign in via browser on this machine (recommended)")
    if device_available:
        print("    device -- device code: approve from a browser on another machine (SSH / headless)")
    print(f"    apikey -- paste an API key from {attribution_url(HONCHO_APP_URL, 'apikey-prompt')}")

    default_method = "oauth"
    if is_remote or not can_browse:
        default_method = "device" if device_available else "oauth"
        print("  (no usable local browser detected — device code recommended)" if device_available else
              "  (no usable local browser detected — browser sign-in may need an SSH tunnel to 127.0.0.1:8765)")
    method = _prompt("oauth, device, or apikey?" if device_available else "OAuth or API key?",
                     default=default_method).strip().lower()

    if device_available and method in {"device", "d"}:
        return _setup_device_login(cfg, hermes_host, write_path, open_browser=can_browse and not is_remote)
    if method in {"oauth", "o"}:
        return _setup_browser_login(cfg, hermes_host, write_path)
    # A leftover grant on the host block would shadow the pasted key.
    stale_grant = existing_oauth is not None or is_oauth_access_token(hermes_host.get("apiKey"))
    current = "" if stale_grant else hermes_host.get("apiKey", "")
    print(f"\n  Current API key: {_mask(current)}")
    new_key = _prompt("Honcho API key (leave blank to keep current)", secret=True)
    key = new_key or current
    if not key:
        print(f"\n  No API key configured. Get yours at {attribution_url(HONCHO_APP_URL, 'missing-key')}\n"
              "  Run 'hermes honcho setup' again once you have a key.\n")
        return False
    hermes_host.pop("oauth", None)
    hermes_host["apiKey"] = key
    return True


def cmd_setup(args) -> None:
    """Interactive Honcho setup wizard."""
    try:
        _setup_wizard(args)
    except ConfigWriteRefused as e:
        print(f"  {e}\n")


def _setup_wizard(args) -> None:
    from .host_config import legacy_path
    cfg = _read_config()
    write_path = _config_path()
    _refuse_unparseable(write_path)  # before the questions, not after them
    print(f"\nHoncho memory setup\n{RULE}\n  Honcho gives Hermes persistent cross-session memory.\n  Config: {write_path}")
    if (read_path := legacy_path(write_path)) not in {None, write_path}:
        print(f"  (seeding from existing config at {read_path})")
    print()
    if not _ensure_sdk_installed():
        return

    hermes_host = cfg.setdefault("hosts", {}).setdefault(_host_key(), {})
    _migrate_pin_key(cfg)  # canonicalize legacy pinPeerName before detection/writes
    _migrate_pin_key(hermes_host)
    # Taken before the prompts populate the block: an existing install must not default to pinning every account.
    new_host = not any(k in hermes_host or k in cfg for k in (*_IDENTITY_MAPPING_KEYS, "peerName", "workspace", "enabled"))

    # --- 1. Cloud or local? ---
    print("  Deployment:\n    cloud -- Honcho cloud (api.honcho.dev)\n    local -- self-hosted Honcho server")
    current_url = hermes_host.get("baseUrl") or ""
    current_deploy = "local" if any(h in current_url for h in ("localhost", "127.0.0.1", "::1")) else "cloud"
    is_local = _prompt("Cloud or local?", default=current_deploy).lower() in {"local", "l"}
    if is_local:
        _setup_local_auth(cfg, hermes_host)
    elif not _setup_cloud_auth(cfg, hermes_host, write_path):
        return

    # --- 3. Identity ---
    current_peer = hermes_host.get("peerName") or cfg.get("peerName", "")
    for key, label, default in (
        ("peerName", "Your name (user peer)", current_peer or os.getenv("USER", "user")),
        ("aiPeer", "AI peer name", _pref(hermes_host, cfg, "aiPeer", "hermes")),
        ("workspace", "Workspace ID", _pref(hermes_host, cfg, "workspace", "hermes")),
    ):
        if new := _prompt(label, default=default):
            hermes_host[key] = new

    _setup_identity_mapping(cfg, hermes_host, current_peer, new_host)
    print("\n  For a gateway with many users and agents, run\n"
          "  'hermes honcho peers map' to map accounts interactively.")

    hermes_host["enabled"] = True
    _write_config(cfg)
    print(f"\n  Config written to {write_path}")

    try:  # auto-enable Honcho as memory provider in config.yaml
        from hermes_cli.config import load_config, save_config
        hermes_config = load_config()
        hermes_config.setdefault("memory", {})["provider"] = "honcho"
        save_config(hermes_config)
        print("  Memory provider set to 'honcho' in config.yaml")
    except Exception as e:
        print(f"  Could not auto-enable in config.yaml: {e}\n  Run: hermes config set memory.provider honcho")

    print("  Testing connection... ", end="", flush=True)
    try:
        hcfg, _client = _connect(_host_key(), reset=True)
        print("OK")
    except Exception as e:
        return print(f"FAILED\n  Error: {e}")

    print(f"""
  Honcho is ready.
  Workspace: {hcfg.workspace_id}
  User:      {hcfg.peer_name}
  AI peer:   {hcfg.ai_peer}
  Sessions:  one per chat, in scopes {', '.join(session_scopes(os.getcwd())) or '(none)'}

  Honcho tools available in chat:
    honcho_chat      -- ask Honcho a question, synthesized answer
    honcho_search    -- semantic search over history
    honcho_profile   -- peer card, key facts
    honcho_conclude  -- persist a user fact to memory

  Other commands:
    hermes honcho status     -- show full config
    hermes honcho peer       -- update peer names
""")


# ── status / peers ─────────────────────────────────────────────────────────

def _active_profile_name() -> str:
    """Active Hermes profile name (respects --target-profile override)."""
    if _profile_override:
        return _profile_override
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "default"


def _all_profile_host_configs() -> list[tuple[str, str, dict]]:
    """(profile_name, host_key, host_block) for every known profile, each block from its own honcho.json."""
    try:
        from hermes_cli.profiles import list_profiles
        profiles = list_profiles()
    except Exception:
        return [(_active_profile_name(), _host_key(), {})]
    root = _get_default_hermes_home()
    return [("default", HOST, _default_block())] + [
        (p.name, host, _profile_block(Path(getattr(p, "path", None) or root / "profiles" / p.name) / "honcho.json", host))
        for p in profiles if p.name != "default" for host in (profile_host_key(p.name),)
    ]


def cmd_status(args) -> None:
    """Show current Honcho config and connection status."""
    if getattr(args, "all", False):
        _cmd_status_all()
        return
    try:
        import honcho  # noqa: F401
    except ImportError:
        print("  honcho-ai is not installed. Run: hermes honcho setup\n")
        return

    from .host_config import is_marked, legacy_path
    cfg = _read_config()
    active_path = _config_path()
    from .client import HonchoClientConfig, get_honcho_client
    not_found = f"  No Honcho config found at {active_path}\n  Run 'hermes honcho setup' to configure.\n"
    try:
        hcfg = HonchoClientConfig.from_global_config(host=_host_key(), config_path=active_path)
    except Exception as e:
        return print(not_found if not cfg else f"  Config error: {e}\n")
    if not cfg and not (hcfg.api_key or hcfg.base_url):  # file missing and no env-var fallback either
        return print(not_found)

    # The OAuth access token is also stored under apiKey, so the auth line
    # distinguishes a refreshable grant from a static key explicitly.
    from .oauth import OAuthCredential
    raw = getattr(hcfg, "raw", None) or {}
    cred = OAuthCredential.from_host_block(raw.get("hosts", {}).get(hcfg.host) or {})
    profile = _active_profile_name()

    if cred is not None:
        import time as _time
        remaining = int(cred.expires_at - _time.time())
        token_state = f"valid {remaining // 60}m" if remaining > 0 else "expired — refreshes on next use"
        auth = f"OAuth ({cred.client_id}, token {token_state})"
    else:
        auth = f"API key ({_mask(hcfg.api_key or '')})"
    print(f"\nHoncho status{f' [{hcfg.host}]' if profile != 'default' else ''}\n" + RULE
          + (f"\n  Profile:        {profile}" if profile != "default" else ""))
    print(f"  Host:           {hcfg.host}\n  Enabled:        {hcfg.enabled}\n  Auth:           {auth}\n"
          f"  Workspace:      {hcfg.workspace_id}\n  Config:         {active_path} (hosts.{hcfg.host})")
    stored = (read_json_or_empty(active_path).get("hosts") or {}).get(hcfg.host)
    if not is_marked(stored) and (source := legacy_path(active_path)) is not None:
        print(f"  Migrating from: {source}  (copied into the host block on next start or save)")

    print(f"""  AI peer:        {hcfg.ai_peer}
  User peer:      {hcfg.peer_name or 'not set'}
  Sessions:       one per chat; scopes here: {', '.join(session_scopes(os.getcwd())) or '(none)'}""")

    if not (hcfg.enabled and (hcfg.api_key or hcfg.base_url)):
        return print(f"\n  Not connected ({'disabled' if not hcfg.enabled else 'no API key or base URL'})\n")
    print("\n  Connection... ", end="", flush=True)
    try:
        _show_peer_cards(hcfg, get_honcho_client(hcfg))
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})\n")


def _show_peer_cards(hcfg, client) -> None:
    """Fetch and display peer cards for the active profile."""
    try:
        mgr, session_key = _session_manager(hcfg, client)
        card = mgr.get_peer_card(session_key)
        if card:
            print(f"\n  User peer card ({len(card)} facts):\n" + "\n".join(f"    - {fact}" for fact in card[:10]))
            if len(card) > 10:
                print(f"    ... and {len(card) - 10} more")
        ai_text = mgr.get_ai_representation(session_key).get("representation", "")
        if ai_text:
            print(f"\n  AI peer representation:\n    {ai_text[:200] + ('...' if len(ai_text) > 200 else '')}")
        print("" if card or ai_text else "\n  No peer data yet (accumulates after first conversation)\n")
    except Exception as e:
        print(f"\n  Peer data unavailable: {e}\n")


def _cmd_status_all() -> None:
    """Show Honcho config overview across all profiles."""
    rows = _all_profile_host_configs()
    active = _active_profile_name()
    print(f"\nHoncho profiles ({len(rows)})\n{'─' * 55}\n"
          f"  {'Profile':<14} {'Host':<22} {'Enabled':<9} {'Recall':<9} {'Write'}\n"
          f"  {'─' * 14} {'─' * 22} {'─' * 9} {'─' * 9} {'─' * 9}")
    for name, host, block in rows:
        enabled = block.get("enabled")
        if enabled is None:  # auto-enabled by a key or server, as the client resolves it
            enabled = bool(_resolve_api_key({}, block)) if block else False
        marker = " *" if name == active else ""
        print(f"  {name + marker:<14} {host:<22} {'yes' if enabled else 'no':<9} "
              f"{block.get('recallMode') or 'hybrid':<9} {block.get('writeFrequency') or 'async'}")
    print("\n  * active profile\n")


def _state_db_path() -> Path:
    """Return the state.db path for the targeted profile."""
    if _profile_override and _profile_override not in {"default", "custom"}:
        try:
            from hermes_cli.profiles import get_profile_dir
            return get_profile_dir(_profile_override) / "state.db"
        except Exception:
            pass
    return get_hermes_home() / "state.db"


def _seen_gateway_accounts(db_path: Path) -> list[dict]:
    """Gateway accounts recorded in state.db, most recent first.

    A session row keeps only its last routing peer, so a shared thread contributes its most recent
    author and not every participant. The row's origin does not record whether the author was a bot.
    """
    if not db_path.exists():
        return []
    import sqlite3
    from contextlib import closing
    # profile_name marks which profile a multiplexing gateway routed the
    # session to; older state.db files predate the column.
    query = """SELECT source, user_id,
                      MAX(COALESCE(display_name, '')),
                      MAX(COALESCE(origin_json, '')),
                      COUNT(*){profiles_col}
                 FROM sessions
                WHERE user_id IS NOT NULL AND user_id != ''
                GROUP BY source, user_id
                ORDER BY MAX(COALESCE(started_at, 0)) DESC"""
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            try:
                rows = conn.execute(query.format(
                    profiles_col=", GROUP_CONCAT(DISTINCT COALESCE(profile_name, 'default'))",
                )).fetchall()
            except sqlite3.OperationalError:
                rows = [r + (None,) for r in conn.execute(query.format(profiles_col="")).fetchall()]
    except sqlite3.Error as e:
        print(f"  (state.db unreadable: {e}; the accounts list is unavailable)", file=sys.stderr)
        return []

    accounts = []
    for source, user_id, display_name, origin_json, count, profiles in rows:
        try:
            origin = dict(json.loads(origin_json)) if origin_json else {}
        except Exception:
            origin = {}
        accounts.append({
            "platform": source or "?",
            "user_id": str(user_id),
            "user_id_alt": str(origin.get("user_id_alt") or ""),
            "label": origin.get("user_name") or display_name or "",
            "sessions": count,
            "profiles": sorted(profiles.split(",")) if profiles else [],
        })
    return accounts


def _preview_peer_resolution(
    user_id: str, *, pin: bool, aliases: dict, prefix: str, peer_name: str,
    user_id_alt: str = "",
) -> str:
    """Resolve through the runtime resolver and label the rung that decided: pin, alias, prefix, raw.
    Only the runtime knows when a prefixed id gets a hash suffix, so the CLI must not recompute it."""
    from .client import HonchoClientConfig
    from .session import HonchoSessionManager

    config = HonchoClientConfig(peer_name=peer_name or None, pin_peer_name=bool(pin),
                                user_peer_aliases=aliases, runtime_peer_prefix=prefix)
    manager = HonchoSessionManager(config=config, runtime_user_peer_name=user_id,
                                   runtime_user_peer_name_alt=user_id_alt or None)
    resolved = manager._resolve_user_peer_id("preview")
    if pin and peer_name:
        return f"{resolved} (pinned)"
    # The runtime resolver tries the alt ID (Signal UUID, Feishu union_id) after the primary.
    aliased = any(isinstance(aliases.get(rid), str) and aliases[rid].strip() for rid in (user_id, user_id_alt) if rid)
    if not aliased and prefix.strip():
        return f"{resolved} (prefixed)"
    return resolved


def _resolution_base(resolved: str) -> str:
    """Strip display suffixes so the name can be checked against peer IDs."""
    return resolved.removesuffix(" (pinned)").removesuffix(" (prefixed)")


# Workspaces holding thousands of peers (public bots) must not stall the CLI.
_PEERS_MAP_FETCH_CAP = 200


def _peers_map_client(workspace: str | None = None):
    """(client, config) for the active host, or (None, None) offline. ``workspace`` overrides the configured one."""
    try:
        from dataclasses import replace
        from .client import HonchoClientConfig, get_honcho_client
        hcfg = HonchoClientConfig.from_global_config(host=_host_key())
        if not (hcfg.api_key or hcfg.base_url):
            return None, None
        if workspace and workspace != hcfg.workspace_id:
            hcfg = replace(hcfg, workspace_id=workspace)
        return get_honcho_client(hcfg), hcfg
    except Exception:
        return None, None


def _api_workspace_peers(client) -> list[str] | None:
    """Workspace peer IDs, at most _PEERS_MAP_FETCH_CAP. None = API unavailable."""
    if client is None:
        return None
    try:
        peers: list[str] = []
        page = client.peers(page=1, size=50)
        # Iterating a SyncPage walks every following page; .items is the one page asked for.
        while True:
            peers += [str(p.id) for p in page.items]
            if len(peers) >= _PEERS_MAP_FETCH_CAP or not page.has_next_page():
                return peers[:_PEERS_MAP_FETCH_CAP]
            page = page.get_next_page()
    except Exception:
        return None


def _api_workspaces(client) -> list[str] | None:
    """Workspace IDs this key can reach. None = API unavailable."""
    if client is None:
        return None
    try:
        return [str(w) for w in client.workspaces(size=50).items]
    except Exception:
        return None


def _api_peer_detail(client, peer_id: str) -> str:
    """Short peek at a peer: its card, or a clear absence note."""
    try:
        card = client.peer(peer_id).get_card()
        if card:
            text = str(card).strip()
            return text[:400] + ("…" if len(text) > 400 else "")
        return "(no peer card yet)"
    except Exception as e:
        return f"(peer detail unavailable: {e})"


def _classify_workspace_peers(
    peer_ids: list[str], cfg: dict, accounts: list[dict],
    aliases: dict, prefix: str, profile_rows: list[tuple[str, str, dict]],
) -> dict[str, str]:
    """Label workspace peers from local config; 'unrecognized' when honest."""
    labels: dict[str, str] = {}
    active_host = _host_key()
    root_peer = cfg.get("peerName") or ""

    hermes_hosts = {hostk for _, hostk, _ in profile_rows}
    for name, hostk, block in profile_rows:
        pn = block.get("peerName") or root_peer
        ai = block.get("aiPeer") or cfg.get("aiPeer") or hostk
        if pn:
            labels.setdefault(
                sanitize_peer_id(pn),
                "your peer (peerName)" if hostk == active_host
                else f"peerName of profile {name}",
            )
        who = "this profile" if hostk == active_host else f"profile {name}"
        labels.setdefault(sanitize_peer_id(ai), f"AI peer · {who}")

    # Host blocks that are not Hermes profiles: other apps sharing the config.
    for hostk, block in (cfg.get("hosts") or {}).items():
        if hostk in hermes_hosts or not isinstance(block, dict):
            continue
        for key, kind in (("peerName", "peer"), ("aiPeer", "AI peer")):
            val = block.get(key)
            if isinstance(val, str) and val.strip():
                labels.setdefault(sanitize_peer_id(val.strip()), f"{kind} of app '{hostk}'")

    for target in aliases.values():
        if isinstance(target, str) and target.strip():
            labels.setdefault(sanitize_peer_id(target.strip()), "alias target")

    for acct in accounts:
        rid = acct["user_id"]
        for candidate in ([rid, prefix + rid] if prefix else [rid]):
            labels.setdefault(sanitize_peer_id(candidate), f"runtime peer · {acct['platform']} {rid}")

    return {
        pid: labels.get(pid) or ("fallback peer (pre-identity traffic)" if pid.startswith("user-") else "unrecognized")
        for pid in peer_ids
    }


def _sibling_resolutions(cfg: dict, acct: dict, profile_rows: list[tuple[str, str, dict]]) -> dict[str, str]:
    """Resolved peer per profile for one account (profile name → peer)."""
    out = {}
    for name, _hostk, block in profile_rows:
        pin, aliases, prefix, _, _ = _resolve_effective_identity_mapping(cfg, block)
        out[name] = _resolution_base(_preview_peer_resolution(
            acct["user_id"], pin=pin, aliases=aliases, prefix=prefix,
            peer_name=block.get("peerName") or cfg.get("peerName") or "",
            user_id_alt=acct["user_id_alt"],
        ))
    return out


def _render_peers_map_view(
    workspace: str, ws_peers: list[str] | None, labels: dict,
    accounts: list[dict], cfg: dict, profile_rows: list[tuple[str, str, dict]], *,
    pin: bool, working: dict, prefix: str, peer_name: str,
) -> None:
    if ws_peers is None:
        print(f"\nWorkspace '{workspace}' — peers unavailable (offline or not configured)")
        print("  Mapping still works; target peers are typed instead of picked.")
    else:
        print(f"\nWorkspace '{workspace}' — {len(ws_peers)} peers\n" + "─" * 62)
        if not ws_peers:
            print("  No peers here yet — peers appear after the first conversation.")
            print("  Wrong workspace? 'w' lists the workspaces this key can see.")
        for i, pid in enumerate(ws_peers, 1):
            print(f"  p{i:<4} {pid:<30} {labels.get(pid, '')}")
        if len(ws_peers) >= _PEERS_MAP_FETCH_CAP:
            print(f"  … listing capped at {_PEERS_MAP_FETCH_CAP} peers.")
        if ws_peers and not any(v.startswith(("your peer", "AI peer")) for v in labels.values()):
            print(f"\n  None of these match your configured identity ('{peer_name or workspace}').")
            print("  Wrong workspace? 'w' lists the workspaces this key can see.")

    active_profile = _active_profile_name()
    print(f"\nGateway accounts seen on this machine ({len(accounts)})\n" + "─" * 62)
    if not accounts:
        print("  None recorded yet. Accounts appear here after the gateway")
        print("  handles a message from them. You can still map a runtime ID")
        print("  by typing it at the prompt below.")
        return
    print(f"  {'#':<4} {'Platform':<10} {'Runtime ID':<22} {'Name':<14} {'Resolves to'}")
    known = set(ws_peers or [])
    for idx, acct in enumerate(accounts, 1):
        resolved = _preview_peer_resolution(
            acct["user_id"], pin=pin, aliases=working, prefix=prefix,
            peer_name=peer_name, user_id_alt=acct["user_id_alt"],
        )
        mine = _resolution_base(resolved)
        marker = "" if ws_peers is None else (" ✓" if mine in known else " ○ new")
        diverging = {
            n: v for n, v in _sibling_resolutions(cfg, acct, profile_rows).items()
            if n != active_profile and v != mine
        }
        div = "  ≠ " + ", ".join(f"{n}→{v}" for n, v in sorted(diverging.items())) if diverging else ""
        profiles = acct.get("profiles") or []
        via = f"  (traffic → {', '.join(profiles)})" if profiles and active_profile not in profiles else ""
        print(
            f"  {idx:<4} {acct['platform']:<10} {acct['user_id']:<22} "
            f"{acct['label'][:13]:<14} {resolved}{marker}{div}{via}"
        )


def _workspaces_flow(client, current_ws: str, cfg: dict, host: str):
    """List reachable workspaces; browse one; optionally repoint the profile.

    Returns (workspace, client, ws_peers) after a confirmed switch, else None.
    """
    ws_list = _api_workspaces(client)
    if not ws_list:
        print("  Workspace list unavailable (offline or not configured).")
        return None
    print(f"\n  Workspaces this key can see ({len(ws_list)}):")
    for i, w in enumerate(ws_list, 1):
        print(f"    {i:<4} {w}{'  ← current' if w == current_ws else ''}")
    print("\n  Tip: for a full workspace browser, install honcho-cli")
    print("  (uv tool install honcho-cli).")
    sel = _prompt("Browse a workspace (number, blank to go back)", default="").strip()
    if not (sel.isdigit() and 1 <= int(sel) <= len(ws_list)):
        return None
    target_ws = ws_list[int(sel) - 1]
    b_client, _ = _peers_map_client(workspace=target_ws)
    b_peers = _api_workspace_peers(b_client)
    if b_peers is None:
        print(f"  Could not list peers of '{target_ws}'.")
        return None
    print(f"\n  Workspace '{target_ws}' — {len(b_peers)} peers")
    for pid in b_peers[:30]:
        print(f"    {pid}")
    if len(b_peers) > 30:
        print(f"    … and {len(b_peers) - 30} more")
    if target_ws == current_ws:
        return None
    if not _yes(_prompt(
        f"Point this profile at '{target_ws}'? Existing memory stays in '{current_ws}'. (y/N)",
        default="n",
    )):
        return None
    cfg.setdefault("hosts", {}).setdefault(host, {})["workspace"] = target_ws
    _write_config(cfg)
    print(f"  workspace → '{target_ws}' (written to host block [{host}])")
    return target_ws, b_client, b_peers


def _save_alias_map(cfg: dict, host: str, working: dict, aliases_from_root: bool) -> None:
    """Persist the edited alias map to the active host block, the only place it is read."""
    cfg.setdefault("hosts", {}).setdefault(host, {})["userPeerAliases"] = working
    _write_config(cfg)
    print(f"\n  userPeerAliases = {working}")
    print(f"  written to host block [{host}] in {_config_path()}\n")


def cmd_peers_map(args) -> None:
    """Interactively map gateway accounts to Honcho user peers."""
    cfg = _read_config()
    host = _host_key()
    hermes_host = _host_block(cfg, host)
    pin, aliases, prefix, aliases_from_root, _ = _resolve_effective_identity_mapping(cfg, hermes_host)
    peer_name = hermes_host.get("peerName") or cfg.get("peerName") or ""

    if pin:
        print("\n  pinUserPeer is on: every gateway account resolves to peer")
        print(f"  '{peer_name or '(peerName not set)'}' and aliases have no effect.")
        print("  Turn the pin off with 'hermes honcho setup' to use per-account peers.")
        if not _yes(_prompt("Edit aliases anyway? (y/N)", default="n")):
            print("  Nothing changed.\n")
            return

    accounts = _seen_gateway_accounts(_state_db_path())
    working = dict(aliases) if isinstance(aliases, dict) else {}
    client, client_cfg = _peers_map_client()
    workspace = (
        getattr(client_cfg, "workspace_id", None)
        or hermes_host.get("workspace") or cfg.get("workspace") or host
    )
    ws_peers = _api_workspace_peers(client)
    # list_profiles() parses every profile's config.yaml; one scan serves every row and re-render.
    profile_rows = _all_profile_host_configs()

    def show() -> dict[str, str]:
        labels = _classify_workspace_peers(ws_peers or [], cfg, accounts, working, prefix, profile_rows)
        _render_peers_map_view(workspace, ws_peers, labels, accounts, cfg, profile_rows,
                               pin=pin, working=working, prefix=prefix, peer_name=peer_name)
        return labels

    labels = show()
    print("\n  Map: account number or a runtime ID · pN inspects a peer ·")
    print("  w lists workspaces · blank finishes.")
    changed = repointed = False
    while True:
        sel = _prompt("Account (blank to finish)", default="").strip()
        if not sel:
            break
        low = sel.lower()

        if low == "w":
            switched = _workspaces_flow(client, workspace, cfg, host)
            if switched:
                workspace, client, ws_peers = switched
                labels = show()
                repointed = True
            continue

        if low.startswith("p") and low[1:].isdigit() and ws_peers:
            n = int(low[1:])
            if 1 <= n <= len(ws_peers):
                pid = ws_peers[n - 1]
                print(f"\n  {pid} — {labels.get(pid, '')}")
                print(f"  {_api_peer_detail(client, pid)}\n")
            continue

        if sel.isdigit() and 1 <= int(sel) <= len(accounts):
            acct = accounts[int(sel) - 1]
            rid, alt = acct["user_id"], acct["user_id_alt"]
            label = f"{acct['platform']} {rid}" + (f" ({acct['label']})" if acct["label"] else "")
        else:
            rid, alt, label = sel, "", sel
        prev_resolved = _resolution_base(_preview_peer_resolution(
            rid, pin=pin, aliases=working, prefix=prefix, peer_name=peer_name, user_id_alt=alt,
        ))

        current = working.get(rid, "")
        hint = " (pN from the peers table, a name, '-' clears)" if ws_peers else ""
        entered = _prompt(f"Peer for {label}{hint}", default=current).strip()
        if entered == "-":
            if rid in working:
                del working[rid]
                changed = True
                print(f"    cleared: {rid}")
            continue
        if ws_peers and entered[:1].lower() == "p" and entered[1:].isdigit() and 0 < int(entered[1:]) <= len(ws_peers):
            entered = ws_peers[int(entered[1:]) - 1]
        if entered and entered != current:
            working[rid] = entered
            changed = True
            print(f"    {rid} → {entered} — future messages resolve to '{entered}'")
            if ws_peers is not None and sanitize_peer_id(entered) not in ws_peers:
                print(f"    '{entered}' is a new peer — created on first message.")
            if prev_resolved in (ws_peers or ()) and prev_resolved != sanitize_peer_id(entered):
                print(f"    peer '{prev_resolved}' keeps its existing history.")

    if not changed:
        print("  Aliases unchanged.\n" if repointed else "  Nothing changed.\n")
        return
    _save_alias_map(cfg, host, working, aliases_from_root)


def cmd_peers(args) -> None:
    """Show peer identities across all profiles."""
    if getattr(args, "peers_action", None) == "map":
        cmd_peers_map(args)
        return

    rows = _all_profile_host_configs()
    cfg = _read_config()
    print(f"\nHoncho peer identities ({len(rows)} profiles)\n{'─' * 50}\n"
          f"  {'Profile':<14} {'User peer':<16} {'AI peer'}\n  {'─' * 14} {'─' * 16} {'─' * 18}")
    for name, host, block in rows:
        user = _pref(block, cfg, "peerName") or "(not set)"
        print(f"  {name:<14} {user:<16} {_pref(block, cfg, 'aiPeer') or host}")
    print()


# ── sessions / map ─────────────────────────────────────────────────────────

# ── peer ───────────────────────────────────────────────────────────────────

def _show_or_set_fields(args, fields: tuple, show) -> None:
    """Body of ``peer``: with no flags, ``show(block, cfg)`` prints the current values; otherwise each
    ``(attr, key, echo)`` flag that is set is written."""
    cfg = _read_config()
    values = {attr: getattr(args, attr, None) for attr, _key, _echo in fields}
    if all(v is None for v in values.values()):
        return show(_active_block(cfg), cfg)
    for attr, key, echo in fields:
        if (value := values[attr]) is None:
            continue
        value = value.strip() if isinstance(value, str) else value
        _set_field(cfg, key, value, echo.format(value))
    _save(cfg)


def cmd_peer(args) -> None:
    """Show or update peer names."""
    def show(hermes, cfg):
        print(f"""
Honcho peers
{RULE}
  User peer:   {_pref(hermes, cfg, 'peerName') or '(not set)'}
    Your identity in Honcho. Messages you send build this peer's card.
  AI peer:     {_pref(hermes, cfg, 'aiPeer') or _host_key()}
    Hermes' identity in Honcho. Seed with 'hermes honcho identity <file>'.
""")
    _show_or_set_fields(args, (("user", "peerName", "User peer -> {}"), ("ai", "aiPeer", "AI peer   -> {}")), show)


# ── identity / migrate ─────────────────────────────────────────────────────

def cmd_identity(args) -> None:
    """Seed AI peer identity or show both peer representations."""
    cfg = _read_config()
    if not _resolve_api_key(cfg):
        return print("  No API key configured. Run 'hermes honcho setup' first.\n")
    file_path = getattr(args, "file", None)
    try:
        hcfg, client = _connect(_host_key())
        mgr, session_key = _session_manager(hcfg, client)
    except Exception as e:
        return print(f"  Honcho connection failed: {e}\n")

    if getattr(args, "show", False):
        from .session import HonchoAuthError
        try:
            user_card = mgr.get_peer_card(session_key)
            ai_rep = mgr.get_ai_representation(session_key)
        except HonchoAuthError as e:
            return print(f"  Honcho authentication failed: {e}\n")
        print(f"\nUser peer ({hcfg.peer_name or 'not set'})\n" + RULE)
        print("\n".join(f"  {fact}" for fact in user_card) if user_card
              else "  No user peer card yet. Send a few messages to build one.")
        print(f"\nAI peer ({hcfg.ai_peer})\n" + RULE)
        print(ai_rep.get("representation") or ai_rep.get("card")
              or "  No representation built yet.\n  Run 'hermes honcho identity <file>' to seed one.")
        print()
        return

    if not file_path:
        print(f"""
Honcho identity management
{RULE}
  User peer: {hcfg.peer_name or 'not set'}
  AI peer:   {hcfg.ai_peer}

    hermes honcho identity --show        — show both peer representations
    hermes honcho identity <file>        — seed AI peer from SOUL.md or any .md/.txt
""")
        return

    p = Path(file_path).expanduser()
    if not p.exists():
        return print(f"  File not found: {p}\n")
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return print(f"  File is empty: {p}\n")
    if mgr.seed_ai_identity(session_key, content, source=p.name):
        print(f"  Seeded AI peer identity from {p.name} into session '{session_key}'\n"
              f"  Honcho will incorporate this into {hcfg.ai_peer}'s representation over time.\n")
    else:
        print("  Failed to seed identity. Check logs for details.\n")


def _find_memory_files(names: list[str]) -> list[Path]:
    """Existing files named ``names`` in cwd then ~/.openclaw, deduplicated."""
    candidates = (d / name for name in names for d in (Path(os.getcwd()), Path.home() / ".openclaw"))
    return list(dict.fromkeys(p for p in candidates if p.exists()))


def _migrate_upload(mgr, session_key: str, user_files: list[Path]) -> None:
    dirs_with_files = set(str(f.parent) for f in user_files)
    # List (not generator) so every directory is attempted even after one succeeds.
    if any([mgr.migrate_memory_files(session_key, d) for d in dirs_with_files]):
        print(f"  Uploaded user memory files from: {', '.join(dirs_with_files)}")
    else:
        print("  Nothing uploaded (files may already be migrated or empty).")


def _migrate_seed(mgr, session_key: str, agent_files: list[Path]) -> None:
    for f in agent_files:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            ok = mgr.seed_ai_identity(session_key, content, source=f.name)
            print(f"    {f.name}: {'seeded' if ok else 'failed'}")


def _offer(question: str, action, files: list[Path]) -> None:
    """Ask, then run ``action(mgr, session_key, files)`` against a fresh client."""
    if not _yes(_prompt(question, default="y")):
        return
    try:
        action(*_session_manager(*_connect(None, reset=True)), files)
    except Exception as e:
        print(f"  Failed: {e}")


def cmd_migrate(args) -> None:
    """Step-by-step migration guide: OpenClaw native memory → Hermes + Honcho."""
    user_files = _find_memory_files(["USER.md", "MEMORY.md"])  # facts about the user
    agent_files = _find_memory_files(["SOUL.md", "IDENTITY.md", "AGENTS.md", "TOOLS.md", "BOOTSTRAP.md"])
    cfg = _read_config()
    has_key = bool(_resolve_api_key(cfg))

    print("\nHoncho migration: OpenClaw native memory → Hermes\n" + "─" * 50)
    print("""
  OpenClaw's native memory stores context in local markdown files
  (USER.md, MEMORY.md, SOUL.md, ...) and injects them via QMD search.
  Honcho replaces that with a cloud-backed, LLM-observable memory layer:
  context is retrieved semantically, injected automatically each turn,
  and enriched by a dialectic reasoning layer that builds over time.

Step 1  Create a Honcho account
""")
    if has_key:
        print(f"  Honcho API key already configured: {_mask(_resolve_api_key(cfg))}\n  Skip to Step 2.")
    else:
        print(f"""  Honcho is a cloud memory service that gives Hermes persistent memory
  across sessions. You need an API key to use it.

  1. Get your API key at {attribution_url(HONCHO_APP_URL, 'walkthrough')}
  2. Run:  hermes honcho setup
     Paste the key when prompted.
""")
        if _yes(_prompt("  Run 'hermes honcho setup' now?", default="y")):
            cmd_setup(args)
            cfg = _read_config()
            has_key = bool(_resolve_api_key(cfg))
        else:
            print("\n  Run 'hermes honcho setup' when ready, then re-run this walkthrough.")

    print("\nStep 2  Detected OpenClaw memory files\n")
    if user_files or agent_files:
        for files, label in ((user_files, "User memory"), (agent_files, "Agent identity")):
            if files:
                peer = "user" if files is user_files else "AI"
                print(f"  {label} ({len(files)} file(s)) — will go to Honcho {peer} peer:")
                print("\n".join(f"    {f}" for f in files))
    else:
        print("  No OpenClaw native memory files found in cwd or ~/.openclaw/.\n"
              "  If your files are elsewhere, copy them here before continuing,\n"
              "  or seed them manually:  hermes honcho identity <path/to/file>")

    print("""
Step 3  Migrate user memory files → Honcho user peer

  USER.md and MEMORY.md contain facts about you that the agent should
  remember across sessions. Honcho will store these under your user peer
  and inject relevant excerpts into the system prompt automatically.
""")
    if user_files:
        print(f"  Found: {', '.join(f.name for f in user_files)}")
        print("""
  These are picked up automatically the first time you run 'hermes'
  with Honcho configured and no prior session history.
  (Hermes calls migrate_memory_files() on first session init.)

  If you want to migrate them now without starting a session:""")
        print("    hermes honcho migrate  — this step handles it interactively\n" * len(user_files), end="")
        if has_key:
            _offer("  Upload user memory files to Honcho now?", _migrate_upload, user_files)
        else:
            print("  Run 'hermes honcho setup' first, then re-run this step.")
    else:
        print("  No user memory files detected. Nothing to migrate here.")

    print("""
Step 4  Seed AI identity files → Honcho AI peer

  SOUL.md, IDENTITY.md, AGENTS.md, TOOLS.md, BOOTSTRAP.md define the
  agent's character, capabilities, and behavioral rules. In OpenClaw
  these are injected via file search at prompt-build time.

  In Hermes, they are seeded once into Honcho's AI peer through the
  observation pipeline. Honcho builds a representation from them and
  from every subsequent assistant message (observe_me=True). Over time
  the representation reflects actual behavior, not just declaration.
""")
    if agent_files:
        print(f"  Found: {', '.join(f.name for f in agent_files)}")
        print()
        if has_key:
            _offer("  Seed AI identity from all detected files now?", _migrate_seed, agent_files)
        else:
            print("  Run 'hermes honcho setup' first, then seed manually:")
            print("\n".join(f"    hermes honcho identity {f}" for f in agent_files))
    else:
        print("  No agent identity files detected.\n  To seed manually:  hermes honcho identity <path/to/SOUL.md>")

    print("""
Step 5  What changes vs. OpenClaw native memory

  Storage
    OpenClaw: markdown files on disk, searched via QMD at prompt-build time.
    Hermes:   cloud-backed Honcho peers. Files can stay on disk as source
              of truth; Honcho holds the live representation.

  Context injection
    OpenClaw: file excerpts injected synchronously before each LLM call.
    Hermes:   the user's peer card at the start of each session, and what
              Honcho recalls about the user, fetched async, on each turn.

  Memory growth
    OpenClaw: you edit files manually to update memory.
    Hermes:   Honcho observes every message and updates representations
              automatically. Files become the seed, not the live store.

  Honcho tools (available to the agent during conversation)
    honcho_chat          — ask Honcho a question, synthesized answer
    honcho_search        — semantic search over stored context
    honcho_profile       — fast peer card snapshot
    honcho_conclude      — write a conclusion/fact back to memory

  Session naming
    OpenClaw: no persistent session concept — files are global.
    Hermes:   one Honcho session per chat, in scopes for its repository
              and this machine.

Step 6  Next steps
""")
    if not has_key:
        print("  1. hermes honcho setup              — configure API key (required)\n"
              "  2. hermes honcho migrate            — re-run this walkthrough")
    else:
        print("""  1. hermes honcho status             — verify Honcho connection
  2. hermes                           — start a session
  3. hermes honcho identity --show    — verify AI peer representation""")
    print()


# ── dispatch / argparse ────────────────────────────────────────────────────

# (subcommand, help, handler, ((arg, kwargs), ...)); order defines --help order.
_SUBCOMMANDS = (
    ("setup", "Initial Honcho setup (redirects to hermes memory setup)", None, ()),
    ("status", "Show current Honcho config and connection status", cmd_status, (
        ("--all", dict(action="store_true", help="Show config overview across all profiles")),
    )),
    ("peers", "Show peer identities across all profiles ('peers map' to map gateway accounts)", cmd_peers, (
        ("peers_action", dict(nargs="?", default=None, choices=("map",), metavar="map",
                              help="'map': interactively map gateway accounts to user peers")),
    )),
    ("peer", "Show or update peer names", cmd_peer, (
        ("--user", dict(metavar="NAME", help="Set user peer name")),
        ("--ai", dict(metavar="NAME", help="Set AI peer name")),
    )),
    ("identity", "Seed or show the AI peer's Honcho identity representation", cmd_identity, (
        ("file", dict(nargs="?", default=None, help="Path to file to seed from (e.g. SOUL.md). Omit to show usage.")),
        ("--show", dict(action="store_true", help="Show current AI peer representation from Honcho")),
    )),
    ("migrate", "Step-by-step migration guide from openclaw-honcho to Hermes Honcho", cmd_migrate, ()),
    ("enable", "Enable Honcho for the active profile", cmd_enable, ()),
    ("disable", "Disable Honcho for the active profile", cmd_disable, ()),
    ("sync", "Sync Honcho config to all existing profiles", cmd_sync, ()),
)
_HANDLERS = {name: handler for name, _help, handler, _args in _SUBCOMMANDS if handler}


def honcho_command(args) -> None:
    """Route honcho subcommands."""
    global _profile_override
    _profile_override = getattr(args, "target_profile", None)
    sub = getattr(args, "honcho_command", None)
    if sub == "setup":  # honcho setup goes through the unified memory-provider path
        print("\n  Honcho is configured via the memory provider system.\n  Running 'hermes memory setup'...\n")
        from hermes_cli.memory_setup import cmd_setup_provider
        return cmd_setup_provider("honcho")
    handler = cmd_status if sub is None else _HANDLERS.get(sub)
    if handler is None:
        return print(f"  Unknown honcho command: {sub}\n"
                     "  Available: status, peers, peer, identity, migrate, enable, disable, sync\n")
    try:
        handler(args)
    except ConfigWriteRefused as e:
        print(f"  {e}\n")


def register_cli(subparser) -> None:
    """Build the ``hermes honcho`` argparse subcommand tree on the ``hermes honcho`` parser."""
    subparser.add_argument("--target-profile", metavar="NAME", dest="target_profile",
                           help="Target a specific profile's Honcho config without switching")
    subs = subparser.add_subparsers(dest="honcho_command")
    for name, help_text, _handler, arguments in _SUBCOMMANDS:
        parser = subs.add_parser(name, help=help_text)
        for flag, kwargs in arguments:
            parser.add_argument(flag, **kwargs)
    subparser.set_defaults(func=honcho_command)
