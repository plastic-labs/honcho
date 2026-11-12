"""Honcho client construction and ``HonchoClientConfig`` resolution.

A profile's settings are ``hosts.<host>`` of ``$HERMES_HOME/honcho.json`` and nothing else in any file
(see ``host_config``), then env vars (HONCHO_API_KEY, HONCHO_BASE_URL, HONCHO_TIMEOUT), then defaults.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import sys
import threading as _threading
import time
import weakref
# --- per-identity client cache ------------------------------------------- One slot per client identity,
# replacing the single process-wide slot that pinned the first profile's workspace and bearer for every
# later profile in multi-profile processes (#69123 multiplexed gateway, #74065 dashboard). The legacy names
# above are retained only for reset bookkeeping.
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from agent.memory_provider import spawn_context_thread as _spawn_context_thread
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

from .attribution import HONCHO_APP_URL, attribution_url
from .client_cache import (
    _DEFAULT_HTTP_TIMEOUT, _client_cache_key, _client_slots, _client_slots_lock, _refresh_oauth, _slot_for,
)

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

HOST = "hermes"
PLUGIN = "hermes-plugin-honcho"


def _sanitize_url(url: str | None) -> str | None:
    """``url``, or None (with a warning) if it carries non-printable ASCII: a stray terminal
    escape in a pasted URL otherwise makes the SDK raise at client construction."""
    if url is None or all(0x20 <= ord(c) < 0x7F for c in url):
        return url
    logger.warning("Honcho base_url contains non-printable characters and will be ignored: %r", url)
    return None


def profile_host_key(profile: str | None) -> str:
    """Return the safe Honcho host key for a Hermes profile."""
    if not profile or profile in {"default", "custom"}:
        return HOST
    sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in profile).strip("_")
    return f"{HOST}_{sanitized or 'profile'}"


def _host_block(raw: dict, host: str) -> dict:
    """Return host config, accepting legacy dot-form profile host keys."""
    hosts = raw.get("hosts") or {}
    block = hosts.get(host, {})
    if block or not host.startswith(f"{HOST}_"):
        return block
    return hosts.get(f"{HOST}.{host[len(HOST) + 1:]}", {})


def resolve_active_host() -> str:
    """Honcho host key: HERMES_HONCHO_HOST (profile-scoped .env), else the active profile. The config's
    ``defaultHost`` is honored only for the default profile so named profiles stay isolated — which is
    also why the override is read through the secret scope: from raw environ it would fold every
    multiplexed profile onto the default profile's host block and peer."""
    explicit = (get_secret("HERMES_HONCHO_HOST", "") or "").strip()
    if explicit:
        return explicit
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile_host = profile_host_key(get_active_profile_name())
    except Exception:
        profile_host = HOST
    if profile_host == HOST:
        try:
            from .host_config import legacy_path
            source = legacy_path(resolve_config_path())  # the profile's file, or until then the one 1.1.0 read
            default_host = str(_read_config(source).get("defaultHost", "")).strip() if source else ""
        except Exception:
            default_host = ""
        if default_host:
            return default_host
    return profile_host


def _read_config(path: Path) -> dict:
    """Parse a honcho.json; {} when absent (parse/OS errors propagate)."""
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def resolve_global_config_path() -> Path:
    """Return the shared Honcho config path for the current HOME."""
    return Path.home() / ".honcho" / "config.json"


def resolve_config_path() -> Path:
    """The active profile's Honcho config: ``$HERMES_HOME/honcho.json``. Resolved per call, since one
    gateway process serves several profiles."""
    return get_hermes_home() / "honcho.json"


# --- config coercion -------------------------------------------------------

def _first_set(*vals, default):
    """First non-None value, else default."""
    return next((val for val in vals if val is not None), default)


def _first_parsed(vals, caster: Callable[[Any], Any], default):
    """First non-None value that ``caster`` accepts, else default."""
    for val in (v for v in vals if v is not None):
        try:
            return caster(val)
        except (ValueError, TypeError):
            pass
    return default


def _positive_float(value: Any) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(value)
    return parsed


def _resolve_optional_float(*values: Any) -> float | None:
    """First value (blank strings included) that parses as a positive float, else None."""
    return _first_parsed(values, _positive_float, None)


class _HostLookup:
    """Host-block-over-root field lookups for one honcho.json."""

    def __init__(self, host_block: dict, raw: dict):
        self.host, self.raw = host_block, raw

    def pick(self, key: str, default=None):
        """Truthy host value, else ``raw.get(key, default)``."""
        return self.host.get(key) or self.raw.get(key, default)

    def pick_set(self, key: str, default=None):
        """Non-None host value, else ``raw.get(key, default)``."""
        return _first_set(self.host.get(key), default=self.raw.get(key, default))

    def vals(self, *keys: str) -> list:
        """Host values then root values for the aliased ``keys`` (lookup order)."""
        return [self.host.get(k) for k in keys] + [self.raw.get(k) for k in keys]

    def flag(self, *keys: str, default: bool) -> bool:
        """First non-None value across host-then-root for the aliased ``keys``, as bool."""
        return bool(_first_set(*self.vals(*keys), default=default))

    def parsed(self, key: str, caster: Callable[[Any], Any], default):
        """First host-then-root value ``caster`` accepts, else default."""
        return _first_parsed(self.vals(key), caster, default)

    def present(self, key: str, default=None):
        """Host value if the key is PRESENT there (even empty/None), else root."""
        return self.host[key] if key in self.host else self.raw.get(key, default)

    def string(self, key: str, default: str = "") -> str:
        """String field where a host-level empty string can override root."""
        value = self.present(key, default)
        return default if value is None else str(value).strip()

    def string_map(self, key: str) -> dict[str, str]:
        """String-to-string map; a host-level map replaces the root map wholesale."""
        source = self.present(key)
        if not isinstance(source, dict):
            return {}
        pairs = ((str(k).strip(), str(v).strip() if v is not None else "") for k, v in source.items())
        return {k: v for k, v in pairs if k and v}


def _is_local_base_url(base_url: str | None) -> bool:
    """True for loopback/RFC1918/link-local/ULA/CGNAT self-hosted Honcho URLs. Local
    deployments can run without auth but the SDK needs a non-empty api_key, so LAN/VPN
    URLs get the same placeholder-key treatment as localhost."""
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").strip().lower()
    except Exception:
        host = ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # Tailscale/other VPN setups often sit in carrier-grade NAT space (100.64.0.0/10).
    cgnat = ip.version == 4 and ipaddress.ip_address("100.64.0.0") <= ip <= ipaddress.ip_address("100.127.255.255")
    return ip.is_loopback or ip.is_private or ip.is_link_local or cgnat


def _env_base_url() -> str | None:
    """HONCHO_BASE_URL / HONCHO_URL (the SDK's own var). A self-hosted URL varies per profile, so it is
    read through the secret scope: the scoped HONCHO_API_KEY beside it must not be sent to the default
    profile's server."""
    return (get_secret("HONCHO_BASE_URL", "") or "").strip() or (get_secret("HONCHO_URL", "") or "").strip() or None


def _warn_default_key_not_inherited(host: str, path: Path) -> None:
    """A named profile's block never takes the default profile's apiKey (profiles are credential-isolated).
    The failure is silent 401s, so say so loudly when the default profile has a key."""
    from hermes_cli.profiles import _get_default_hermes_home
    from .host_config import effective_block, legacy_path

    default_path = _get_default_hermes_home() / "honcho.json"
    try:
        has_key = legacy_path(default_path) is not None and bool(effective_block(default_path, HOST).get("apiKey"))
    except (ValueError, OSError):
        return
    if has_key:
        logger.warning("Honcho host block '%s' has no apiKey; the default '%s' host's key is NOT inherited "
                       "(profiles are credential-isolated). Set apiKey on hosts.%s in %s or this profile "
                       "runs unauthenticated.", host, HOST, host, path)


def _connection_fields(look: _HostLookup, host: str) -> dict[str, Any]:
    """Resolve identity/credential/transport fields (host block -> env -> default)."""
    api_key = look.pick("apiKey") or get_secret("HONCHO_API_KEY")
    base_url = _sanitize_url(look.pick("baseUrl") or _env_base_url())
    return {
        "workspace_id": look.pick("workspace") or host,
        "ai_peer": look.pick("aiPeer") or host,
        "api_key": api_key,
        "environment": "production",  # the migration folded `environment: local` into baseUrl
        "base_url": base_url,
        "timeout": _resolve_optional_float(os.environ.get("HONCHO_TIMEOUT")),
        # Explicit enabled (host, then root) wins; else auto-enable on key/url.
        "enabled": _first_set(*look.vals("enabled"), default=bool(api_key or base_url)),
    }


def _behavior_fields(look: _HostLookup) -> dict[str, Any]:
    """Resolve the user peer and the gateway identity settings (host block -> defaults)."""
    return {
        "peer_name": look.pick("peerName"),
        # pinUserPeer is the clearer name; the original pinPeerName stays accepted.
        "pin_peer_name": look.flag("pinUserPeer", "pinPeerName", default=False),
        "user_peer_aliases": look.string_map("userPeerAliases"),
        "runtime_peer_prefix": look.string("runtimePeerPrefix"),
        "a2a_sessions": look.flag("a2aSessions", default=True),
    }


@dataclass
class HonchoClientConfig:
    """Configuration for Honcho client, resolved for a specific host."""

    host: str = HOST
    workspace_id: str = "hermes"
    api_key: str | None = None
    environment: str = "production"
    base_url: str | None = None  # self-hosted override of the environment mapping
    timeout: float | None = None  # SDK HTTP timeout, seconds
    # Identity
    peer_name: str | None = None
    ai_peer: str = "hermes"
    # True: peer_name wins over gateway runtime identity (Telegram UID, ...), so a
    # single-user deployment keeps one memory across platforms.
    # This keeps memory unified across platforms for single-user deployments where Honcho's one peer-name is
    # an unambiguous identity — otherwise each platform would fork memory into its own peer (#14984).
    # Default ``False`` preserves existing multi-user behaviour.
    pin_peer_name: bool = False
    # Gateway runtime user id -> stable Honcho peer; host map replaces root map.
    user_peer_aliases: dict[str, str] = field(default_factory=dict)
    runtime_peer_prefix: str = ""  # prefix for unknown runtime user ids, e.g. "telegram_"
    enabled: bool = False
    # Bot-authored DMs write into their own session per sender bot.
    a2a_sessions: bool = True
    # {"hosts": {host: block}}: the block as read, so nothing downstream can reach the file's top level.
    raw: dict[str, Any] = field(default_factory=dict)
    # The block holds a setting, vs auto-enabled from a stray env key.
    explicitly_configured: bool = False
    # Provenance captured at resolution time; bound consumers use these instead of
    # re-resolving (the resolvers read a ContextVar background threads can't see).
    # Provenance: WHERE this config was resolved from, captured at resolution time (inside the caller's
    # profile scope). Bound consumers (session manager, OAuth refresh paths) use these instead of
    # re-resolving resolve_config_path()/get_hermes_home() later — those resolvers read a ContextVar that
    # background threads cannot see, so re-resolution from a daemon thread silently lands on the DEFAULT
    # profile (#69123, #74065).
    config_path: Path | None = None
    hermes_home: Path | None = None

    def bound_config_path(self) -> Path:
        """Config path this was resolved from; ambient fallback for hand-built configs."""
        return self.config_path if self.config_path is not None else resolve_config_path()

    @classmethod
    def from_env(cls, workspace_id: str = "hermes", host: str | None = None) -> HonchoClientConfig:
        """Create config from environment variables (fallback)."""
        resolved_host = host or resolve_active_host()
        api_key = get_secret("HONCHO_API_KEY")
        base_url = _sanitize_url(_env_base_url())
        return cls(
            host=resolved_host, workspace_id=workspace_id, api_key=api_key, base_url=base_url,
            environment=get_secret("HONCHO_ENVIRONMENT", "") or "production",
            timeout=_resolve_optional_float(os.environ.get("HONCHO_TIMEOUT")),
            ai_peer=resolved_host, enabled=bool(api_key or base_url),
            config_path=resolve_config_path(), hermes_home=get_hermes_home(),
        )

    @classmethod
    def from_global_config(cls, host: str | None = None, config_path: Path | None = None) -> HonchoClientConfig:
        """Config from the profile's host block, falling back to env when no config file exists.
        ``host=None`` derives it from the active Hermes profile. ``raw`` holds only that block, so
        nothing downstream can read the file's top level."""
        from .host_config import effective_block, legacy_path

        resolved_host = host or resolve_active_host()
        path = config_path or resolve_config_path()
        if legacy_path(path) is None:
            logger.debug("No Honcho config at %s, falling back to env", path)
            return cls.from_env(host=resolved_host)
        try:
            block = effective_block(path, resolved_host)
        except (ValueError, OSError) as e:
            logger.warning("Failed to read %s: %s, falling back to env", path, e)
            return cls.from_env(host=resolved_host)

        look = _HostLookup(block, {})
        cfg = cls(
            host=resolved_host, **_connection_fields(look, resolved_host), **_behavior_fields(look),
            raw={"hosts": {resolved_host: block}},
            explicitly_configured=any(k != "schemaVersion" for k in block),
            config_path=path, hermes_home=get_hermes_home(),
        )
        if not cfg.api_key and resolved_host != HOST and cfg.explicitly_configured:
            _warn_default_key_not_inherited(resolved_host, path)
        return cfg

    # Honcho rejects session IDs over 100 chars; long gateway keys (Matrix
    # rooms + thread ids, Slack threads) overflow after sanitization.
    _HONCHO_SESSION_ID_MAX_LEN, _HONCHO_SESSION_ID_HASH_LEN = 100, 8

    @classmethod
    def _enforce_session_id_limit(cls, sanitized: str, original: str) -> str:
        """Truncate to the limit with a ``-<sha256 prefix>`` suffix hashed over the ORIGINAL key:
        two long keys sharing a prefix stay distinct; keys that sanitize identically still
        collide intentionally (same logical session)."""
        max_len, hash_len = cls._HONCHO_SESSION_ID_MAX_LEN, cls._HONCHO_SESSION_ID_HASH_LEN
        if len(sanitized) <= max_len:
            return sanitized
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:hash_len]
        prefix = sanitized[: max_len - hash_len - 1].rstrip("-")
        return f"{prefix}-{digest}"

    # ``Platform.LOCAL.value`` in ``gateway.config``: the platform slot of a session key for a
    # Hermes CLI, TUI, Desktop or ACP chat routed through the gateway runtime.
    _LOCAL_GATEWAY_PLATFORM = "local"

    @classmethod
    def is_local_gateway_key(cls, key: str) -> bool:
        """``agent:<ns>:local:...`` — a Hermes-surface (CLI/TUI/Desktop/ACP) chat keyed by the gateway.

        ``gateway.session.build_session_key`` lays a key out as ``agent:<ns>:<platform>:<chat_type>...``;
        the namespace is one segment (``main``, ``main~`` or a profile id), so the platform is always
        the third. Only that slot is read: a chat id such as ``local-<sha>`` further along never matches."""
        parts = key.split(":")
        return len(parts) >= 3 and parts[0] == "agent" and parts[2] == cls._LOCAL_GATEWAY_PLATFORM

    @classmethod
    def resolve_session_name(cls, session_id: str | None = None, gateway_session_key: str | None = None) -> str | None:
        """One Honcho session per Hermes chat: the gateway's key for a chat the gateway routes (a Telegram or
        Discord chat or thread, or a Desktop/TUI chat it carries), which stays put across /new, compression and
        restarts; else the Hermes session id. None when there is neither."""
        import re

        key = gateway_session_key or session_id or ""
        slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', key).strip('-')
        return cls._enforce_session_id_limit(slug, key) if slug else None


def _repo_name(cwd: str) -> str:
    """Repository name for ``cwd``, shared by every linked worktree (``--git-common-dir`` is the main
    repo's ``.git``, or a bare repo); the directory name when ``cwd`` is not in a git repository."""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", cwd=cwd, timeout=5, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out is None or out.returncode != 0 or not out.stdout.strip():
        return Path(cwd).resolve().name
    common = (Path(cwd) / out.stdout.strip()).resolve()
    return common.parent.name if common.name == ".git" else common.name.removesuffix(".git")


def session_scopes(cwd: str | None) -> list[str]:
    """Honcho scopes a new session joins: its repository, when the chat has a working directory, and this
    machine's short hostname. Lowercased, anything outside ``[a-z0-9_-]`` turned into ``-``."""
    import re
    import socket

    names = ([_repo_name(cwd)] if cwd else []) + [socket.gethostname().split(".")[0]]
    scopes: list[str] = []
    for name in names:
        scope = re.sub(r"[^a-z0-9_-]", "-", name.lower())
        if re.search(r"[a-z0-9]", scope) and scope not in scopes:
            scopes.append(scope)
    return scopes


# Threads keyed by the provider or manager that owns them, so shutdown never waits on another agent's work.
_plugin_threads: "weakref.WeakKeyDictionary[Any, weakref.WeakSet]" = weakref.WeakKeyDictionary()
_plugin_threads_lock = _threading.Lock()


def spawn_context_thread(
    target, *, name: str, daemon: bool = True, args: tuple = (), owner: Any = None,
) -> "_threading.Thread":
    """agent.memory_provider.spawn_context_thread plus an ``owner``: the thread is registered so
    join_plugin_threads can wait on exactly the threads this provider or manager spawned."""
    thread = _spawn_context_thread(target, name=name, daemon=daemon, args=args)
    if owner is not None:
        with _plugin_threads_lock:
            _plugin_threads.setdefault(owner, weakref.WeakSet()).add(thread)
    return thread


def join_plugin_threads(owners, timeout: float) -> list[str]:
    """Join every live thread spawned for ``owners`` within one shared ``timeout``. Returns the names
    of the threads still running when the budget ran out."""
    deadline = time.monotonic() + max(0.0, timeout)
    me = _threading.current_thread()
    with _plugin_threads_lock:
        threads = [t for owner in owners if owner is not None
                   for t in list(_plugin_threads.get(owner, ())) if t.is_alive() and t is not me]
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return [t.name for t in threads if t.is_alive()]


def close_honcho_clients() -> None:
    """Close the HTTP pool of every cached client and drop the slots. Process exit only: one client
    serves every manager with the same identity, so a per-agent shutdown must never call this."""
    with _client_slots_lock:
        slots = list(_client_slots.values())
        _client_slots.clear()
    for slot in slots:
        close = getattr(getattr(slot.peek(), "_http", None), "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


_exit_close_registered = False


def _register_exit_close() -> None:
    global _exit_close_registered
    with _client_slots_lock:
        if _exit_close_registered:
            return
        _exit_close_registered = True
    atexit.register(close_honcho_clients)


def get_honcho_client(config: HonchoClientConfig | None = None) -> Honcho:
    """Get or create the Honcho client for this config's identity. Clients are cached PER
    IDENTITY (host, workspace, provenance paths, credential fingerprint, timeout) so
    multi-profile processes don't share a first-config-wins client. With no config the active
    honcho.json is resolved — correct only on threads that see the profile ContextVar; pass a
    bound config elsewhere. Each identity's client is built once under concurrent first calls.

    See #69123, #74065.
    """
    key = _client_cache_key(config)
    slot = _slot_for(key)
    cached = slot.peek()
    if cached is not None:
        _refresh_oauth(config, cached, slot)
        refreshed = slot.peek()
        if refreshed is not None:
            return refreshed
        # Slot was reset by a failed in-place rotation — rebuild below.

    if config is None:
        config = HonchoClientConfig.from_global_config()

    # Start with a live access token rather than 401ing an hour in.
    _refresh_oauth(config)

    if not config.api_key and not config.base_url:
        raise ValueError(f"Honcho API key not found. Get your API key at {attribution_url(HONCHO_APP_URL, 'missing-key-error')}, "
                         "then run 'hermes honcho setup' or set HONCHO_API_KEY. "
                         "For local instances, set HONCHO_BASE_URL instead.")

    return slot.get(lambda: _build_client(config))


def _hermes_version() -> str:
    """The running Hermes version, or "" if ``hermes_cli`` can't be imported (defensive tests)."""
    try:
        from hermes_cli import __version__
        return __version__.strip()
    except Exception:
        return ""


def _plugin_version() -> str:
    """This plugin's version, from the ``plugin.yaml`` manifest Hermes installs beside this module.

    Not ``importlib.metadata``: an installed plugin directory carries the manifest but no dist-info.
    """
    with contextlib.suppress(Exception):
        for line in Path(__file__).with_name("plugin.yaml").read_text().splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip("\"'") or "unknown"
    return "unknown"


def telemetry_headers() -> dict[str, str]:
    """Client-identity headers for telemetry attribution, e.g.::

        X-Honcho-Host: hermes/0.18.2 (darwin)
        X-Honcho-Plugin: hermes-plugin-honcho/1.2.0

    The version is omitted from the host when it can't be resolved. Values are recorded verbatim
    by the API's telemetry middleware; nothing else reads them, so they must never raise.
    """
    version = _hermes_version()
    return {
        "X-Honcho-Host": f"{HOST}/{version} ({sys.platform})" if version else f"{HOST} ({sys.platform})",
        "X-Honcho-Plugin": f"{PLUGIN}/{_plugin_version()}",
    }


AGENT_MODEL_HEADER = "X-Honcho-Agent-Model"


def set_agent_model_header(client: Any, model: str) -> None:
    """Write ``model`` into a built client's headers, or delete the header when it is empty, so a
    request never carries a model the agent is no longer running. Covers the sync client and the
    config the SDK builds its async client from. Best-effort: SDK internals may move."""
    with contextlib.suppress(Exception):
        http = getattr(client, "_http", None)
        config = getattr(client, "_http_config", None) or {}
        async_http = getattr(client, "_async_http", None)
        for headers in (getattr(http, "default_headers", None), config.get("default_headers"),
                        getattr(async_http, "default_headers", None)):
            if not isinstance(headers, dict):
                continue
            if model:
                headers[AGENT_MODEL_HEADER] = model
            else:
                headers.pop(AGENT_MODEL_HEADER, None)


def _build_client(config: HonchoClientConfig) -> "Honcho":
    """Construct the SDK client (runs inside the slot factory so racing callers share one)."""
    with contextlib.suppress(Exception):  # lazy-dep failures fall through to the canonical import error below
        pass  # dependencies come from pyproject.toml (installed by Hermes on install/enable/update)
    try:
        from honcho import Honcho
    except ImportError:
        raise ImportError("honcho-ai is required for Honcho integration. Install it with: pip install honcho-ai  "
                          "(or run `hermes honcho setup` to configure).")

    # config.yaml honcho.base_url / timeout fill whatever honcho.json left unset.
    base_url, timeout = config.base_url, config.timeout
    if not base_url or timeout is None:
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            honcho_cfg = load_config().get("honcho", {})
            if isinstance(honcho_cfg, dict):
                base_url = base_url or _sanitize_url(honcho_cfg.get("base_url", "").strip() or None)
                if timeout is None:
                    timeout = _resolve_optional_float(honcho_cfg.get("timeout"), honcho_cfg.get("request_timeout"))
    if timeout is None:
        timeout = _DEFAULT_HTTP_TIMEOUT  # an unconfigured install must not hang on a stalled request

    if base_url:
        logger.info("Initializing Honcho client (base_url: %s, workspace: %s)", base_url, config.workspace_id)
    else:
        # Name the SDK's environment fallback at INFO so a self-hosted user whose
        # config wasn't picked up notices they're talking to the public cloud.
        logger.info("Initializing Honcho client (host: %s, workspace: %s, base_url unset — SDK will resolve from environment=%s)",
                    config.host, config.workspace_id, config.environment)

    # Local instances need no key but the SDK wants a non-empty string: honor a
    # key set EXPLICITLY in honcho.json (host block or root) and treat an
    # env-sourced key as likely-cloud, substituting the placeholder.
    raw = config.raw or {}
    explicit_key = _host_block(raw, config.host).get("apiKey") or raw.get("apiKey")
    api_key = "local" if _is_local_base_url(base_url) and not explicit_key else config.api_key
    kwargs: dict = {"workspace_id": config.workspace_id, "api_key": api_key, "environment": config.environment,
                    "timeout": timeout, "default_headers": telemetry_headers()}
    if base_url:
        # The SDK's route builders already carry the version prefix ("/v3/..."), so
        # strip a trailing version segment from any base_url to avoid "/v3/v3/...".
        import re
        kwargs["base_url"] = re.sub(r"/v\d+/*$", "", base_url).rstrip("/")
    _register_exit_close()
    return Honcho(**kwargs)


def reset_honcho_client() -> None:
    """Reset all cached Honcho clients (tests, OAuth re-login)."""
    with _client_slots_lock:
        _client_slots.clear()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'SingletonSlot': ('plugins.plugin_utils', 'SingletonSlot'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
