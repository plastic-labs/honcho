"""Build the ``honcho config`` report, every secret redacted.

Read-only by construction: it never refreshes a token, never writes
``config.json`` and never starts or stops containers. Three modes:

    offline   files and env only (``--no-network``)
    local     + ``docker ps`` / ``compose ps`` / ``inspect`` and a ``/health``
              GET against each running local stack (the default)
    deep      + the remote calls: ``/health`` and ``/openapi.json`` on the
              resolved API, one authenticated list-workspaces call per
              distinct key (the CLI's and each integration's own), and server
              versions for running local stacks (``--deep``)

Sections (JSON keys, in order):

    meta          CLI / SDK / Python versions, platform, config dir, mode
    files         config.json stat + a name-only listing of the config dir
    config_file   the whole config.json (other tools' ``hosts.*`` included)
    env           HONCHO_* vars, and the LLM_/EMBEDDING_/… vars ``honcho start``
                  forwards into a profile ``.env``
    integrations  harness plugins and MCP clients wired up on this machine
    resolved      what the CLI actually uses, with the source of each value
    probes        reachability, server version, authenticated call (deep)
    local_stack   per profile: state, ports, image, services, .env,
                  config.toml; other Honcho containers on the host
    warnings      problems found above: stable id, severity, short text, fix

Paths under the home directory are written as ``~/…``.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import time
import tomllib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from platform import python_version
from typing import Any
from urllib.parse import urlsplit

import httpx

from honcho_cli import __version__
from honcho_cli import config as cfg
from honcho_cli.common import get_flag_overrides, get_resolved_config
from honcho_cli.config import CLIConfig, identity_headers
from honcho_cli.integrations import detect_integrations
from honcho_cli.local.docker import (
    DockerError,
    compose_ps,
    image_versions,
    list_honcho_containers,
    services_running,
    stack_containers_up,
)
from honcho_cli.local.env import SETTINGS_PREFIXES, is_placeholder_key, read_env_file
from honcho_cli.local.health import api_healthy
from honcho_cli.local.profile import (
    LocalProfile,
    list_profile_names,
    load_profile,
    profiles_dir,
    validate_profile_name,
)
from honcho_cli.redact import (
    is_secret_path,
    redact_tree,
    redact_value,
    safe_url,
    secret_marker,
)

_ERROR_NAMES = {
    "ConnectError": "connection refused",
    "ConnectTimeout": "connection timed out",
    "ReadTimeout": "read timed out",
    "InvalidURL": "invalid URL",
    "UnsupportedProtocol": "unsupported URL scheme",
}

PRECEDENCE = (
    "Flags override environment variables, which override ~/.honcho/config.json; "
    "built-in defaults apply last. One exception: a valid OAuth login is used "
    "before HONCHO_API_KEY."
)
PROBE_TIMEOUT = 5.0
DOCKER_TIMEOUT = 8.0

# Env vars other Honcho tools read that the CLI does not — a common reason
# "I set the variable and nothing changed".
_FOREIGN_ENV = {
    "HONCHO_URL": "HONCHO_BASE_URL",
    "HONCHO_ENDPOINT": "HONCHO_BASE_URL",
    "HONCHO_WORKSPACE": "HONCHO_WORKSPACE_ID",
    "HONCHO_CONFIG_PATH": "HONCHO_CONFIG_DIR",
}
_PROVIDER_KEYS = {
    "openai": "LLM_OPENAI_API_KEY",
    "anthropic": "LLM_ANTHROPIC_API_KEY",
    "gemini": "LLM_GEMINI_API_KEY",
}
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
def _fix_for(name: str, where: str) -> str:
    """The command that replaces the key at ``where`` — not just one that sounds right."""
    if name == "codex" and "config.toml" in where:
        return "codex-honcho install  (re-copies your current key into ~/.codex/config.toml)"
    if name == "openclaw":
        return "openclaw honcho setup --reconfigure  (without --reconfigure it keeps the saved key)"
    if name == "claude_code" and ".claude.json" in where:
        return (
            "claude mcp remove honcho, then claude mcp add honcho --transport http "
            'https://mcp.honcho.dev --header "Authorization: Bearer <your key>"'
        )
    return f"replace the key at {where}"


def build_report(
    *, network: bool = True, deep: bool = False, profile: str | None = None
) -> dict[str, Any]:
    """Assemble the full report. ``profile`` limits the local-stack section."""
    mode = "offline" if not network else ("deep" if deep else "local")
    # Snapshot before ``CLIConfig.load`` pops empty HONCHO_* vars from the env.
    honcho_env = {k: v for k, v in os.environ.items() if k.startswith("HONCHO_")}
    stack_env = {k: v for k, v in os.environ.items() if k.startswith(SETTINGS_PREFIXES)}
    config = get_resolved_config()
    raw, parse_error = _read_raw_config()
    warnings: list[dict[str, Any]] = []

    integrations, integration_keys = detect_integrations(raw)
    report: dict[str, Any] = {
        "meta": _meta(mode),
        "files": _files(parse_error, warnings),
        "config_file": _config_file(raw, parse_error),
        "env": _env(honcho_env, stack_env, warnings),
        "integrations": integrations,
        "resolved": _resolved(config, raw, honcho_env, warnings),
        "probes": _probes(config, mode, warnings),
        "local_stack": _local_stack(profile, mode, raw, config, integrations, warnings),
    }
    _integration_keys(integrations, integration_keys, raw, config, mode, warnings)
    report["warnings"] = warnings
    return report


def _warn(
    warnings: list[dict[str, Any]],
    check: str,
    short: str,
    *,
    detail: str | None = None,
    subject: str | None = None,
    path: str | None = None,
    fix: str | None = None,
    severity: str = "warning",
) -> None:
    """Record one finding. ``short`` fits on a summary line; ``detail`` adds context."""
    warnings.append(
        {
            "check": check,
            "severity": severity,
            "subject": subject,
            "path": path,
            "short": short,
            "detail": detail,
            "fix": fix,
        }
    )


# --------------------------------------------------------------------------- #
# sections


def _meta(mode: str) -> dict[str, Any]:
    try:
        sdk = metadata.version("honcho-ai")
    except metadata.PackageNotFoundError:
        sdk = None
    return {
        "cli_version": __version__,
        "sdk_version": sdk,
        "python": python_version(),
        "platform": sys.platform,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "precedence": PRECEDENCE,
        "config_dir": _tilde(cfg.CONFIG_DIR),
        "config_dir_source": "env HONCHO_CONFIG_DIR" if os.environ.get("HONCHO_CONFIG_DIR") else "default",
    }


def _read_raw_config() -> tuple[dict | None, str | None]:
    """Parsed config.json, or the reason it could not be used."""
    path = cfg.CONFIG_FILE
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return None, "top-level value is not a JSON object"
    return data, None


def _files(parse_error: str | None, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    path = cfg.CONFIG_FILE
    shown = _tilde(path)
    info: dict[str, Any] = {"path": shown, "exists": path.exists()}
    if info["exists"]:
        st = path.stat()
        mode = f"{stat.S_IMODE(st.st_mode):o}"
        info.update(mode=mode, size=st.st_size, mtime=_iso(st.st_mtime), parse_ok=parse_error is None)
        if os.name == "posix" and mode != "600":
            _warn(
                warnings, "files/permissions", f"config.json is mode {mode}; it holds credentials",
                path=shown, fix=f"chmod 600 {shown}",
            )
        if parse_error:
            _warn(
                warnings, "files/unparseable", "config.json is not valid JSON",
                detail=f"{parse_error}. The CLI treats it as empty, and `honcho init` would overwrite it.",
                path=shown, fix="fix the JSON by hand, or move the file aside and run `honcho init`",
            )
    else:
        _warn(
            warnings, "files/missing", "no config file", path=shown,
            fix="honcho init (or export HONCHO_API_KEY)",
        )

    entries: list[dict[str, Any]] = []
    if cfg.CONFIG_DIR.is_dir():
        for child in sorted(cfg.CONFIG_DIR.iterdir()):
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": st.st_size,
                    "mtime": _iso(st.st_mtime),
                }
            )
    return {"config_file": info, "config_dir_entries": entries}


def _config_file(raw: dict | None, parse_error: str | None) -> dict | None:
    if parse_error:
        return {"error": parse_error}
    if raw is None:
        return None
    redacted = redact_tree(raw)
    # The per-directory session map is one line per project the user ever
    # opened — noise for a report, and a list of their directory names.
    sessions = redacted.get("sessions")
    if isinstance(sessions, dict):
        redacted["sessions"] = {
            "count": len(sessions),
            "current_dir_session": sessions.get(os.getcwd()),
        }
    return redacted


def _env(
    honcho_env: dict[str, str], stack_env: dict[str, str], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    for name, cli_name in _FOREIGN_ENV.items():
        if honcho_env.get(name):
            _warn(
                warnings, "env/ignored-var", f"{name} is set but the CLI reads {cli_name}",
                detail="Other Honcho tools may honour it; the CLI does not.",
                path=f"env {name}", fix=f"set {cli_name} for the CLI", severity="info",
            )
    for name in cfg.ENV_MAP.values():
        if name in honcho_env and honcho_env[name] == "":
            _warn(warnings, "env/empty-var", f"{name} is set but empty; ignored",
                  path=f"env {name}", severity="info")
    return {
        "honcho": {k: redact_tree(v, key=k) for k, v in sorted(honcho_env.items())},
        "stack_forwarded": {k: redact_tree(v, key=k) for k, v in sorted(stack_env.items())},
    }


def _resolved(
    config: CLIConfig,
    raw: dict | None,
    honcho_env: dict[str, str],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    base_url = safe_url(config.base_url)
    file_url = raw.get("environmentUrl") if raw else None
    if honcho_env.get("HONCHO_BASE_URL"):
        url_source = "env HONCHO_BASE_URL"
    elif isinstance(file_url, str) and file_url:
        url_source = "config.json environmentUrl"
    else:
        url_source = "default"

    tokens = config.usable_oauth()
    env_key = honcho_env.get("HONCHO_API_KEY") or ""
    file_key = raw.get("apiKey") if raw and isinstance(raw.get("apiKey"), str) else ""
    # Mirrors CLIConfig.resolved_api_key, labelled.
    if tokens and tokens.access_valid():
        mode, source, value = "oauth", "config.json oauth", tokens.access_token
    elif config.api_key:
        mode, value = "api-key", config.api_key
        source = "env HONCHO_API_KEY" if env_key else "config.json apiKey"
    elif tokens:
        mode, source, value = "oauth-expired", "config.json oauth", tokens.access_token
    else:
        mode, source, value = "none", None, ""
    credential: dict[str, Any] = {"mode": mode, "source": source}
    if value:
        credential.update({k: v for k, v in secret_marker(value).items() if k != "redacted"})
    # Other places that hold a key the CLI is not using right now.
    shadows = []
    if env_key and source != "env HONCHO_API_KEY":
        shadows.append({"source": "env HONCHO_API_KEY", "same_key": env_key == value})
    if file_key and source != "config.json apiKey":
        shadows.append({"source": "config.json apiKey", "same_key": file_key == value})
    credential["shadows"] = shadows

    oauth: dict[str, Any] = {"present": False}
    if config.oauth:
        o = config.oauth
        remaining = o.access_expires_at - time.time()
        oauth = {
            "present": True,
            "host": safe_url(o.host) if o.host else None,
            "host_matches_base_url": o.matches_host(config.base_url),
            "expires_in_seconds": int(remaining) if math.isfinite(remaining) else None,
            "refresh_token_present": bool(o.refresh_token),
            "client_id": o.client_id or None,
            "scope": o.scope or None,
        }
        if not math.isfinite(remaining):
            _warn(warnings, "auth/oauth-expiry-invalid", "oauth.accessExpiresAt is not a number",
                  path="config.json oauth.accessExpiresAt", fix="honcho init")
        if not o.matches_host(config.base_url):
            _warn(
                warnings, "auth/oauth-host-mismatch",
                f"OAuth login is for {oauth['host']}, not {base_url}; not used",
                path="config.json oauth.host", fix="honcho init", severity="info",
            )

    if mode == "oauth" and env_key:
        _warn(
            warnings, "auth/env-key-shadowed",
            "HONCHO_API_KEY is set, but the OAuth login takes precedence",
            path="env HONCHO_API_KEY", fix="honcho init → API key, to use the key instead",
            severity="info",
        )
    if mode == "oauth-expired":
        if tokens and tokens.refresh_token:
            _warn(warnings, "auth/oauth-expired", "OAuth login expired; refreshes on the next API call",
                  path="config.json oauth", severity="info")
        else:
            _warn(warnings, "auth/oauth-expired", "OAuth login expired and can't refresh",
                  path="config.json oauth", fix="honcho init")
    if mode == "api-key" and tokens:
        _warn(
            warnings, "auth/oauth-expired",
            "OAuth login expired; using the API key until it refreshes",
            path="config.json oauth", severity="info",
        )
    if mode == "none":
        _warn(warnings, "auth/no-credentials", "no API key or login",
              fix="honcho init (or export HONCHO_API_KEY)")
    if raw:
        a, b = raw.get("environmentUrl"), raw.get("baseUrl")
        if isinstance(a, str) and isinstance(b, str) and a.rstrip("/") != b.rstrip("/"):
            _warn(
                warnings, "config/url-drift", "environmentUrl and baseUrl differ",
                detail=(
                    f"environmentUrl {safe_url(a)} (read by the CLI) vs baseUrl "
                    f"{safe_url(b)} (read by harness plugins)"
                ),
                path="config.json", fix="set both to the same URL",
            )

    flags = get_flag_overrides()
    scope: dict[str, Any] = {}
    for name, letter, env_var in (
        ("workspace", "w", "HONCHO_WORKSPACE_ID"),
        ("peer", "p", "HONCHO_PEER_ID"),
        ("session", "s", "HONCHO_SESSION_ID"),
    ):
        val = getattr(config, f"{name}_id")
        if flags.get(name):
            src: str | None = f"flag -{letter}"
        elif honcho_env.get(env_var):
            src = f"env {env_var}"
        else:
            src = None
        scope[name] = {
            "value": redact_value(name, val) if val else None,
            "source": src,
            "set_with": f"-{letter} or {env_var}",
        }

    return {
        "base_url": {"value": base_url, "source": url_source},
        "credential": credential,
        "oauth": oauth,
        **scope,
    }


def _probes(config: CLIConfig, mode: str, warnings: list[dict[str, Any]]) -> dict[str, Any]:
    if mode != "deep":
        return {"skipped": "--no-network" if mode == "offline" else "not checked; add --deep"}
    url = safe_url(config.base_url)
    api: dict[str, Any] = {"url": url}
    started = time.monotonic()
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(config.base_url.rstrip("/") + "/health")
        api["reachable"] = True
        api["health_status"] = response.status_code
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as e:
        api["reachable"] = False
        # The exception's own message can quote the URL (an "invalid port"
        # that is really part of a password), so only its type is kept.
        api["error"] = _ERROR_NAMES.get(type(e).__name__, type(e).__name__)
        _warn(warnings, "api/unreachable", f"API unreachable at {url}", detail=api["error"],
              path="resolved base_url", fix="check base_url above; `honcho doctor` for more")
    api["latency_ms"] = int((time.monotonic() - started) * 1000)

    key = config.resolved_api_key()
    if api["reachable"]:
        api["server_version"] = _server_version(config.base_url, key)
    if not key:
        api["authenticated"] = None
        api["detail"] = "skipped — no credentials"
    elif api["reachable"]:
        outcome = _probe_key(config.base_url, key, secret_marker(key).get("masked"))
        api["key_status"] = outcome["status"]
        api["authenticated"] = {"accepted": True, "scoped": True, "rejected": False}.get(outcome["status"])
        api["detail"] = {"accepted": "OK", "scoped": "OK (scoped key)"}.get(outcome["status"], outcome["detail"])
        if api["authenticated"] is False:
            _warn(warnings, "api/auth-failed", f"the API refused the CLI's key: {outcome['detail']}",
                  path="resolved credential", fix="honcho init")
        elif api["authenticated"] is None:
            _warn(warnings, "api/auth-unverified", f"couldn't verify the CLI's key ({outcome['status']}: "
                  f"{outcome['detail']})", path="resolved credential", severity="info")
    else:
        api["authenticated"] = False
        api["detail"] = "skipped — unreachable"
    return {"api": api}


def _local_stack(
    profile_flag: str | None,
    mode: str,
    raw: dict | None,
    config: CLIConfig,
    integrations: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    all_names = list_profile_names()
    names = [validate_profile_name(profile_flag)] if profile_flag else all_names

    docker: dict[str, Any]
    others: Any
    if mode == "offline":
        docker, others = {"checked": False}, None
    else:
        try:
            managed = {f"honcho-{name}" for name in all_names}
            others = [
                c for c in list_honcho_containers(timeout=DOCKER_TIMEOUT)
                if c.get("compose_project") not in managed
            ]
            docker = {"checked": True, "available": True}
        except DockerError as e:
            docker = {"checked": True, "available": False, "error": f"{e.code}: {e.message}"}
            others = None

    profiles: list[dict[str, Any]] = []
    served: dict[int, dict[str, Any]] = {}
    for name in names:
        profile = load_profile(name)
        if not profile.compose_file().exists():
            known = ", ".join(all_names) or "none"
            _warn(
                warnings, "stack/profile-missing", f"no local stack named '{name}'",
                detail=f"Known profiles: {known}.", subject=name, path=_tilde(profile.dir()),
                fix=f"honcho start --profile {name}",
            )
            profiles.append({"name": name, "exists": False})
            continue
        entry = _profile_entry(profile, mode, docker, warnings)
        profiles.append(entry)
        port = profile.api_port
        clash = served.get(port)
        if clash is not None:
            loser = entry if clash["state"] == "running" else clash
            winner = clash if loser is entry else entry
            _warn(
                warnings, "stack/port-clash",
                f"declares API port {port}, which '{winner['name']}' also uses",
                detail="`honcho start` moves a stack to a free port when its own is taken.",
                subject=loser["name"], path=f"{loser['dir']}/profile.json", severity="info",
            )
            if entry["state"] == "running":
                served[port] = entry
        else:
            served[port] = entry

    # Static localhost URLs go stale when `honcho start` lands on a new port.
    for label, shown, url, port in _loopback_refs(raw, config.base_url):
        subject = _owner(label, integrations)
        hit = served.get(port)
        if hit is None:
            if mode != "offline" and api_healthy(_origin(url), timeout=1.0):
                continue  # something else serves it — a dev server, a repo compose stack
            _warn(
                warnings, "stack/stale-url", f"{label} → localhost:{port}, but no local stack uses that port",
                detail=f"{label} is {shown}. Local stacks: "
                + (", ".join(f"{e['name']} :{e['ports']['api']}" for e in served.values()) or "none"),
                subject=subject, path=label,
                fix=f"`honcho status` lists stacks and ports; point {label} at a running one",
            )
        elif mode != "offline" and hit["state"] != "running":
            _warn(
                warnings, "stack/stale-url", f"{label} → localhost:{port}, but '{hit['name']}' is {hit['state']}",
                subject=subject, path=label, fix=f"honcho start --profile {hit['name']}",
            )
    return {
        "profiles_dir": _tilde(profiles_dir()),
        "docker": docker,
        "profiles": profiles,
        "other_containers": others,
    }


def _profile_entry(
    profile: LocalProfile, mode: str, docker: dict[str, Any], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": profile.name,
        "exists": True,
        "dir": _tilde(profile.dir()),
        "state": "not checked",
        "api_url": profile.base_url,
        "ports": {"api": profile.api_port, "db": profile.db_port, "redis": profile.redis_port},
        "image": profile.image,
        "image_version": None,
        "server_version": None,
        "services": None,
        "profile_json": _read_json(profile.profile_file()),
    }
    if mode != "offline" and docker.get("available"):
        try:
            ps = compose_ps(profile, timeout=DOCKER_TIMEOUT)
            services = services_running(ps)
            entry["services"] = services
            if services:
                api_container = f"{profile.project_name}-api-1"
                versions = image_versions([api_container], timeout=DOCKER_TIMEOUT)
                entry["image_version"] = versions.get(api_container)
            healthy = stack_containers_up(ps) and api_healthy(profile.base_url)
            entry["state"] = "running" if healthy else ("unhealthy" if services else "stopped")
        except DockerError as e:
            entry["state"] = "unknown"
            entry["services"] = {"error": f"{e.code}: {e.message}"}
        if entry["state"] == "unhealthy":
            _warn(
                warnings, "stack/unhealthy", "containers are up but the API is not responding",
                detail=", ".join(f"{k} {v}" for k, v in (entry["services"] or {}).items()),
                subject=profile.name, path=profile.base_url,
                fix=f"docker compose -p {profile.project_name} logs api",
            )
    elif mode != "offline":
        entry["state"] = "unknown"
    if mode == "deep" and entry["state"] == "running":
        entry["server_version"] = _server_version(profile.base_url)

    raw_env = read_env_file(profile.env_file())
    # Only NAME=value lines are shown. A pasted base64 secret ("abc…xyz==")
    # also parses as a key; it is counted, never printed.
    env = {
        k: v for k, v in raw_env.items()
        if _ENV_NAME.match(k) and not re.fullmatch(r"=+", v) and ("_" in k or len(k) < 20)
    }
    entry["env"] = redact_tree(env) if profile.env_file().exists() else None
    if len(env) != len(raw_env):
        entry["env_unparsed_lines"] = len(raw_env) - len(env)
    toml_raw, toml_error = _load_toml(profile.config_file())
    entry["config_toml"] = {"error": toml_error} if toml_error else redact_tree(toml_raw)

    transports = {v.lower() for k, v in env.items() if k.upper().endswith("TRANSPORT") and v}
    transports |= _toml_transports(toml_raw)
    env_path = _tilde(profile.env_file())
    for provider, key_name in _PROVIDER_KEYS.items():
        if key_name not in env:
            continue
        if is_placeholder_key(env[key_name]):
            _warn(
                warnings, "stack/placeholder-key", f"{key_name} is a placeholder",
                subject=profile.name, path=f"{env_path} {key_name}",
                fix=f"set a real key there, or `honcho start --profile {profile.name} --setup basic`",
            )
        elif transports and not any(provider in t for t in transports):
            _warn(
                warnings, "stack/unused-key", f"{key_name} is set but no model uses {provider}",
                subject=profile.name, path=f"{env_path} {key_name}", severity="info",
            )
    return entry


def _integration_keys(
    integrations: dict[str, Any],
    keys: list[dict[str, Any]],
    raw: dict | None,
    config: CLIConfig,
    mode: str,
    warnings: list[dict[str, Any]],
) -> None:
    """Describe the Honcho keys integrations hold; under ``deep``, test each one.

    An integration having its own key is normal — OpenClaw keeps its own
    config, and a scoped key per integration is good practice — so a key that
    merely differs from the CLI's is reported, not flagged. What breaks an
    integration is a key the API refuses, typically one copied at install time
    and later rotated; only that is a warning. A key is only ever sent to the
    API that integration itself uses; when that can't be determined it is not
    tested at all.
    """
    known = _known_credentials(raw, config)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for k in keys:
        entry = integrations.get(k["group"], {}).get(k["name"])
        if entry is None:
            continue
        value, target = k["value"], k.get("target")
        marker = secret_marker(value)
        blocks = entry.get("host_blocks") or []
        # Prefer the integration's own hosts.<block> key, then any known key.
        candidates = sorted(known, key=lambda kv: not (kv[0].startswith("hosts.") and kv[0].split(".")[1] in blocks))
        matches = next((label for label, known_value in candidates if known_value == value), None)
        record: dict[str, Any] = {
            "where": k["where"],
            "kind": marker.get("kind"),
            "masked": marker.get("masked"),
            "fingerprint": marker.get("fingerprint"),
            "matches": matches,
            "status": "not checked",
            "accepted": None,
        }
        if k.get("format_error"):
            record.update(status="malformed", detail=k["format_error"])
            _warn(
                warnings, "integration/auth-format",
                "MCP Authorization header is missing 'Bearer '",
                detail="The hosted MCP server only accepts `Authorization: Bearer <key>`, "
                "so every request from this client fails.",
                subject=k["name"], path=k["where"], fix=f"set {k['where']} to 'Bearer <your key>'",
            )
        elif mode == "deep" and target is None:
            record.update(status="untested", detail="the entry doesn't name the API it uses")
        elif mode == "deep":
            if (target, value) not in results:
                results[(target, value)] = _probe_key(target, value, record["masked"])
            outcome = results[(target, value)]
            record.update(
                status=outcome["status"], detail=outcome["detail"], checked_against=safe_url(target),
                accepted={"accepted": True, "scoped": True, "rejected": False}.get(outcome["status"]),
            )
            _key_finding(warnings, k, record, outcome, target)
        entry.setdefault("keys", []).append(record)

    local_cli = _loopback_port(config.base_url) is not None
    for name, entry in integrations.get("detected", {}).items():
        mcp = entry.get("mcp")
        if local_cli and isinstance(mcp, dict):
            for field in ("url", "serverUrl", "uri"):
                url = mcp.get(field)
                if isinstance(url, str) and _host(url) == "mcp.honcho.dev":
                    _warn(
                        warnings, "integration/hosted-mcp",
                        "MCP points at hosted Honcho while the CLI is local",
                        detail=f"{mcp.get('path')} {field} = {url}; CLI base_url {safe_url(config.base_url)}",
                        subject=name, path=mcp.get("path"), severity="info",
                    )


def _known_credentials(raw: dict | None, config: CLIConfig) -> list[tuple[str, str]]:
    """``(label, value)`` for every key the CLI and the shared file know, best match first."""
    known: list[tuple[str, str]] = []
    active = config.resolved_api_key()
    if active:
        known.append(("the CLI's key", active))
    env_key = os.environ.get("HONCHO_API_KEY")
    if env_key:
        known.append(("HONCHO_API_KEY", env_key))
    if raw:
        if isinstance(raw.get("apiKey"), str) and raw["apiKey"]:
            known.append(("config.json apiKey", raw["apiKey"]))
        hosts = raw.get("hosts") if isinstance(raw.get("hosts"), dict) else {}

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "apiKey" and isinstance(v, str) and v:
                        known.append((f"{path}.apiKey", v))
                    else:
                        walk(v, f"{path}.{k}")

        for block, data in hosts.items():
            walk(data, f"hosts.{block}")
    return known


def _probe_key(target: str, key: str, masked: str | None) -> dict[str, Any]:
    """Ask ``target`` whether it accepts ``key``: one list-workspaces call.

    That route needs an admin key, so a valid scoped key is refused too — the
    server's message tells the two apart. Only the server's short message is
    kept, with the key itself scrubbed out of it.
    """
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.post(
                target.rstrip("/") + "/v3/workspaces/list",
                json={},
                headers={"Authorization": f"Bearer {key}", **identity_headers()},
            )
    except httpx.TimeoutException:
        return {"status": "unreachable", "detail": "timed out", "http": None}
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as e:
        return {"status": "unreachable", "detail": type(e).__name__, "http": None}
    try:
        message = str(response.json().get("detail", ""))
    except (ValueError, AttributeError):
        message = ""
    status = classify_key_response(response.status_code, message)
    # Show the server's words only when they are one of its fixed messages:
    # anything else could echo the key back.
    shown = message if message.strip().lower() in _KNOWN_AUTH_MESSAGES else f"HTTP {response.status_code}"
    return {"status": status, "detail": shown, "http": response.status_code}


_KNOWN_AUTH_MESSAGES = frozenset(
    m.lower()
    for m in (
        "Invalid API key",
        "Missing API key",
        "Invalid JWT",
        "JWT expired",
        "No access token provided",
        "Resource requires admin privileges",
        "JWT not permissioned for this resource",
        "Invalid JWT scope: peer/session token missing workspace",
    )
)


def classify_key_response(status_code: int, message: str) -> str:
    """accepted | scoped | expired | rejected | error, from a list-workspaces reply.

    Hosted Honcho says "Invalid API key" / "Missing API key"; a self-hosted
    server says "Invalid JWT" / "JWT expired" / "No access token provided".
    Both answer a valid non-admin key with "requires admin privileges" or
    "not permissioned" — a working key with a narrower scope.
    """
    if status_code == 200:
        return "accepted"
    lowered = message.lower()
    if status_code in (401, 403):
        if "admin privileges" in lowered or "not permissioned" in lowered:
            return "scoped"
        if "expired" in lowered:
            return "expired"
        if any(word in lowered for word in ("invalid", "missing", "no access token")):
            return "rejected"
    return "error"


def _key_finding(
    warnings: list[dict[str, Any]], k: dict[str, Any], record: dict[str, Any], outcome: dict[str, Any], target: str
) -> None:
    what = {"mcp": "MCP key", "plugin": "plugin key"}.get(k["section"], "key")
    host = _host(target)
    if outcome["status"] == "rejected":
        _warn(
            warnings, "integration/key-rejected", f"{what} rejected by {host}",
            detail=f"{record['masked'] or 'The key'} here was refused ({outcome['detail']}), so {k['name']} cannot reach Honcho.",
            subject=k["name"], path=k["where"],
            fix=_fix_for(k["name"], k["where"]),
        )
    elif outcome["status"] == "expired":
        _warn(warnings, "integration/key-expired", f"{what} has expired",
              detail="OAuth tokens expire hourly; the integration refreshes its own.",
              subject=k["name"], path=k["where"], severity="info")
    elif outcome["status"] == "unreachable":
        _warn(warnings, "integration/api-unreachable", f"couldn't reach {host} to test the {what}",
              detail=outcome["detail"], subject=k["name"], path=k["where"], severity="info")
    elif outcome["status"] == "error":
        _warn(warnings, "integration/key-unknown", f"{host} gave an unexpected answer for the {what}",
              detail=outcome["detail"], subject=k["name"], path=k["where"], severity="info")


def _host(url: str) -> str:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        host = None
    return host or "the configured URL"  # never echo an unparseable URL: it may hold a password


# --------------------------------------------------------------------------- #
# helpers


def _server_version(base_url: str, api_key: str = "") -> str | None:
    """``info.version`` from the API's OpenAPI document, if it serves one.

    Managed deployments gate the document behind auth, so the key is sent
    when there is one; the response is parsed for one field and dropped.
    A build that does not stamp its version reports ``unknown``: treated as
    no answer.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT) as client:
            response = client.get(base_url.rstrip("/") + "/openapi.json", headers=headers)
        info = response.json().get("info") if response.status_code == 200 else None
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None
    version = info.get("version") if isinstance(info, dict) else None
    # Only something shaped like a version: never echo arbitrary server text.
    if isinstance(version, str) and re.fullmatch(r"v?\d+(?:\.\d+){0,3}(?:[-+][\w.]{1,24})?", version):
        return version
    return None


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="seconds")


def _tilde(path: Path | str) -> str:
    text, home = str(path), str(Path.home())
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return redact_tree(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _load_toml(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, None
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh), None
    except (tomllib.TOMLDecodeError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"


def _toml_transports(obj: Any) -> set[str]:
    """Every ``transport`` value anywhere in a parsed config.toml."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() == "transport" and isinstance(v, str) and v:
                found.add(v.lower())
            else:
                found |= _toml_transports(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _toml_transports(v)
    return found


def _owner(label: str, integrations: dict[str, Any]) -> str | None:
    """The integration a ``hosts.<block>…`` config path belongs to."""
    if not label.startswith("hosts."):
        return None
    block = label.split(".")[1]
    for name, entry in integrations.get("detected", {}).items():
        if block in entry.get("host_blocks", []):
            return name
    return block


def _loopback_refs(raw: dict | None, resolved_base_url: str) -> list[tuple[str, str, str, int]]:
    """(label, masked url, raw url, port) for every loopback URL in config.json + the resolved one."""
    refs: list[tuple[str, str, str, int]] = []

    def visit(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, str) and not is_secret_path(path.split(".")):
            port = _loopback_port(obj)
            if port is not None:
                refs.append((path, _origin(obj), obj, port))

    if raw:
        visit(raw, "")
    port = _loopback_port(resolved_base_url)
    if port is not None and not any(r[2] == resolved_base_url for r in refs):
        refs.append(("resolved base_url", _origin(resolved_base_url), resolved_base_url, port))
    return refs


def _origin(url: str) -> str:
    """``scheme://host:port`` alone: no userinfo, path or query, so a probe
    against it can never carry a credential embedded in the configured URL."""
    parts = urlsplit(url)
    host = f"[{parts.hostname}]" if parts.hostname and ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}:{parts.port}" if parts.port else f"{parts.scheme}://{host}"


def _loopback_port(value: str) -> int | None:
    if not value.startswith(("http://", "https://")):
        return None
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if host not in _LOOPBACK:
        return None
    return port if port is not None else (443 if parts.scheme == "https" else 80)
