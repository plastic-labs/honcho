"""Docker daemon + Compose helpers for the local stack."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from honcho_cli.local import STACK_SERVICES
from honcho_cli.local.profile import LocalProfile
from honcho_cli.output import print_error

_INFO_TIMEOUT = 8
_DOCKER_DESKTOP_WAIT = 60

# Docker Desktop puts credential helpers here; GUI-installed Docker often
# leaves them off PATH for venv / IDE terminals.
_DARWIN_DOCKER_BIN = Path("/Applications/Docker.app/Contents/Resources/bin")
_CRED_HELPER_MARKERS = ("error getting credentials", "docker-credential-desktop")


class DockerError(Exception):
    """Docker is missing, the daemon is down, or a Compose command failed."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def exit(self) -> None:
        print_error(self.code, self.message, self.details or None)
        raise SystemExit(1)


def ensure_docker(*, wait: float = _DOCKER_DESKTOP_WAIT) -> None:
    """Require Docker Compose v2 with a running daemon.

    On macOS, if the daemon is down, tries to launch Docker Desktop and polls
    until ``docker info`` succeeds or ``wait`` seconds elapse.
    """
    if shutil.which("docker") is None:
        raise DockerError(
            "DOCKER_NOT_INSTALLED",
            "Docker is not installed. Install Docker Desktop (or another Compose-v2 runtime) and retry.",
        )
    if _docker_info_ok():
        _require_compose_v2()
        return
    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Docker"], check=False, capture_output=True)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if _docker_info_ok():
                _require_compose_v2()
                return
            time.sleep(2)
    raise DockerError(
        "DOCKER_NOT_RUNNING",
        "Docker is installed but the daemon is not running. Start Docker Desktop and retry.",
    )


def port_available(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing is accepting connections on ``host:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def allocate_host_ports(
    profile: LocalProfile,
    *,
    pinned: frozenset[str] = frozenset(),
) -> tuple[LocalProfile, dict[str, tuple[int, int]]]:
    """Move api/db/redis host ports that are already bound.

    Names in ``pinned`` (``api`` / ``database`` / ``redis``) were set by a
    flag and fail instead of moving.
    """
    taken: set[int] = set()
    remapped: dict[str, tuple[int, int]] = {}
    chosen: dict[str, int] = {}
    for name, field, flag in (
        ("api", "api_port", "--api-port"),
        ("database", "db_port", "--db-port"),
        ("redis", "redis_port", "--redis-port"),
    ):
        preferred = getattr(profile, field)
        port = preferred
        if name in pinned:
            if preferred in taken or not port_available(preferred):
                raise DockerError(
                    "PORT_IN_USE",
                    f"Host port {preferred} for {name} is already in use. "
                    f"Pass {flag} with a free port, or stop the other process.",
                    {"port": preferred, "service": name, "flag": flag},
                )
        else:
            while port in taken or not port_available(port):
                port += 1
                if port > preferred + 100:
                    raise DockerError(
                        "PORT_IN_USE",
                        f"Could not find a free host port near {preferred}.",
                        {"preferred": preferred},
                    )
            if port != preferred:
                remapped[name] = (preferred, port)
        taken.add(port)
        chosen[field] = port
    return profile.overlay(**chosen), remapped


def compose_argv(profile: LocalProfile) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(profile.compose_file()),
        "--project-directory",
        str(profile.dir()),
        "-p",
        profile.project_name,
    ]


def compose_up(profile: LocalProfile) -> None:
    """``docker compose up -d``. Compose output goes to stderr.

    If a Docker Desktop credential helper is missing from PATH, retries
    once with ``credsStore`` stripped so public GHCR/Hub pulls can proceed.
    """
    proc = _run_compose(profile, ["up", "-d"], check=False)
    if proc.returncode == 0:
        return
    if _looks_like_cred_helper_error(proc):
        config_dir = _docker_config_without_creds_store()
        if config_dir is not None:
            try:
                retry = _run_compose(
                    profile,
                    ["up", "-d"],
                    check=False,
                    extra_env={"DOCKER_CONFIG": str(config_dir)},
                )
            finally:
                shutil.rmtree(config_dir, ignore_errors=True)
            if retry.returncode == 0:
                return
            proc = retry
        raise DockerError(
            "DOCKER_CREDENTIALS",
            "Docker could not read registry credentials "
            "(docker-credential-desktop is not on PATH). "
            "Quit and reopen your terminal, or add Docker Desktop's bin "
            "directory to PATH, then retry.",
            {"project": profile.project_name, "exit_code": proc.returncode},
        )
    raise DockerError(
        "COMPOSE_FAILED",
        "docker compose failed. See output above, or run `docker compose -p "
        f"{profile.project_name} logs`.",
        {"project": profile.project_name, "exit_code": proc.returncode},
    )


def compose_down(profile: LocalProfile, *, wipe: bool = False) -> None:
    args = ["down"]
    if wipe:
        args.append("-v")
    _run_compose(profile, args, capture=False)


def compose_ps(profile: LocalProfile) -> list[dict]:
    """Parsed ``docker compose ps --format json`` (array or NDJSON)."""
    proc = _run_compose(profile, ["ps", "--format", "json"], capture=True, check=False)
    if proc.returncode != 0:
        return []
    return _parse_ps(proc.stdout or "")


def services_running(ps: list[dict]) -> dict[str, str]:
    """Map service name → state for the four stack services.

    State is ``running``, ``healthy``, ``exited``, etc. Prefer Docker's
    Health field when present.
    """
    out: dict[str, str] = {}
    for row in ps:
        service = str(row.get("Service") or row.get("Name") or "")
        # "honcho-local-api-1" → try Service first; fall back to suffix match
        if service not in STACK_SERVICES:
            for name in STACK_SERVICES:
                if (
                    service == name
                    or service.endswith(f"-{name}-1")
                    or f"_{name}_" in service
                ):
                    service = name
                    break
            else:
                continue
        health = str(row.get("Health") or "").lower()
        state = str(row.get("State") or row.get("Status") or "").lower()
        if health:
            out[service] = health
        elif "health" in state:
            # e.g. "running (healthy)"
            out[service] = state
        else:
            out[service] = state or "unknown"
    return out


def stack_containers_up(ps: list[dict]) -> bool:
    """True when all four services are running (deriver has no healthcheck)."""
    states = services_running(ps)
    if any(name not in states for name in STACK_SERVICES):
        return False
    for state in states.values():
        if "exit" in state or state in {"dead", "paused"}:
            return False
        if "running" not in state and "healthy" not in state:
            return False
    return True


def _helper_path_dirs() -> list[str]:
    """Directories that commonly contain ``docker-credential-*`` helpers."""
    candidates = [
        _DARWIN_DOCKER_BIN,
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
        Path.home() / ".docker" / "bin",
        Path("/opt/docker-desktop/bin"),
    ]
    return [str(p) for p in candidates if p.is_dir()]


def docker_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """``os.environ`` plus Docker Desktop helper dirs prepended to PATH."""
    env = os.environ.copy()
    extras = _helper_path_dirs()
    if extras:
        env["PATH"] = os.pathsep.join([*extras, env.get("PATH", "")])
    if extra:
        env.update(extra)
    return env


def _looks_like_cred_helper_error(proc: subprocess.CompletedProcess[str]) -> bool:
    text = f"{proc.stderr or ''}{proc.stdout or ''}"
    return any(marker in text for marker in _CRED_HELPER_MARKERS)


def _docker_user_config() -> Path:
    return Path.home() / ".docker" / "config.json"


def _docker_config_without_creds_store() -> Path | None:
    """Copy ``~/.docker/config.json`` minus credsStore/credHelpers.

    Public pulls (GHCR, Docker Hub) do not need the Desktop helper. Returns
    None when there is nothing to strip.
    """
    src = _docker_user_config()
    if not src.exists():
        return None
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if "credsStore" not in data and "credHelpers" not in data:
        return None
    data = dict(data)
    data.pop("credsStore", None)
    data.pop("credHelpers", None)
    tmp = Path(tempfile.mkdtemp(prefix="honcho-docker-"))
    (tmp / "config.json").write_text(json.dumps(data, indent=2) + "\n")
    return tmp


def _docker_info_ok() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=_INFO_TIMEOUT,
            env=docker_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _require_compose_v2() -> None:
    try:
        proc = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=_INFO_TIMEOUT,
            env=docker_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise DockerError(
            "DOCKER_COMPOSE_MISSING",
            "Honcho start requires Docker Compose v2 (the `docker compose` plugin).",
            {"error": str(e)},
        ) from e
    if proc.returncode != 0:
        raise DockerError(
            "DOCKER_COMPOSE_MISSING",
            "Honcho start requires Docker Compose v2 (the `docker compose` plugin).",
        )


def _run_compose(
    profile: LocalProfile,
    args: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = compose_argv(profile) + args
    cwd: Path = profile.dir()
    env = docker_env(extra_env)
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    except FileNotFoundError as e:
        raise DockerError(
            "DOCKER_NOT_INSTALLED",
            "Docker is not installed. Install Docker Desktop (or another Compose-v2 runtime) and retry.",
        ) from e
    except OSError as e:
        raise DockerError("COMPOSE_FAILED", str(e), {"command": cmd}) from e
    if not capture:
        if proc.stdout:
            sys.stderr.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
    if check and proc.returncode != 0:
        raise DockerError(
            "COMPOSE_FAILED",
            "docker compose failed. See output above, or run `docker compose -p "
            f"{profile.project_name} logs`.",
            {"project": profile.project_name, "exit_code": proc.returncode},
        )
    return proc


def _parse_ps(stdout: str) -> list[dict]:
    text = stdout.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
