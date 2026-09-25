"""Docker daemon + Compose helpers for the local stack."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from honcho_cli.local import STACK_SERVICES
from honcho_cli.local.profile import LocalProfile
from honcho_cli.output import print_error

_DAEMON_DOWN_MARKERS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "failed to connect to the docker api",
    "error during connect",
)
_COMPOSE_MISSING_MARKERS = (
    "'compose' is not a docker command",
    "unknown command: compose",
    "docker: unknown command",
)
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


_CONFIG_PATHS = ("/app/config.toml.example", "/app/config.toml")
_CONFIG_HEADER = (
    "# Copied from {image} by honcho start. This file is not overwritten on later starts.\n"
    "# Secrets belong in .env (environment variables win over this file).\n\n"
)


def image_is_digest(ref: str) -> bool:
    """True when ``ref`` is already pinned to a content digest."""
    return "@sha256:" in ref.lower()


def image_repository(ref: str) -> str:
    """Strip a tag or digest from a Docker image reference."""
    if "@" in ref:
        return ref.split("@", 1)[0]
    last_slash = ref.rfind("/")
    last_colon = ref.rfind(":")
    if last_colon > last_slash:
        return ref[:last_colon]
    return ref


def pin_image(image: str) -> str:
    """Pull ``image`` if needed and return a digest-pinned reference.

    ``ghcr.io/plastic-labs/honcho:latest`` becomes
    ``ghcr.io/plastic-labs/honcho@sha256:...`` so the profile does not
    float when ``:latest`` moves. Already-pinned refs are left alone.
    """
    if image_is_digest(image):
        if not _image_exists(image):
            _pull(image)
        return image
    _pull(image)
    digest = _repo_digest(image)
    if not digest:
        raise DockerError(
            "IMAGE_PIN_FAILED",
            f"Pulled {image} but could not resolve a registry digest to pin.",
            {"image": image},
        )
    return digest


def seed_config_toml(profile: LocalProfile) -> bool:
    """Copy the image's ``config.toml.example`` into the profile if missing.

    Returns True when a file was written. Never overwrites an existing
    ``config.toml``.
    """
    dest = profile.config_file()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False
    copied = _copy_from_image(profile.image, _CONFIG_PATHS)
    if copied is None:
        raise DockerError(
            "CONFIG_MISSING",
            f"Could not copy config.toml from {profile.image}.",
            {"image": profile.image},
        )
    dest.write_text(_CONFIG_HEADER.format(image=profile.image) + copied)
    return True


def compose_up(
    profile: LocalProfile,
    *,
    recreate: tuple[str, ...] = (),
) -> None:
    """``docker compose up -d``. Compose output goes to stderr.

    ``recreate`` names services to ``--force-recreate`` (used after ``--setup``
    on an already-running stack so new ``.env`` values take effect).
    """
    args = ["up", "-d"]
    if recreate:
        args.extend(["--force-recreate", *recreate])
    _run_compose(profile, args)


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


def _unavailable(proc: subprocess.CompletedProcess[str]) -> DockerError | None:
    """Map a failed docker/compose process to a user-facing error, if obvious."""
    text = f"{proc.stderr or ''}{proc.stdout or ''}"
    lower = text.lower()
    if any(marker in lower for marker in _DAEMON_DOWN_MARKERS):
        return DockerError(
            "DOCKER_NOT_RUNNING",
            "Docker is installed but the daemon is not running. Start it and retry.",
        )
    if any(marker in lower for marker in _COMPOSE_MISSING_MARKERS):
        return DockerError(
            "DOCKER_COMPOSE_MISSING",
            "Honcho start requires Docker Compose v2 (the `docker compose` plugin).",
        )
    if any(marker in text for marker in _CRED_HELPER_MARKERS):
        return DockerError(
            "DOCKER_CREDENTIALS",
            "Docker could not read registry credentials "
            "(docker-credential-desktop is not on PATH). "
            "Quit and reopen your terminal, or add Docker Desktop's bin "
            "directory to PATH, then retry.",
            {"exit_code": proc.returncode},
        )
    return None


def _run_compose(
    profile: LocalProfile,
    args: list[str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = compose_argv(profile) + args
    cwd: Path = profile.dir()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
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
    if proc.returncode != 0:
        classified = _unavailable(proc)
        if classified is not None:
            raise classified
    if check and proc.returncode != 0:
        raise DockerError(
            "COMPOSE_FAILED",
            "docker compose failed. See output above, or run `docker compose -p "
            f"{profile.project_name} logs`.",
            {"project": profile.project_name, "exit_code": proc.returncode},
        )
    return proc


def _run_docker(
    args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise DockerError(
            "DOCKER_NOT_INSTALLED",
            "Docker is not installed. Install Docker Desktop (or another Compose-v2 runtime) and retry.",
        ) from e
    except OSError as e:
        raise DockerError("DOCKER_FAILED", str(e), {"command": args}) from e
    if proc.returncode == 0:
        return proc
    classified = _unavailable(proc)
    if classified is not None:
        raise classified
    if check:
        raise DockerError(
            "DOCKER_FAILED",
            f"docker {' '.join(args)} failed.",
            {
                "exit_code": proc.returncode,
                "stderr": (proc.stderr or "")[-500:],
            },
        )
    return proc


def _pull(image: str) -> None:
    proc = _run_docker(["pull", image], check=False)
    if proc.stdout:
        sys.stderr.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise DockerError(
            "IMAGE_PULL_FAILED",
            f"Failed to pull {image}.",
            {"image": image, "exit_code": proc.returncode},
        )


def _image_exists(image: str) -> bool:
    return _run_docker(["image", "inspect", image], check=False).returncode == 0


def _repo_digest(image: str) -> str | None:
    proc = _run_docker(
        ["image", "inspect", "--format", "{{json .RepoDigests}}", image],
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        digests = json.loads((proc.stdout or "").strip() or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(digests, list):
        return None
    repo = image_repository(image)
    for item in digests:
        if isinstance(item, str) and item.startswith(repo + "@"):
            return item
    for item in digests:
        if isinstance(item, str) and "@sha256:" in item:
            return item
    return None


def _copy_from_image(image: str, paths: tuple[str, ...]) -> str | None:
    """Create a stopped container and copy the first path that exists."""
    name = f"honcho-seed-{os.getpid()}-{time.time_ns()}"
    created = _run_docker(["create", "--name", name, image], check=False)
    if created.returncode != 0:
        cid = (created.stdout or "").strip() or name
        _run_docker(["rm", "-f", cid], check=False)
        return None
    cid = (created.stdout or "").strip() or name
    try:
        with tempfile.TemporaryDirectory(prefix="honcho-cfg-") as tmp:
            dest = Path(tmp) / "config.toml"
            for path in paths:
                if dest.exists():
                    dest.unlink()
                copied = _run_docker(["cp", f"{cid}:{path}", str(dest)], check=False)
                if copied.returncode == 0 and dest.exists():
                    return dest.read_text(encoding="utf-8")
    finally:
        _run_docker(["rm", "-f", cid], check=False)
    return None


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
