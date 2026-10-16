"""Run an explicit Honcho checkout with the CLI's Python interpreter."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from honcho_cli.local.docker import (
    DockerError,
    allocate_host_ports,
    compose_down,
    compose_up,
    port_available,
)
from honcho_cli.local.env import read_env_file
from honcho_cli.local.health import api_healthy
from honcho_cli.local.profile import LocalProfile, save_profile
from honcho_cli.local.supervisor import control, read_state, write_state


class NativeError(DockerError):
    """An invalid launch request or a failed local process."""


@contextlib.contextmanager
def lifecycle_lock(profile: LocalProfile):
    if os.name != "posix":
        raise NativeError(
            "NATIVE_UNSUPPORTED",
            "Native startup currently supports macOS and Linux. Use --backend compose on this platform.",
        )
    import fcntl

    profile.dir().mkdir(parents=True, exist_ok=True)
    profile.dir().chmod(0o700)
    with (profile.dir() / "native.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NativeError(
                "STACK_BUSY", "Another start or stop is using this profile."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def checkout(source: Path | None) -> Path:
    candidate = (source or Path.cwd()).expanduser().resolve()
    for root in [candidate] if source else [candidate, *candidate.parents]:
        if (root / "src/runtime.py").is_file() and (root / "pyproject.toml").is_file():
            return root
    raise NativeError(
        "RUNTIME_NOT_FOUND",
        "Native startup needs a Honcho checkout containing src/runtime.py. Pass --source /path/to/honcho and run its `uv run --all-packages honcho start --backend native`.",
    )


def runtime_identity(root: Path) -> dict[str, Any]:
    if sys.version_info < (3, 13):
        raise NativeError(
            "RUNTIME_PYTHON",
            "The server requires Python 3.13+. Run `uv run --all-packages honcho start --backend native` from the Honcho checkout.",
        )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, json; print(json.dumps([m for m in ('fastapi', 'psycopg', 'sqlalchemy') if importlib.util.find_spec(m) is None]))",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    if json.loads(probe.stdout):
        raise NativeError(
            "RUNTIME_NOT_INSTALLED",
            "Server dependencies are missing from this interpreter. Run `uv sync --all-packages` in the checkout and invoke its CLI with `uv run --all-packages honcho`. ",
        )
    with (root / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "source": str(root),
        "python": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "version": version,
        "commit": commit,
        "dirty": dirty,
        "mode": "processes",
    }


def _toml_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as file:
            sections = tomllib.load(file)
    except (OSError, ValueError) as exc:
        raise NativeError("INVALID_CONFIG", f"Cannot read {path}: {exc}") from exc
    result = {}
    for section, values in sections.items():
        if not isinstance(values, dict):
            raise NativeError(
                "INVALID_CONFIG", f"Expected a settings table for {section} in {path}."
            )
        prefix = "" if section.lower() == "app" else section.upper() + "_"
        for key, value in values.items():
            result[prefix + key.upper()] = (
                value if isinstance(value, str) else json.dumps(value)
            )
    return result


def environment(profile: LocalProfile, root: Path) -> dict[str, str]:
    # Freeze config into the child environment. src.config's legacy dotenv
    # override=True must not undo profile/host overrides after the launcher chdir.
    result = {
        **_toml_env(root / "config.toml"),
        **_toml_env(profile.config_file()),
        **read_env_file(root / ".env"),
        **read_env_file(profile.env_file()),
        **os.environ,
    }
    result.update(PYTHON_DOTENV_DISABLED="1", HONCHO_CONFIG_TOML_DISABLED="1")
    return result


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _dependencies(profile: LocalProfile) -> None:
    # Compose accepts JSON. This project owns only its Postgres and Redis.
    services = {
        "database": {
            "image": "pgvector/pgvector:pg15",
            "ports": [f"127.0.0.1:{profile.db_port}:5432"],
            "environment": {
                "POSTGRES_PASSWORD": "postgres",
                "POSTGRES_USER": "postgres",
                "POSTGRES_DB": "postgres",
            },
            "volumes": ["pgdata:/var/lib/postgresql/data"],
            "healthcheck": {
                "test": ["CMD-SHELL", "pg_isready -U postgres -d postgres"],
                "interval": "1s",
                "timeout": "5s",
                "retries": 60,
            },
        },
        "redis": {
            "image": "redis:8.2",
            "ports": [f"127.0.0.1:{profile.redis_port}:6379"],
            "volumes": ["redis-data:/data"],
            "healthcheck": {
                "test": ["CMD", "redis-cli", "ping"],
                "interval": "1s",
                "timeout": "5s",
                "retries": 60,
            },
        },
    }
    profile.compose_file().write_text(
        json.dumps(
            {"services": services, "volumes": {"pgdata": {}, "redis-data": {}}},
            indent=2,
        )
        + "\n"
    )
    compose_up(profile, services=("database", "redis"), wait=True)


def payload(profile: LocalProfile) -> dict[str, Any]:
    saved = read_state(profile.dir())
    live = control(saved)
    state = live or saved
    healthy = bool(live and api_healthy(profile.base_url))
    inactive = "exited" if state.get("status") == "stopped" else "unknown"
    services = {
        name: detail["state"] if live else inactive
        for name, detail in state.get("services", {}).items()
    }
    return {
        "profile": profile.name,
        "backend": "native",
        "providers": state.get("providers", "live"),
        "status": "running"
        if healthy
        else "unhealthy"
        if live
        else "stopped"
        if not saved or saved.get("status") == "stopped"
        else "unreachable",
        "runtime": state.get("runtime", {}),
        "instance": state.get("instance"),
        "dependencies": state.get("dependencies", "external"),
        "endpoints": state.get(
            "endpoints",
            {
                "api": profile.base_url,
                "docs": profile.base_url + "/docs",
                "postgres": "external",
                "redis": "external",
            },
        ),
        "services": services,
        "processes": {
            name: {**detail, "state": services[name]}
            for name, detail in state.get("services", {}).items()
        },
        "logs": str(profile.dir()),
        "exitCode": state.get("exitCode"),
        "hint": f"HONCHO_BASE_URL={profile.base_url} honcho workspace list",
    }


def _stop_processes(profile: LocalProfile, *, timeout: float = 35) -> None:
    state = read_state(profile.dir())
    if not control(state, "stop"):
        if state.get("status") in {"starting", "running"}:
            raise NativeError(
                "OWNERSHIP_UNVERIFIED",
                f"The supervisor is unreachable. No processes or containers were stopped. Inspect logs in {profile.dir()}.",
            )
        return  # Never trust a PID in a stale manifest.
    deadline = time.monotonic() + timeout
    while read_state(profile.dir()).get("status") != "stopped":
        if time.monotonic() >= deadline:
            raise NativeError(
                "STOP_TIMEOUT",
                f"Native processes are still draining. See {profile.dir()}.",
            )
        time.sleep(0.1)


def stop(profile: LocalProfile, *, wipe: bool = False) -> dict[str, Any]:
    with lifecycle_lock(profile):
        state = read_state(profile.dir())
        _stop_processes(profile)
        if state.get("dependencies") == "docker" and profile.compose_file().exists():
            compose_down(profile, wipe=wipe)
        elif wipe:
            raise NativeError(
                "EXTERNAL_DEPENDENCIES",
                "This profile uses external dependencies; it owns no database volumes to wipe.",
            )
        return payload(profile)


def start(
    profile: LocalProfile,
    *,
    source: Path | None,
    dependencies: str,
    timeout: float,
    migrate: bool,
    api_workers: int,
    providers: str = "live",
    pinned: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    with lifecycle_lock(profile):
        old = read_state(profile.dir())
        if control(old):
            if (
                (
                    source is not None
                    and str(checkout(source)) != old["runtime"]["source"]
                )
                or dependencies != old["dependencies"]
                or profile.base_url != old["endpoints"]["api"]
                or api_workers != old["runtime"].get("apiWorkers", 1)
                or migrate
                or providers != old.get("providers", "live")
            ):
                raise NativeError(
                    "STACK_RUNNING",
                    "This profile is already running. Stop it before changing its launch options.",
                )
            return payload(profile)
        if old.get("status") in {"running", "starting"}:
            raise NativeError(
                "OWNERSHIP_UNVERIFIED",
                f"The previous supervisor is unreachable. Inspect {profile.dir()} before starting another instance.",
            )
        root = checkout(source)
        identity = runtime_identity(root)
        identity["apiWorkers"] = api_workers
        if dependencies == "docker":
            profile, _ = allocate_host_ports(profile, pinned=pinned)
        elif not port_available(profile.api_port):
            raise NativeError(
                "PORT_IN_USE",
                f"API port {profile.api_port} is in use. Pass --api-port with a free port.",
            )
        env = environment(profile, root)
        if dependencies == "docker" and any(
            key.upper() in {"DB", "CACHE"}
            or key.upper().startswith(("DB__", "CACHE__"))
            for key in env
        ):
            raise NativeError(
                "CONFLICTING_DEPENDENCIES",
                "--dependencies docker owns the database/cache endpoints. Use flat DB_* and CACHE_* settings instead of DB/CACHE JSON or double-underscore overrides.",
            )
        if dependencies == "external" and not env.get("DB_CONNECTION_URI"):
            raise NativeError(
                "MISSING_DATABASE",
                "Set DB_CONNECTION_URI in the environment, checkout config, or profile; alternatively use --dependencies docker.",
            )
        if dependencies == "docker":
            env.update(
                DB_CONNECTION_URI=f"postgresql+psycopg://postgres:postgres@127.0.0.1:{profile.db_port}/postgres",
                CACHE_URL=f"redis://127.0.0.1:{profile.redis_port}/0",
                CACHE_ENABLED="true",
            )
        env["API_WORKERS"] = str(api_workers)
        mock_port = None
        if providers == "mock":
            if importlib.util.find_spec("honcho_mock_provider") is None:
                raise NativeError(
                    "MOCK_NOT_INSTALLED",
                    "Install the optional mock provider with `uv sync --all-packages` in the checkout, or install `honcho-cli[mock]` in your server environment.",
                )
            from honcho_cli.local.mock import VALIDATE
            from honcho_cli.local.mock import environment as mock_environment

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                mock_port = probe.getsockname()[1]
            env = mock_environment(env, f"http://127.0.0.1:{mock_port}/v1")
            validation = subprocess.run(
                [sys.executable, "-c", VALIDATE],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if validation.returncode:
                raise NativeError(
                    "INVALID_MOCK_CONFIG",
                    "The server rejected the mock provider configuration. Check your non-model settings and the checkout's supported configuration.",
                )
        profile = profile.overlay(backend="native", providers=providers)
        save_profile(profile)
        instance = uuid.uuid4().hex
        commands = {
            "api": [
                sys.executable,
                "-m",
                "src.runtime",
                "api",
                "--host",
                "127.0.0.1",
                "--port",
                str(profile.api_port),
            ],
            "deriver": [sys.executable, "-m", "src.runtime", "deriver"],
        }
        if mock_port:
            commands = {
                "mock-provider": [
                    sys.executable,
                    "-m",
                    "honcho_mock_provider",
                    "--port",
                    str(mock_port),
                ],
                **commands,
            }
        state: dict[str, Any] = {
            "instance": instance,
            "socket": f"/tmp/honcho-{os.getuid()}-{instance}.sock",
            "runtime": identity,
            "commands": commands,
            "status": "starting",
            "dependencies": dependencies,
            "providers": providers,
            "services": {},
            "endpoints": {
                "api": profile.base_url,
                "docs": profile.base_url + "/docs",
                "postgres": _redact_url(env["DB_CONNECTION_URI"]),
                "redis": _redact_url(env.get("CACHE_URL", "redis://127.0.0.1:6379/0"))
                if env.get("CACHE_ENABLED", "false").lower() == "true"
                else "disabled",
            },
        }
        if mock_port:
            state["endpoints"]["mock-provider"] = f"http://127.0.0.1:{mock_port}"
            state["healthChecks"] = {"mock-provider": f"http://127.0.0.1:{mock_port}"}
        write_state(profile.dir(), state)
        supervisor = None
        try:
            if dependencies == "docker":
                _dependencies(profile)
            if migrate or dependencies == "docker":
                with (profile.dir() / "migrate.log").open("ab") as log:
                    subprocess.run(
                        [sys.executable, "-m", "src.runtime", "migrate"],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout,
                    )
            with (profile.dir() / "supervisor.log").open("ab") as log:
                supervisor = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "honcho_cli.local.supervisor",
                        str(profile.dir()),
                        instance,
                    ],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if supervisor.poll() is not None:
                    raise NativeError(
                        "NATIVE_FAILED",
                        f"A native process exited during startup. See logs in {profile.dir()}.",
                    )
                if control(state) and api_healthy(profile.base_url):
                    return payload(profile)
                time.sleep(0.2)
            raise NativeError(
                "HEALTH_TIMEOUT",
                f"Native API did not become healthy within {timeout:g}s. See logs in {profile.dir()}.",
            )
        except BaseException as exc:
            if supervisor is not None and supervisor.poll() is None:
                supervisor.terminate()
                try:
                    supervisor.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    raise NativeError(
                        "STOP_TIMEOUT",
                        f"Startup failed and children are still draining. See {profile.dir()}.",
                    ) from None
            if dependencies == "docker":
                compose_down(profile)
            if supervisor is None:
                state.update(status="stopped", exitCode=1)
                write_state(profile.dir(), state)
            if isinstance(exc, subprocess.SubprocessError):
                raise NativeError(
                    "NATIVE_FAILED",
                    f"Native startup command failed. See logs in {profile.dir()}.",
                ) from exc
            raise
