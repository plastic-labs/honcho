"""Run a traced local Honcho stack: API + deriver exporting to OTel Desktop Viewer.

    uv run python -m scripts.local_tracing            # mock LLM provider, no spend
    uv run python -m scripts.local_tracing --llm env  # your .env provider config

Starts, in one foreground process (Ctrl-C stops everything it started):

- OTel Desktop Viewer (unless one is already listening on the OTLP port),
- the mock LLM/embedding provider (`--llm mock`, the default),
- the API and the deriver with TRACING_ENABLED=true and SAMPLE_RATE=1.0.

`--llm mock` runs against a throwaway database (created, migrated, dropped on
exit) on the Postgres in DB_CONNECTION_URI. `--llm env` uses your normal .env
configuration, database included, and only adds the tracing settings on top.

The viewer is a separate binary:

    go install github.com/ctrlspice/otel-desktop-viewer@v0.6.0

Query traces from a terminal or an agent with its CLI (`otel-desktop-viewer
skills` prints the agent guide), e.g. `otel-desktop-viewer traces --json`, or
look up the trace id an API response returns in `X-Honcho-Trace-Id`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.engine.url import make_url

from scripts.run_ts_sdk_tests import (
    DEFAULT_DB_URI,
    PROJECT_ROOT,
    _admin_exec,  # pyright: ignore[reportPrivateUsage]
    _free_port,  # pyright: ignore[reportPrivateUsage]
    _server_env,  # pyright: ignore[reportPrivateUsage]
    _stop,  # pyright: ignore[reportPrivateUsage]
    _wait_for,  # pyright: ignore[reportPrivateUsage]
)

LOG_DIR = PROJECT_ROOT / ".cache" / "local-tracing"


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _tracing_env(otlp_port: int) -> dict[str, str]:
    return {
        "TRACING_ENABLED": "true",
        "TRACING_EXPORTER": "otlp",
        "TRACING_OTLP_ENDPOINT": f"http://127.0.0.1:{otlp_port}",
        "TRACING_SAMPLE_RATE": "1.0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a traced local Honcho stack (see module docstring)."
    )
    parser.add_argument("--llm", choices=["mock", "env"], default="mock")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--otlp-port", type=int, default=4318)
    parser.add_argument("--viewer-port", type=int, default=8001)
    parser.add_argument(
        "--viewer", default="otel-desktop-viewer", help="viewer binary path"
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen[bytes]] = []
    db_to_drop: tuple[str, str] | None = None

    def spawn(
        name: str, cmd: list[str], env: dict[str, str]
    ) -> subprocess.Popen[bytes]:
        log = (LOG_DIR / f"{name}.log").open("ab")
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=log)
        procs.append(proc)
        print(f"  {name:<9} pid {proc.pid}, log {LOG_DIR / f'{name}.log'}")
        return proc

    try:
        if _listening(args.otlp_port):
            print(f"Reusing the viewer already listening on :{args.otlp_port}")
        else:
            viewer: str = shutil.which(str(args.viewer)) or str(args.viewer)
            if not Path(viewer).exists():
                print(f"Viewer binary not found: {args.viewer} (see module docstring)")
                return 1
            proc = spawn(
                "viewer",
                [
                    viewer,
                    "--open-browser=false",
                    "--http",
                    str(args.otlp_port),
                    "--grpc",
                    str(_free_port()),
                    "--browser-port",
                    str(args.viewer_port),
                ],
                dict(os.environ),
            )
            _wait_for(f"http://127.0.0.1:{args.viewer_port}", proc, "viewer")

        if args.llm == "mock":
            base_uri = (
                dotenv_values(PROJECT_ROOT / ".env").get("DB_CONNECTION_URI")
                or os.environ.get("DB_CONNECTION_URI")
                or DEFAULT_DB_URI
            )
            db_name = (
                f"local_tracing_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
            )
            db_uri = (
                make_url(base_uri)
                .set(database=db_name)
                .render_as_string(hide_password=False)
            )
            mock_port = _free_port()
            env = _server_env(db_uri, f"http://127.0.0.1:{mock_port}/v1")
            # Derive each message as it arrives instead of waiting for a full
            # batch, so its worker trace shows up right away.
            env["DERIVER_FLUSH_ENABLED"] = "true"
            _admin_exec(base_uri, f'CREATE DATABASE "{db_name}"')
            db_to_drop = (base_uri, db_name)
        else:
            mock_port = None
            env = dict(os.environ)
        env |= _tracing_env(args.otlp_port)

        subprocess.run(
            [sys.executable, "-m", "src.migrate"], cwd=PROJECT_ROOT, env=env, check=True
        )
        uvicorn = [sys.executable, "-m", "uvicorn", "--host", "127.0.0.1"]
        if mock_port is not None:
            mock = spawn(
                "mock-llm",
                [*uvicorn, "--port", str(mock_port), "src.mock_provider.main:app"],
                env,
            )
            _wait_for(f"http://127.0.0.1:{mock_port}/docs", mock, "mock provider")
        api = spawn(
            "api", [*uvicorn, "--port", str(args.api_port), "src.main:app"], env
        )
        spawn("deriver", [sys.executable, "-m", "src.deriver"], env)
        _wait_for(f"http://127.0.0.1:{args.api_port}/health", api, "Honcho API")

        print(
            f"\nHoncho API  http://127.0.0.1:{args.api_port}"
            + f"\nViewer UI   http://127.0.0.1:{args.viewer_port}"
            + f"\nOTLP        http://127.0.0.1:{args.otlp_port}"
            + f"\nQuery       {args.viewer} traces --endpoint "
            + f"http://127.0.0.1:{args.viewer_port} --json"
            + "\nCtrl-C to stop.\n"
        )
        try:
            while all(proc.poll() is None for proc in procs):
                time.sleep(1)
            print("A service exited; see its log. Stopping.")
            return 1
        except KeyboardInterrupt:
            return 0
    finally:
        for proc in reversed(procs):
            _stop(proc)
        if db_to_drop is not None:
            _admin_exec(
                db_to_drop[0], f'DROP DATABASE IF EXISTS "{db_to_drop[1]}" WITH (FORCE)'
            )


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    sys.exit(main())
