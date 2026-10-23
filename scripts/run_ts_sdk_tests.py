"""Run the TypeScript SDK test suite against a real, throwaway Honcho server.

Creates a fresh database on the Postgres server named by ``DB_CONNECTION_URI``,
migrates it, starts the mock LLM provider and the Honcho API as subprocesses,
runs ``bun test`` in ``sdks/typescript`` with ``HONCHO_TEST_URL`` pointing at the
API, then tears everything down. No model provider or API key is needed: every
LLM and embedding call goes to ``src/mock_provider``.

    uv run python scripts/run_ts_sdk_tests.py [bun test args...]

or, from ``sdks/typescript``, ``bun run test``.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_PATH = PROJECT_ROOT / "sdks" / "typescript"
DEFAULT_DB_URI = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
STARTUP_TIMEOUT_SECONDS = 60


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(url: str, proc: subprocess.Popen[bytes], name: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310 - localhost only
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    raise RuntimeError(f"{name} did not become ready at {url}")


def _admin_exec(base_uri: str, statement: str) -> None:
    """Run one statement against the server's maintenance database."""
    engine = create_engine(
        make_url(base_uri).set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(statement)
    finally:
        engine.dispose()


def _server_env(db_uri: str, mock_url: str) -> dict[str, str]:
    """Honcho's settings for a no-auth, no-cache API backed by the mock provider."""
    env = {
        **os.environ,
        # src.config loads .env with override=True, which would put a developer's
        # real provider keys and base URLs back over the mock provider below.
        "PYTHON_DOTENV_DISABLED": "1",
        "DB_CONNECTION_URI": db_uri,
        "AUTH_USE_AUTH": "false",
        "CACHE_ENABLED": "false",
        "SENTRY_ENABLED": "false",
        "METRICS_ENABLED": "false",
        "TELEMETRY_ENABLED": "false",
        "LANGFUSE_PUBLIC_KEY": "",
        # Only truthiness is checked; keeps a stray module 401ing, not spending.
        "LLM_OPENAI_API_KEY": "mock-provider-not-a-key",
        "LLM_OPENAI_BASE_URL": mock_url,
        "EMBEDDING_MODEL_CONFIG__TRANSPORT": "openai",
        "EMBEDDING_MODEL_CONFIG__MODEL": "mock-embedding",
        "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": mock_url,
        "DERIVER_MODEL_CONFIG__TRANSPORT": "openai",
        "DERIVER_MODEL_CONFIG__MODEL": "mock",
        "SUMMARY_MODEL_CONFIG__TRANSPORT": "openai",
        "SUMMARY_MODEL_CONFIG__MODEL": "mock",
    }
    for level in ("minimal", "low", "medium", "high", "max"):
        env[f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG__TRANSPORT"] = "openai"
        env[f"DIALECTIC_LEVELS__{level}__MODEL_CONFIG__MODEL"] = "mock"
    return env


def _stop(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    # Same precedence as src.config, which loads .env over the environment.
    base_uri = (
        dotenv_values(PROJECT_ROOT / ".env").get("DB_CONNECTION_URI")
        or os.environ.get("DB_CONNECTION_URI")
        or DEFAULT_DB_URI
    )
    db_name = f"ts_sdk_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:4]}"
    db_uri = (
        make_url(base_uri).set(database=db_name).render_as_string(hide_password=False)
    )
    mock_port, api_port = _free_port(), _free_port()
    mock_url = f"http://127.0.0.1:{mock_port}/v1"
    api_url = f"http://127.0.0.1:{api_port}"
    env = _server_env(db_uri, mock_url)

    # Server output would bury bun's report; keep it in a file for failures.
    log_fd, log_path = tempfile.mkstemp(prefix="honcho-ts-sdk-", suffix=".log")
    returncode = 1
    mock_proc: subprocess.Popen[bytes] | None = None
    api_proc: subprocess.Popen[bytes] | None = None
    _admin_exec(base_uri, f'CREATE DATABASE "{db_name}"')
    try:
        subprocess.run(
            [sys.executable, "-m", "src.migrate"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_fd,
            stderr=log_fd,
            check=True,
        )
        uvicorn = [sys.executable, "-m", "uvicorn", "--host", "127.0.0.1"]
        uvicorn += ["--log-level", "warning"]
        mock_proc = subprocess.Popen(
            [*uvicorn, "--port", str(mock_port), "src.mock_provider.main:app"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_fd,
            stderr=log_fd,
        )
        api_proc = subprocess.Popen(
            [*uvicorn, "--port", str(api_port), "src.main:app"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_fd,
            stderr=log_fd,
        )
        _wait_for(f"http://127.0.0.1:{mock_port}/docs", mock_proc, "mock provider")
        _wait_for(f"{api_url}/health", api_proc, "Honcho API")

        result = subprocess.run(
            ["bun", "test", *sys.argv[1:]],
            cwd=SDK_PATH,
            env={**os.environ, "HONCHO_TEST_URL": api_url, "HONCHO_MAX_RETRIES": "0"},
        )
        returncode = result.returncode
        return returncode
    finally:
        _stop(api_proc)
        _stop(mock_proc)
        os.close(log_fd)
        _admin_exec(base_uri, f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        if returncode == 0:
            os.unlink(log_path)
        else:
            print(f"Server logs (migrations, mock provider, API): {log_path}")


if __name__ == "__main__":
    sys.exit(main())
