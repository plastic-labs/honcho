"""Native lifecycle tests use real supervised processes and isolated profiles."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from honcho_cli.local import native
from honcho_cli.local.profile import LocalProfile, list_profile_names, load_profile
from honcho_cli.local.supervisor import control, read_state
from honcho_cli.main import app
from typer.testing import CliRunner


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "test-version"\n')
    (root / "src/runtime.py").write_text("""
import os, signal, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if sys.argv[1] == "api":
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
    HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Health).serve_forever()
elif sys.argv[1] == "deriver":
    if os.environ.get("TEST_WORKER_FAIL"):
        time.sleep(float(os.environ["TEST_WORKER_FAIL"]))
        sys.exit(7)
    while True:
        time.sleep(0.1)
elif sys.argv[1] == "migrate":
    sys.exit(3)
""")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setenv(
        "DB_CONNECTION_URI", "postgresql://secret:password@localhost:1234/test"
    )
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path / "profiles-root")
    monkeypatch.setattr(
        native,
        "runtime_identity",
        lambda path: {
            "source": str(path),
            "python": sys.executable,
            "version": "test-version",
            "commit": "test-commit",
        },
    )
    return root


@pytest.fixture
def profile(checkout):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    profile = LocalProfile("native-test", api_port=port)
    yield profile
    native.stop(load_profile(profile.name))


def launch(profile, checkout, **options):
    return native.start(
        profile,
        source=checkout,
        dependencies="external",
        timeout=5,
        migrate=False,
        api_workers=1,
        **options,
    )


def test_native_lifecycle_identity_and_private_metadata(checkout, profile):
    result = launch(profile, checkout)
    assert result["status"] == "running"
    assert result["runtime"]["source"] == str(checkout)
    assert result["runtime"]["commit"] == "test-commit"
    assert result["services"] == {"api": "running", "deriver": "running"}
    assert result["endpoints"]["postgres"] == "postgresql://localhost:1234/test"
    assert "password" not in (profile.dir() / "native.json").read_text()
    assert (profile.dir().stat().st_mode & 0o777) == 0o700
    assert load_profile(profile.name).backend == "native"
    assert list_profile_names() == [profile.name]
    again = launch(profile, checkout)
    assert again["instance"] == result["instance"]
    stopped = native.stop(profile)
    assert stopped["status"] == "stopped"
    assert all(p["state"] == "exited" for p in stopped["processes"].values())
    assert stopped["exitCode"] == 0
    assert not control(read_state(profile.dir()))
    assert native.port_available(profile.api_port)


def test_first_child_failure_stops_sibling(checkout, profile, monkeypatch):
    monkeypatch.setenv("TEST_WORKER_FAIL", "1.5")
    launch(profile, checkout)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = read_state(profile.dir())
        if state["status"] == "stopped":
            break
        time.sleep(0.05)
    assert state["exitCode"] == 7
    assert all(service["state"] == "exited" for service in state["services"].values())
    assert native.port_available(profile.api_port)


def test_start_failure_cleans_up(checkout, profile, monkeypatch):
    monkeypatch.setenv("TEST_WORKER_FAIL", "0.01")
    monkeypatch.setattr(native, "api_healthy", lambda *args: False)
    with pytest.raises(native.NativeError, match="exited during startup"):
        launch(profile, checkout)
    assert not control(read_state(profile.dir()))
    assert native.port_available(profile.api_port)


def test_stale_pid_cannot_signal_another_process(checkout, profile):
    profile.dir().mkdir(parents=True)
    (profile.dir() / "native.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "running",
                "instance": "stale",
                "socket": "/tmp/honcho-no-such-socket",
                "dependencies": "external",
            }
        )
    )
    with pytest.raises(native.NativeError, match="supervisor is unreachable"):
        native.stop(profile)
    with pytest.raises(native.NativeError, match="supervisor is unreachable"):
        launch(profile, checkout)
    state = read_state(profile.dir())
    state["status"] = "stopped"
    (profile.dir() / "native.json").write_text(json.dumps(state))


def test_config_precedence_and_no_source_mutation(checkout, profile, monkeypatch):
    (checkout / "config.toml").write_text(
        '[db]\nconnection_uri = "source-toml"\n[deriver]\nworkers = 2\n'
    )
    (checkout / ".env").write_text('DB_CONNECTION_URI="source-dotenv"\n')
    profile.dir().mkdir(parents=True)
    profile.config_file().write_text("[deriver]\nworkers = 3\n")
    profile.env_file().write_text('DB_CONNECTION_URI="profile-dotenv"\n')
    monkeypatch.setenv("DERIVER_WORKERS", "4")
    env = native.environment(profile, checkout)
    assert env["DB_CONNECTION_URI"].endswith("localhost:1234/test")
    assert env["DERIVER_WORKERS"] == "4"
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert env["HONCHO_CONFIG_TOML_DISABLED"] == "1"
    assert "source-toml" in (checkout / "config.toml").read_text()


def test_cli_native_start_status_stop(checkout, profile):
    runner = CliRunner()
    started = runner.invoke(
        app,
        [
            "start",
            "--backend",
            "native",
            "--source",
            str(checkout),
            "--profile",
            profile.name,
            "--api-port",
            str(profile.api_port),
            "--json",
        ],
    )
    assert started.exit_code == 0, started.stderr
    assert json.loads(started.stdout)["backend"] == "native"
    status = runner.invoke(app, ["status", "--profile", profile.name, "--json"])
    assert status.exit_code == 0, status.stderr
    assert json.loads(status.stdout)["runtime"]["version"] == "test-version"
    stopped = runner.invoke(app, ["stop", "--profile", profile.name, "--json"])
    assert stopped.exit_code == 0, stopped.stderr
    assert json.loads(stopped.stdout)["status"] == "stopped"


def test_native_import_does_not_load_server():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import honcho_cli.local.native; assert not any(m == 'src' or m.startswith(('src.', 'sqlalchemy', 'fastapi')) for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_migration_failure_never_starts_services(checkout, profile):
    with pytest.raises(native.NativeError, match="startup command failed"):
        native.start(
            profile,
            source=checkout,
            dependencies="external",
            timeout=5,
            migrate=True,
            api_workers=1,
        )
    assert not control(read_state(profile.dir()))
    assert native.port_available(profile.api_port)


def test_owned_dependencies_reject_nested_external_database(
    checkout, profile, monkeypatch
):
    monkeypatch.setenv(
        "DB", '{"CONNECTION_URI":"postgresql://external.example/production"}'
    )
    monkeypatch.setattr(native, "allocate_host_ports", lambda p, **kwargs: (p, {}))
    with pytest.raises(native.NativeError, match="owns the database/cache endpoints"):
        native.start(
            profile,
            source=checkout,
            dependencies="docker",
            timeout=5,
            migrate=False,
            api_workers=1,
        )
    assert not profile.profile_file().exists()
