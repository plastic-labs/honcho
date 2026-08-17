"""CLI tests for `honcho start` / `stop` / `status`."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from honcho_cli.local.docker import DockerError
from honcho_cli.main import app
from typer.testing import CliRunner


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.commands.setup.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith("HONCHO_")]:
        monkeypatch.delenv(k)
    return f


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _host_ports_free(monkeypatch):
    """Don't let a local Redis/Postgres steal default ports during CLI tests."""
    monkeypatch.setattr("honcho_cli.local.docker.port_available", lambda *a, **k: True)


_PS = [
    {"Service": "api", "State": "running", "Health": "healthy"},
    {"Service": "deriver", "State": "running"},
    {"Service": "database", "State": "running", "Health": "healthy"},
    {"Service": "redis", "State": "running", "Health": "healthy"},
]


class TestStartErrors:
    def test_inference_local_rejected(self, cfg, runner):
        result = runner.invoke(
            app, ["start", "--inference", "local", "--llm-api-key", "sk-test"]
        )
        assert result.exit_code == 1
        err = json.loads(result.stderr)
        assert err["error"]["code"] == "INFERENCE_UNSUPPORTED"

    def test_inference_hybrid_rejected(self, cfg, runner):
        result = runner.invoke(
            app, ["start", "--inference", "hybrid", "--llm-api-key", "sk-test"]
        )
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INFERENCE_UNSUPPORTED"

    def test_invalid_profile_name(self, cfg, runner):
        result = runner.invoke(
            app, ["start", "--profile", "../etc", "--llm-api-key", "sk-test"]
        )
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_PROFILE"

    def test_docker_not_installed(self, cfg, runner):
        err = DockerError("DOCKER_NOT_INSTALLED", "Docker is not installed.")
        with patch("honcho_cli.commands.stack.ensure_docker", side_effect=err):
            result = runner.invoke(app, ["start", "--llm-api-key", "sk-test"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "DOCKER_NOT_INSTALLED"

    def test_missing_llm_key_noninteractive(self, cfg, runner):
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
        ):
            result = runner.invoke(app, ["start"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "MISSING_LLM_KEY"


class TestStartHappy:
    def test_json_endpoints_and_does_not_mutate_config(self, cfg, runner):
        cfg.write_text(
            json.dumps({"apiKey": "k", "environmentUrl": "https://api.honcho.dev"})
        )
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
            patch("honcho_cli.commands.stack.compose_up") as up,
            patch("honcho_cli.commands.stack.wait_for_health", return_value=True),
            patch("honcho_cli.commands.stack.compose_ps", return_value=_PS),
        ):
            result = runner.invoke(app, ["start", "--llm-api-key", "sk-test", "--json"])
        assert result.exit_code == 0, result.stderr
        up.assert_called_once()
        payload = json.loads(result.stdout)
        assert payload["status"] == "running"
        assert payload["profile"] == "local"
        assert payload["inference"] == "cloud"
        assert payload["endpoints"]["api"] == "http://127.0.0.1:8000"
        assert payload["endpoints"]["docs"].endswith("/docs")
        assert "HONCHO_BASE_URL=http://127.0.0.1:8000" in payload["hint"]
        on_disk = json.loads(cfg.read_text())
        assert on_disk["environmentUrl"] == "https://api.honcho.dev"
        assert on_disk["apiKey"] == "k"

    def test_idempotent_skips_compose_up(self, cfg, runner):
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=True),
            patch("honcho_cli.commands.stack.compose_up") as up,
            patch("honcho_cli.commands.stack.compose_ps", return_value=_PS),
        ):
            result = runner.invoke(app, ["start", "--llm-api-key", "sk-test"])
        assert result.exit_code == 0, result.stderr
        up.assert_not_called()
        payload = json.loads(result.stdout)
        assert payload["status"] == "running"

    def test_health_timeout(self, cfg, runner):
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
            patch("honcho_cli.commands.stack.compose_up"),
            patch("honcho_cli.commands.stack.wait_for_health", return_value=False),
        ):
            result = runner.invoke(app, ["start", "--llm-api-key", "sk-test"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "HEALTH_TIMEOUT"

    def test_custom_api_port_in_endpoints(self, cfg, runner):
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
            patch("honcho_cli.commands.stack.compose_up"),
            patch("honcho_cli.commands.stack.wait_for_health", return_value=True),
            patch("honcho_cli.commands.stack.compose_ps", return_value=_PS),
        ):
            result = runner.invoke(
                app, ["start", "--llm-api-key", "sk-test", "--api-port", "8001"]
            )
        payload = json.loads(result.stdout)
        assert payload["endpoints"]["api"] == "http://127.0.0.1:8001"

    def test_remaps_busy_redis_port(self, cfg, runner, monkeypatch):
        monkeypatch.setattr(
            "honcho_cli.local.docker.port_available",
            lambda port, host="127.0.0.1": port != 6379,
        )
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
            patch("honcho_cli.commands.stack.compose_up"),
            patch("honcho_cli.commands.stack.wait_for_health", return_value=True),
            patch("honcho_cli.commands.stack.compose_ps", return_value=_PS),
        ):
            result = runner.invoke(app, ["start", "--llm-api-key", "sk-test"])
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert ":6379/" not in payload["endpoints"]["redis"]
        assert payload["endpoints"]["redis"].startswith("redis://127.0.0.1:")

    def test_pinned_busy_port_errors(self, cfg, runner, monkeypatch):
        monkeypatch.setattr(
            "honcho_cli.local.docker.port_available",
            lambda port, host="127.0.0.1": port != 6379,
        )
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
        ):
            result = runner.invoke(
                app, ["start", "--llm-api-key", "sk-test", "--redis-port", "6379"]
            )
        assert result.exit_code == 1
        err = json.loads(result.stderr)
        assert err["error"]["code"] == "PORT_IN_USE"
        assert err["error"]["details"]["flag"] == "--redis-port"


class TestStop:
    def test_stop_missing_stack_is_ok(self, cfg, runner):
        result = runner.invoke(app, ["stop"])
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "stopped"

    def test_stop_wipe_passes_wipe_flag(self, cfg, runner, tmp_path):
        compose = tmp_path / "profiles" / "local" / "docker-compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text("services: {}\n")
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.compose_down") as down,
        ):
            result = runner.invoke(app, ["stop", "--wipe"])
        assert result.exit_code == 0, result.stderr
        down.assert_called_once()
        assert down.call_args.kwargs["wipe"] is True
        assert json.loads(result.stdout)["status"] == "wiped"


class TestStatus:
    def test_status_not_found(self, cfg, runner):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert json.loads(result.stderr)["error"]["code"] == "STACK_NOT_FOUND"

    def test_status_running(self, cfg, runner, tmp_path):
        compose = tmp_path / "profiles" / "local" / "docker-compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text("services: {}\n")
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.compose_ps", return_value=_PS),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=True),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == "running"
        assert payload["endpoints"]["api"] == "http://127.0.0.1:8000"

    def test_status_stopped_exits_nonzero(self, cfg, runner, tmp_path):
        compose = tmp_path / "profiles" / "local" / "docker-compose.yml"
        compose.parent.mkdir(parents=True)
        compose.write_text("services: {}\n")
        with (
            patch("honcho_cli.commands.stack.ensure_docker"),
            patch("honcho_cli.commands.stack.compose_ps", return_value=[]),
            patch("honcho_cli.commands.stack.stack_healthy", return_value=False),
        ):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert json.loads(result.stdout)["status"] == "stopped"
