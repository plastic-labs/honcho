"""CLI contracts for `honcho start` / `stop` / `status`."""

from __future__ import annotations

import json
import os

import pytest
from honcho_cli.local.docker import image_is_digest, image_repository
from honcho_cli.main import app
from typer.testing import CliRunner


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.json"
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("honcho_cli.config.CONFIG_FILE", f)
    monkeypatch.setattr("honcho_cli.commands.setup.CONFIG_FILE", f)
    for k in [k for k in os.environ if k.startswith(("HONCHO_", "LLM_"))]:
        monkeypatch.delenv(k)
    return f


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _host_ports_free(monkeypatch):
    monkeypatch.setattr("honcho_cli.local.docker.port_available", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _stub_image_pin(monkeypatch):
    def fake_pin(image: str) -> str:
        if image_is_digest(image):
            return image
        return f"{image_repository(image)}@sha256:cafedeadbeef"

    monkeypatch.setattr("honcho_cli.commands.stack.pin_image", fake_pin)
    monkeypatch.setattr("honcho_cli.commands.stack.seed_config_toml", lambda profile: False)


_PS = [
    {"Service": "api", "State": "running", "Health": "healthy"},
    {"Service": "deriver", "State": "running"},
    {"Service": "database", "State": "running", "Health": "healthy"},
    {"Service": "redis", "State": "running", "Health": "healthy"},
]


def test_start_does_not_rewrite_environment_url(cfg, runner, monkeypatch):
    cfg.write_text(
        json.dumps({"apiKey": "k", "environmentUrl": "https://api.honcho.dev"})
    )
    monkeypatch.setattr("honcho_cli.commands.stack.stack_healthy", lambda profile: False)
    monkeypatch.setattr("honcho_cli.commands.stack.compose_up", lambda profile, **k: None)
    monkeypatch.setattr("honcho_cli.commands.stack.wait_for_health", lambda *a, **k: True)
    monkeypatch.setattr("honcho_cli.commands.stack.compose_ps", lambda profile: _PS)
    monkeypatch.setenv("LLM_OPENAI_API_KEY", "sk-test")
    result = runner.invoke(app, ["start", "--json"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["endpoints"]["api"] == "http://127.0.0.1:8000"
    assert payload["image"].endswith("@sha256:cafedeadbeef")
    on_disk = json.loads(cfg.read_text())
    assert on_disk["environmentUrl"] == "https://api.honcho.dev"


def test_start_requires_llm_key(cfg, runner, monkeypatch):
    monkeypatch.setattr("honcho_cli.commands.stack.stack_healthy", lambda profile: False)
    result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "MISSING_LLM_KEY"


def test_stop_already_stopped_skips_down(cfg, runner, tmp_path, monkeypatch):
    compose = tmp_path / "profiles" / "local" / "docker-compose.yml"
    compose.parent.mkdir(parents=True)
    compose.write_text("services: {}\n")
    monkeypatch.setattr("honcho_cli.commands.stack.compose_ps", lambda profile: [])
    down = []
    monkeypatch.setattr(
        "honcho_cli.commands.stack.compose_down",
        lambda profile, wipe=False: down.append(wipe),
    )
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0, result.stderr
    assert down == []
    assert json.loads(result.stdout)["status"] == "stopped"


def test_status_lists_profiles_or_one(cfg, runner, tmp_path, monkeypatch):
    for name, port in (("demo", 8001), ("local", 8000)):
        d = tmp_path / "profiles" / name
        d.mkdir(parents=True)
        (d / "docker-compose.yml").write_text("services: {}\n")
        (d / "profile.json").write_text(json.dumps({"apiPort": port}) + "\n")

    monkeypatch.setattr(
        "honcho_cli.commands.stack.compose_ps",
        lambda profile: _PS if profile.name == "local" else [],
    )
    monkeypatch.setattr(
        "honcho_cli.commands.stack.stack_healthy",
        lambda profile: profile.name == "local",
    )
    listed = runner.invoke(app, ["status"])
    assert listed.exit_code == 0, listed.stderr
    rows = json.loads(listed.stdout)["profiles"]
    by_name = {row["profile"]: row for row in rows}
    assert by_name["local"]["status"] == "running"
    assert by_name["demo"]["endpoints"]["api"] == "http://127.0.0.1:8001"

    one = runner.invoke(app, ["status", "--profile", "local"])
    assert one.exit_code == 0, one.stderr
    payload = json.loads(one.stdout)
    assert payload["profile"] == "local"
    assert "profiles" not in payload


def test_start_setup_requires_tty(cfg, runner):
    result = runner.invoke(app, ["start", "--setup", "basic", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["error"]["code"] == "SETUP_REQUIRES_TTY"


def test_start_setup_recreates_when_already_running(cfg, runner, monkeypatch):
    from honcho_cli.local.setup import SetupAnswers

    ups: list[tuple[str, ...]] = []
    monkeypatch.setattr("honcho_cli.commands.stack.use_json", lambda: False)
    monkeypatch.setattr("honcho_cli.commands.stack.stack_healthy", lambda profile: True)
    monkeypatch.setattr(
        "honcho_cli.commands.stack.compose_up",
        lambda profile, **k: ups.append(k.get("recreate", ())),
    )
    monkeypatch.setattr("honcho_cli.commands.stack.wait_for_health", lambda *a, **k: True)
    monkeypatch.setattr("honcho_cli.commands.stack.compose_ps", lambda profile: _PS)
    monkeypatch.setattr(
        "honcho_cli.commands.stack.run_setup",
        lambda mode, path, config_path=None: SetupAnswers(
            mode="basic",
            provider="openai",
            api_key="sk-wiz",
            chat_model="gpt-test",
        ),
    )
    pins: list[str] = []
    monkeypatch.setattr(
        "honcho_cli.commands.stack.pin_image",
        lambda image: pins.append(image) or image,
    )
    result = runner.invoke(app, ["start", "--setup", "basic"])
    assert result.exit_code == 0, result.stderr
    assert pins == []
    assert ups == [("api", "deriver")]
