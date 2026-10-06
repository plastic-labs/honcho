"""Mock presets must override nested credentials and remain portable to Compose."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from honcho_cli.local.docker import stack_containers_up
from honcho_cli.local.env import read_env_file, render_stack
from honcho_cli.local.mock import VALIDATE, environment
from honcho_cli.local.profile import LocalProfile


def hostile_settings() -> dict[str, str]:
    return {
        "LLM_OPENAI_API_KEY": "live-secret",
        "DERIVER_MODEL_CONFIG": json.dumps(
            {
                "transport": "anthropic",
                "model": "live-model",
                "fallback": {"transport": "gemini", "model": "live-fallback"},
            }
        ),
        "DERIVER_MODEL_CONFIG__OVERRIDES__API_KEY_ENV": "LIVE_KEY",
        "DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://live.example.com",
        "DERIVER_WORKERS": "3",
        "DIALECTIC_LEVELS__high__MODEL_CONFIG__FALLBACK__OVERRIDES__API_KEY": "live-secret",
        "DIALECTIC_LEVELS__minimal__MAX_TOOL_ITERATIONS": "2",
        "EMBEDDING_MODEL_CONFIG__TRANSPORT": "gemini",
        "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://live.example.com",
        "DREAM": json.dumps(
            {
                "DEDUCTION_MODEL_CONFIG": {
                    "transport": "anthropic",
                    "model": "live-model",
                }
            }
        ),
        "TELEMETRY_ENABLED": "true",
        "TELEMETRY": json.dumps(
            {"ENABLED": True, "ENDPOINT": "https://billing.example.com"}
        ),
        "SENTRY_ENABLED": "true",
        "LANGFUSE_PUBLIC_KEY": "live-secret",
    }


def test_effective_settings_cannot_escape_mock_overrides():
    env = environment({**os.environ, **hostile_settings()}, "http://127.0.0.1:8916/v1")
    code = (
        VALIDATE
        + "\nassert settings.DERIVER.WORKERS == 3\nassert settings.DIALECTIC.LEVELS['minimal'].MAX_TOOL_ITERATIONS == 2\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not any(key.startswith("DERIVER_MODEL_CONFIG") for key in env)
    for section in ("LLM", "DERIVER", "DREAM", "EMBEDDING", "DIALECTIC", "TELEMETRY"):
        assert "live-secret" not in env[section]
        assert "live.example.com" not in env[section]


def test_compose_mock_preserves_live_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("honcho_cli.config.CONFIG_DIR", tmp_path)
    profile = LocalProfile("mock-test", providers="mock")
    profile.dir().mkdir(parents=True)
    profile.config_file().write_text(
        '[deriver]\nworkers = 4\n[deriver.model_config]\ntransport = "anthropic"\nmodel = "live-model"\n'
    )
    render_stack(profile, extra=hostile_settings())
    live = profile.env_file().read_text()
    assert "live-secret" in live
    mock = read_env_file(profile.dir() / ".mock.env")
    assert "live-secret" not in mock["LLM"]
    assert json.loads(mock["DERIVER"])["WORKERS"] == 3
    compose = profile.compose_file().read_text()
    assert "mock-provider:" in compose
    assert "condition: service_completed_successfully" in compose
    assert "path: .mock.env" in compose
    assert "path: .env" not in compose
    # Returning to live mode removes the services and uses the original env.
    render_stack(profile.overlay(providers="live"))
    assert "mock-provider:" not in profile.compose_file().read_text()
    assert "path: .env" in profile.compose_file().read_text()
    assert "live-secret" in profile.env_file().read_text()


def test_mock_config_exit_does_not_mask_api_failure():
    rows = [
        {"Service": name, "State": "running"}
        for name in ("api", "deriver", "database", "redis", "mock-provider")
    ]
    rows.append({"Service": "mock-config", "State": "exited"})
    assert stack_containers_up(rows, mock=True)
    rows[0]["State"] = "exited"
    assert not stack_containers_up(rows, mock=True)
    assert not stack_containers_up(rows[:4], mock=True)
