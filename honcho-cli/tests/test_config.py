"""Tests for config management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from honcho_cli.config import CLIConfig


class TestCLIConfig:
    def test_defaults(self):
        config = CLIConfig()
        assert config.base_url == "https://api.honcho.dev"
        assert config.api_key == ""
        assert config.workspace_id == ""

    def test_env_override(self):
        with patch.dict(os.environ, {"HONCHO_API_KEY": "test-key", "HONCHO_BASE_URL": "http://localhost:8000"}):
            config = CLIConfig.load()
            assert config.api_key == "test-key"
            assert config.base_url == "http://localhost:8000"

    def test_save_and_load(self, tmp_path):
        config_file = tmp_path / "config.toml"
        # Clear env vars so they don't override file values
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("HONCHO_")}
        with patch("honcho_cli.config.CONFIG_FILE", config_file), patch("honcho_cli.config.CONFIG_DIR", tmp_path), patch.dict(os.environ, clean_env, clear=True):
            config = CLIConfig(
                base_url="http://localhost:8000",
                api_key="test-key-123",
                workspace_id="my-ws",
            )
            config.save()

            loaded = CLIConfig.load()
            assert loaded.base_url == "http://localhost:8000"
            assert loaded.api_key == "test-key-123"
            assert loaded.workspace_id == "my-ws"

    def test_redacted(self):
        config = CLIConfig(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef")
        redacted = config.redacted()
        assert "eyJhbGci" in redacted["api_key"]
        assert "cdef" in redacted["api_key"]
        assert "..." in redacted["api_key"]

    def test_redacted_short_key(self):
        config = CLIConfig(api_key="short")
        redacted = config.redacted()
        assert redacted["api_key"] == "***"

    def test_redacted_empty_key(self):
        config = CLIConfig(api_key="")
        redacted = config.redacted()
        assert redacted["api_key"] == ""
