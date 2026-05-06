"""Tests for self-hosted configuration — BASE_URL, contact info, cloud deps stripping."""

import os

import pytest


class TestSelfHostConfig:
    """Configuration defaults for self-hosted deployments."""

    def test_base_url_defaults_to_localhost(self):
        """BASE_URL defaults to localhost:8000 for self-hosted."""
        from src.config import AppSettings
        settings = AppSettings()
        assert settings.BASE_URL == "http://localhost:8000"

    def test_contact_email_defaults_to_empty(self):
        """CONTACT_EMAIL is empty by default (self-hosters fill it in)."""
        from src.config import AppSettings
        settings = AppSettings()
        assert settings.CONTACT_EMAIL == ""

    def test_telemetry_disabled_by_default(self):
        """CloudEvents telemetry is disabled by default for self-hosted."""
        from src.config import TelemetrySettings
        telemetry = TelemetrySettings()
        assert telemetry.ENABLED is False
        assert telemetry.ENDPOINT is None

    def test_sentry_disabled_by_default(self):
        """Sentry error tracking is disabled by default."""
        from src.config import SentrySettings
        sentry = SentrySettings()
        assert sentry.ENABLED is False

    def test_auth_disabled_by_default(self):
        """Authentication is off by default (self-hosters enable it)."""
        from src.config import AuthSettings
        auth = AuthSettings()
        assert auth.USE_AUTH is False

    def test_base_url_from_env(self, monkeypatch):
        """BASE_URL can be configured via environment variable."""
        monkeypatch.setenv("BASE_URL", "https://honcho.example.com")
        from src.config import AppSettings
        settings = AppSettings()
        assert settings.BASE_URL == "https://honcho.example.com"

    def test_contact_email_from_env(self, monkeypatch):
        """CONTACT_EMAIL can be configured via environment variable."""
        monkeypatch.setenv("CONTACT_EMAIL", "admin@example.com")
        from src.config import AppSettings
        settings = AppSettings()
        assert settings.CONTACT_EMAIL == "admin@example.com"


class TestOpenAPIMetadata:
    """OpenAPI docs reflect self-hosted configuration."""

    async def test_openapi_uses_configured_base_url(self, client, monkeypatch):
        """OpenAPI servers list uses the configured BASE_URL."""
        monkeypatch.setenv("BASE_URL", "https://custom.example.com")
        # Reload app with new config
        from src.config import settings as app_settings
        app_settings.BASE_URL = "https://custom.example.com"

        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        servers = data.get("servers", [])
        urls = [s["url"] for s in servers]
        assert "https://custom.example.com" in urls
        assert "http://localhost:8000" in urls

    async def test_openapi_contact_reflects_config(self, client, monkeypatch):
        """OpenAPI contact info uses configured values."""
        from src.config import settings as app_settings
        app_settings.CONTACT_EMAIL = "test@test.com"

        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        contact = data.get("info", {}).get("contact", {})
        assert contact.get("email") == "test@test.com"


class TestEmitterDisabledByDefault:
    """Telemetry emitter is disabled when no endpoint is configured."""

    def test_emitter_disabled_without_endpoint(self):
        """Emitter with no endpoint is disabled."""
        from src.telemetry.emitter import TelemetryEmitter
        emitter = TelemetryEmitter(endpoint=None)
        assert emitter.enabled is False

    def test_emitter_enabled_with_endpoint(self):
        """Emitter with an endpoint is enabled."""
        from src.telemetry.emitter import TelemetryEmitter
        emitter = TelemetryEmitter(endpoint="https://example.com/events")
        assert emitter.enabled is True

    def test_emit_noop_when_disabled(self):
        """emit() is a no-op when emitter is disabled."""
        from src.telemetry.emitter import TelemetryEmitter
        emitter = TelemetryEmitter(endpoint=None)

        class FakeEvent:
            def category(self):
                return "test"

        # Should not raise
        emitter.emit(FakeEvent())
        assert emitter.buffer_size == 0
