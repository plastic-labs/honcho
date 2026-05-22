from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from src.config import ModelConfig
from src.llm import registry
from src.llm.backends.codex import CodexResponsesBackend
from src.llm.codex_oauth import CodexOAuthCredentials
from src.llm.types import ProviderClient


def test_codex_oauth_client_bypasses_default_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_client = object()
    codex_client = object()

    def fake_resolve(**kwargs: Any) -> CodexOAuthCredentials:
        assert kwargs["auth_path"] == "/tmp/auth.json"
        return CodexOAuthCredentials(
            access_token="oauth-access-token",
            base_url="https://chatgpt.com/backend-api/codex",
            default_headers={"originator": "codex_cli_rs"},
            auth_path=Path("/tmp/auth.json"),
        )

    def fake_client(
        base_url: str,
        api_key: str,
        default_headers: tuple[tuple[str, str], ...],
    ) -> object:
        assert base_url == "https://chatgpt.com/backend-api/codex"
        assert api_key == "oauth-access-token"
        assert default_headers == (("originator", "codex_cli_rs"),)
        return codex_client

    monkeypatch.setattr(registry, "resolve_codex_oauth_credentials", fake_resolve)
    monkeypatch.setattr(registry, "get_codex_oauth_client", fake_client)
    monkeypatch.setitem(registry.CLIENTS, "openai", default_client)

    client = registry.client_for_model_config(
        "openai",
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
            codex_auth_path="/tmp/auth.json",
        ),
    )

    assert client is codex_client


def test_codex_oauth_uses_responses_backend() -> None:
    backend = registry.backend_for_provider(
        "openai",
        cast(ProviderClient, object()),
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
        ),
    )

    assert isinstance(backend, CodexResponsesBackend)
