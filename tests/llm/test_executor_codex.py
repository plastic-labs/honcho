from __future__ import annotations

from typing import Any

import pytest

from src.config import ModelConfig
from src.llm import executor
from src.llm.backend import CompletionResult


@pytest.mark.asyncio
async def test_inner_call_resolves_codex_client_from_effective_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_client = object()
    codex_backend = object()

    async def fake_client_for_config(
        provider: str,
        config: ModelConfig,
    ) -> object:
        assert provider == "openai"
        assert config.auth_mode == "codex_oauth"
        return codex_client

    def fake_backend_for_provider(
        provider: str,
        client: object,
        config: ModelConfig,
    ) -> object:
        assert provider == "openai"
        assert client is codex_client
        assert config.auth_mode == "codex_oauth"
        return codex_backend

    async def fake_execute_completion(*args: Any, **kwargs: Any) -> CompletionResult:
        assert args[0] is codex_backend
        assert kwargs["max_tokens"] == 42
        return CompletionResult(content="ok", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(executor, "aclient_for_model_config", fake_client_for_config)
    monkeypatch.setattr(executor, "backend_for_provider", fake_backend_for_provider)
    monkeypatch.setattr(executor, "execute_completion", fake_execute_completion)
    monkeypatch.setattr(executor, "CLIENTS", {})

    response = await executor.honcho_llm_call_inner(
        "openai",
        "gpt-5.5",
        "hello",
        max_tokens=100,
        selected_config=ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
            max_output_tokens=42,
        ),
    )

    assert response.content == "ok"
