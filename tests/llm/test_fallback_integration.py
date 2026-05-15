"""Integration tests for LLM model fallback (t_124ab52c).

Tests the full honcho_llm_call() flow with mocked backends to verify:
- 5.1: Primary fails with 429 → fallback succeeds
- 5.2: Primary returns 429 → fallback triggers within 1 retry (not 3)
- Both fail → error raised
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.config import (
    ConfiguredModelSettings,
    FallbackModelSettings,
)
from src.llm.api import honcho_llm_call
from src.llm.backend import CompletionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FailingBackend:
    """Backend that always raises a retryable error."""

    def __init__(self, status_code: int = 429):
        self.status_code = status_code
        self.call_count = 0

    async def complete(self, **kwargs):
        self.call_count += 1
        exc = Exception(f"HTTP {self.status_code}")
        exc.status_code = self.status_code
        raise exc

    async def stream(self, **kwargs):
        self.call_count += 1
        exc = Exception(f"HTTP {self.status_code}")
        exc.status_code = self.status_code
        raise exc


class SuccessBackend:
    """Backend that always succeeds."""

    def __init__(self, content: str = "fallback response"):
        self.content = content
        self.call_count = 0

    async def complete(self, **kwargs):
        self.call_count += 1
        return CompletionResult(content=self.content)

    async def stream(self, **kwargs):
        self.call_count += 1
        yield MagicMock(content=self.content, is_done=True)


def make_config(
    primary_transport: str = "openai",
    primary_model: str = "gpt-4o",
    fallback_transport: str = "lmstudio",
    fallback_model: str = "qwen3.5-9b",
) -> ConfiguredModelSettings:
    return ConfiguredModelSettings(
        model=primary_model,
        transport=primary_transport,
        fallback=FallbackModelSettings(
            model=fallback_model,
            transport=fallback_transport,
        ),
    )


# ---------------------------------------------------------------------------
# 5.1: Primary fails → fallback succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_fails_fallback_succeeds():
    """When primary fails with retryable error, fallback model is used and succeeds."""
    from src.llm import executor as executor_mod

    config = make_config()
    primary_backend = FailingBackend(status_code=429)
    fallback_backend = SuccessBackend(content="fallback response")

    def mock_backend_for_provider(provider, client):
        if provider == "openai":
            return primary_backend
        elif provider == "lmstudio":
            return fallback_backend
        raise ValueError(f"Unexpected provider: {provider}")

    mock_client = MagicMock()

    with (
        patch("src.llm.runtime.client_for_model_config", return_value=mock_client),
        patch.object(executor_mod, "backend_for_provider", side_effect=mock_backend_for_provider),
    ):
        result = await honcho_llm_call(
            model_config=config,
            prompt="test prompt",
            max_tokens=100,
            enable_retry=True,
            retry_attempts=3,
        )

    assert primary_backend.call_count == 1, (
        f"Primary should be called exactly 1 time, was {primary_backend.call_count}"
    )
    assert fallback_backend.call_count >= 1, (
        f"Fallback should have been called, was {fallback_backend.call_count}"
    )
    assert result.content == "fallback response"


# ---------------------------------------------------------------------------
# 5.2: Primary returns 429 → fallback triggers within 1 retry (not 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_triggers_on_first_failure():
    """Fallback should trigger on the first 429, not wait for all 3 retries."""
    from src.llm import executor as executor_mod

    config = make_config()
    primary_backend = FailingBackend(status_code=429)
    fallback_backend = SuccessBackend(content="fallback response")

    def mock_backend_for_provider(provider, client):
        if provider == "openai":
            return primary_backend
        elif provider == "lmstudio":
            return fallback_backend
        raise ValueError(f"Unexpected provider: {provider}")

    mock_client = MagicMock()

    with (
        patch("src.llm.runtime.client_for_model_config", return_value=mock_client),
        patch.object(executor_mod, "backend_for_provider", side_effect=mock_backend_for_provider),
    ):
        result = await honcho_llm_call(
            model_config=config,
            prompt="test prompt",
            max_tokens=100,
            enable_retry=True,
            retry_attempts=3,
        )

    assert primary_backend.call_count == 1, (
        f"Primary should be called exactly 1 time (first failure triggers fallback), "
        f"but was called {primary_backend.call_count} times"
    )
    assert fallback_backend.call_count >= 1
    assert result.content == "fallback response"


# ---------------------------------------------------------------------------
# Both primary and fallback fail → proper error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_primary_and_fallback_fail():
    """When both primary and fallback fail, an error should be raised."""
    from src.llm import executor as executor_mod

    config = make_config()
    primary_backend = FailingBackend(status_code=429)
    fallback_backend = FailingBackend(status_code=500)

    def mock_backend_for_provider(provider, client):
        if provider == "openai":
            return primary_backend
        elif provider == "lmstudio":
            return fallback_backend
        raise ValueError(f"Unexpected provider: {provider}")

    mock_client = MagicMock()

    with (
        patch("src.llm.runtime.client_for_model_config", return_value=mock_client),
        patch.object(executor_mod, "backend_for_provider", side_effect=mock_backend_for_provider),
    ):
        with pytest.raises(Exception):
            await honcho_llm_call(
                model_config=config,
                prompt="test prompt",
                max_tokens=100,
                enable_retry=True,
                retry_attempts=2,
            )
