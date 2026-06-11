from typing import Any

import pytest

from src.llm.openrouter import attribution_headers


def test_openrouter_attribution_headers_only_for_openrouter() -> None:
    assert (
        attribution_headers(
            base_url="http://localhost:8000/v1",
            app_url="https://example.test",
            app_title="Honcho Test",
        )
        is None
    )

    assert attribution_headers(
        base_url="https://openrouter.ai/api/v1/",
        app_url="https://example.test",
        app_title="Honcho Test",
    ) == {
        "HTTP-Referer": "https://example.test",
        "X-OpenRouter-Title": "Honcho Test",
        "X-Title": "Honcho Test",
    }


def test_openrouter_attribution_headers_omits_empty_values() -> None:
    assert attribution_headers(
        base_url="https://openrouter.ai/api/v1",
        app_url=None,
        app_title="Honcho Test",
    ) == {
        "X-OpenRouter-Title": "Honcho Test",
        "X-Title": "Honcho Test",
    }

    assert (
        attribution_headers(
            base_url="https://openrouter.ai/api/v1",
            app_url=None,
            app_title=None,
        )
        is None
    )


def test_openai_override_client_receives_openrouter_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.llm import registry

    calls: list[dict[str, Any]] = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(registry.settings.LLM, "OPENROUTER_APP_URL", "https://app.test")
    monkeypatch.setattr(registry.settings.LLM, "OPENROUTER_APP_TITLE", "Honcho Test")
    monkeypatch.setattr(registry, "AsyncOpenAI", FakeAsyncOpenAI)
    registry.get_openai_override_client.cache_clear()

    try:
        registry.get_openai_override_client(
            "https://openrouter.ai/api/v1",
            "test-key",
        )
    finally:
        registry.get_openai_override_client.cache_clear()

    assert calls == [
        {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "default_headers": {
                "HTTP-Referer": "https://app.test",
                "X-OpenRouter-Title": "Honcho Test",
                "X-Title": "Honcho Test",
            },
        }
    ]
