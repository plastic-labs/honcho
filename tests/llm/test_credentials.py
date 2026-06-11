import pytest

from src.config import ModelConfig, settings
from src.llm.credentials import resolve_credentials


def test_transport_credentials_use_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "ANTHROPIC_API_KEY", "anthropic-test-key")

    credentials = resolve_credentials(
        ModelConfig(model="claude-haiku-4-5", transport="anthropic")
    )

    assert credentials == {"api_key": "anthropic-test-key", "api_base": None}


def test_openai_transport_credentials_use_per_model_config() -> None:
    credentials = resolve_credentials(
        ModelConfig(
            model="my-local-model",
            transport="openai",
            api_key="local-key",
            base_url="http://localhost:8000/v1",
        )
    )

    assert credentials == {
        "api_key": "local-key",
        "api_base": "http://localhost:8000/v1",
    }


def test_openai_transport_credentials_fall_back_to_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "OPENAI_API_KEY", "openai-test-key")

    credentials = resolve_credentials(
        ModelConfig(
            model="my-local-model",
            transport="openai",
        )
    )

    assert credentials == {
        "api_key": "openai-test-key",
        "api_base": None,
    }


def test_codex_oauth_credentials_do_not_require_api_key() -> None:
    credentials = resolve_credentials(
        ModelConfig(
            model="gpt-5.5",
            transport="openai",
            auth_mode="codex_oauth",
        )
    )

    assert credentials == {
        "api_key": None,
        "api_base": "https://chatgpt.com/backend-api/codex",
    }
