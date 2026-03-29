import pytest

from src.config import ModelConfig, settings
from src.llm.credentials import resolve_credentials


def test_provider_native_credentials_use_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "ANTHROPIC_API_KEY", "anthropic-test-key")

    credentials = resolve_credentials(ModelConfig(model="anthropic/claude-haiku-4-5"))

    assert credentials == {"api_key": "anthropic-test-key"}


def test_openai_compatible_credentials_use_per_model_config() -> None:
    credentials = resolve_credentials(
        ModelConfig(
            model="openai/my-local-model",
            transport="openai_compatible",
            api_key="local-key",
            base_url="http://localhost:8000/v1",
        )
    )

    assert credentials == {
        "api_key": "local-key",
        "api_base": "http://localhost:8000/v1",
    }


def test_openai_compatible_credentials_fall_back_to_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings.LLM,
        "OPENAI_COMPATIBLE_API_KEY",
        "compat-test-key",
    )
    monkeypatch.setattr(
        settings.LLM,
        "OPENAI_COMPATIBLE_BASE_URL",
        "http://compat.local/v1",
    )

    credentials = resolve_credentials(
        ModelConfig(
            model="openai/my-local-model",
            transport="openai_compatible",
        )
    )

    assert credentials == {
        "api_key": "compat-test-key",
        "api_base": "http://compat.local/v1",
    }


def test_vllm_compat_provider_uses_vllm_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "VLLM_API_KEY", "vllm-test-key")
    monkeypatch.setattr(settings.LLM, "VLLM_BASE_URL", "http://localhost:8000/v1")

    credentials = resolve_credentials(
        ModelConfig(
            model="openai/my-local-model",
            transport="openai_compatible",
            compat_provider="vllm",
        )
    )

    assert credentials == {
        "api_key": "vllm-test-key",
        "api_base": "http://localhost:8000/v1",
    }
