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
