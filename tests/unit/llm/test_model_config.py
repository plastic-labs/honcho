import pytest

from src.config import DeriverSettings, DialecticLevelSettings, ModelConfig


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url is required"):
        ModelConfig(
            model="openai/my-local-model",
            transport="openai_compatible",
            api_key="test-key",
        )


def test_provider_native_rejects_base_url() -> None:
    with pytest.raises(ValueError, match="base_url is only valid"):
        ModelConfig(
            model="anthropic/claude-haiku-4-5",
            transport="provider_native",
            base_url="https://example.com/v1",
        )


def test_openai_compatible_fallback_requires_base_url() -> None:
    with pytest.raises(ValueError, match="fallback_base_url is required"):
        ModelConfig(
            model="anthropic/claude-haiku-4-5",
            fallback_model="openai/my-local-model",
            fallback_transport="openai_compatible",
        )


def test_anthropic_thinking_budget_has_minimum() -> None:
    with pytest.raises(ValueError, match="thinking_budget_tokens must be >= 1024"):
        ModelConfig(
            model="anthropic/claude-haiku-4-5",
            thinking_budget_tokens=512,
        )


def test_reasoning_effort_alias_populates_generic_thinking_effort() -> None:
    config = ModelConfig(
        model="openai/gpt-5",
        reasoning_effort="minimal",
    )

    assert config.thinking_effort == "minimal"
    assert config.reasoning_effort == "minimal"


def test_for_model_overrides_model_and_transport() -> None:
    config = ModelConfig(model="anthropic/claude-haiku-4-5")

    updated = config.for_model(
        "openai/my-local-model",
        transport_override="openai_compatible",
    )

    assert updated.model == "openai/my-local-model"
    assert updated.transport == "openai_compatible"
    assert config.transport == "provider_native"


def test_deriver_settings_to_model_config_qualifies_legacy_model() -> None:
    settings = DeriverSettings(
        PROVIDER="google",
        MODEL="gemini-2.5-flash-lite",
        THINKING_BUDGET_TOKENS=1024,
        MAX_OUTPUT_TOKENS=4096,
    )

    config = settings.to_model_config()

    assert config.model == "gemini/gemini-2.5-flash-lite"
    assert config.transport == "provider_native"
    assert config.thinking_budget_tokens == 1024
    assert config.max_output_tokens == 4096


def test_provider_params_default_to_empty_dict() -> None:
    config = ModelConfig(model="openai/gpt-4.1-mini")

    assert config.provider_params == {}


def test_dialectic_level_settings_to_model_config_handles_fallback() -> None:
    settings = DialecticLevelSettings(
        PROVIDER="anthropic",
        MODEL="claude-haiku-4-5",
        BACKUP_PROVIDER="google",
        BACKUP_MODEL="gemini-2.5-pro",
        THINKING_BUDGET_TOKENS=1024,
        MAX_TOOL_ITERATIONS=2,
    )

    config = settings.to_model_config()

    assert config.model == "anthropic/claude-haiku-4-5"
    assert config.fallback_model == "gemini/gemini-2.5-pro"
    assert config.fallback_transport == "provider_native"
