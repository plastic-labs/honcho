import pytest

from src.config import (
    ConfiguredModelSettings,
    DeriverSettings,
    DialecticLevelSettings,
    ModelConfig,
    ModelOverrideSettings,
    SummarySettings,
    resolve_model_config,
)


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
    config = ModelConfig.model_validate(
        {
            "model": "openai/gpt-5",
            "reasoning_effort": "minimal",
        }
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
    assert settings.MODEL_CONFIG is not None
    assert settings.MODEL_CONFIG.model == "gemini/gemini-2.5-flash-lite"


def test_summary_settings_syncs_legacy_fields_from_nested_model_config() -> None:
    settings = SummarySettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="anthropic/claude-haiku-4-5",
            fallback_model="gemini/gemini-2.5-pro",
            thinking_budget_tokens=1024,
        ),
    )

    assert settings.PROVIDER == "anthropic"
    assert settings.MODEL == "claude-haiku-4-5"
    assert settings.BACKUP_PROVIDER == "google"
    assert settings.BACKUP_MODEL == "gemini-2.5-pro"
    assert settings.THINKING_BUDGET_TOKENS == 1024


def test_resolve_model_config_reads_override_env_and_provider_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_LOCAL_API_KEY", "test-key")

    configured = ConfiguredModelSettings(
        model="openai/my-local-model",
        transport="openai_compatible",
        overrides=ModelOverrideSettings(
            api_key_env="SUMMARY_LOCAL_API_KEY",
            base_url="http://localhost:8000/v1",
            provider_params={"verbosity": "low"},
        ),
    )

    resolved = resolve_model_config(configured)

    assert resolved.api_key == "test-key"
    assert resolved.base_url == "http://localhost:8000/v1"
    assert resolved.provider_params == {"verbosity": "low"}


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


def test_dialectic_level_settings_accepts_nested_model_config() -> None:
    settings = DialecticLevelSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="anthropic/claude-haiku-4-5",
            fallback_model="gemini/gemini-2.5-pro",
            thinking_budget_tokens=1024,
        ),
        MAX_TOOL_ITERATIONS=2,
    )

    assert settings.PROVIDER == "anthropic"
    assert settings.MODEL == "claude-haiku-4-5"
    assert settings.BACKUP_PROVIDER == "google"
    assert settings.BACKUP_MODEL == "gemini-2.5-pro"
    assert settings.THINKING_BUDGET_TOKENS == 1024
