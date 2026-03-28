from pathlib import Path

import pytest

from src.config import (
    ConfiguredModelSettings,
    DeriverSettings,
    DialecticLevelSettings,
    DreamSettings,
    ModelConfig,
    ModelOverrideSettings,
    SummarySettings,
    load_toml_config,
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


def test_deriver_settings_syncs_legacy_model_into_model_config() -> None:
    settings = DeriverSettings(
        PROVIDER="google",
        MODEL="gemini-2.5-flash-lite",
        THINKING_BUDGET_TOKENS=1024,
        MAX_OUTPUT_TOKENS=4096,
    )

    assert settings.MODEL_CONFIG is not None
    assert settings.MODEL_CONFIG.model == "gemini/gemini-2.5-flash-lite"
    resolved = resolve_model_config(settings.MODEL_CONFIG)
    assert resolved.transport == "provider_native"
    assert resolved.thinking_budget_tokens == 1024
    assert resolved.max_output_tokens == 4096


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


def test_dialectic_level_settings_syncs_fallback_into_model_config() -> None:
    settings = DialecticLevelSettings(
        PROVIDER="anthropic",
        MODEL="claude-haiku-4-5",
        BACKUP_PROVIDER="google",
        BACKUP_MODEL="gemini-2.5-pro",
        THINKING_BUDGET_TOKENS=1024,
        MAX_TOOL_ITERATIONS=2,
    )

    if settings.MODEL_CONFIG is None:
        raise AssertionError("Expected DIALECTIC MODEL_CONFIG to be resolved")
    resolved = resolve_model_config(settings.MODEL_CONFIG)
    assert resolved.model == "anthropic/claude-haiku-4-5"
    assert resolved.fallback_model == "gemini/gemini-2.5-pro"
    assert resolved.fallback_transport == "provider_native"


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


def test_dream_specialist_model_configs_inherit_main_model_defaults() -> None:
    settings = DreamSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="anthropic/claude-sonnet-4-5",
            fallback_model="gemini/gemini-2.5-pro",
            thinking_budget_tokens=4096,
            max_output_tokens=12_000,
            overrides=ModelOverrideSettings(
                provider_params={"verbosity": "low"},
            ),
        ),
        DEDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="anthropic/claude-haiku-4-5",
        ),
        INDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="anthropic/claude-opus-4-1",
        ),
    )

    if settings.DEDUCTION_MODEL_CONFIG is None:
        raise AssertionError("Expected DREAM DEDUCTION MODEL_CONFIG to be resolved")
    if settings.INDUCTION_MODEL_CONFIG is None:
        raise AssertionError("Expected DREAM INDUCTION MODEL_CONFIG to be resolved")

    assert settings.DEDUCTION_MODEL_CONFIG.model == "anthropic/claude-haiku-4-5"
    assert settings.DEDUCTION_MODEL_CONFIG.fallback_model == "gemini/gemini-2.5-pro"
    assert settings.DEDUCTION_MODEL_CONFIG.thinking_budget_tokens == 4096
    assert settings.DEDUCTION_MODEL_CONFIG.max_output_tokens == 12_000
    assert settings.DEDUCTION_MODEL_CONFIG.overrides.provider_params == {
        "verbosity": "low"
    }
    assert settings.INDUCTION_MODEL_CONFIG.model == "anthropic/claude-opus-4-1"
    assert settings.INDUCTION_MODEL_CONFIG.thinking_budget_tokens == 4096


def test_config_toml_example_uses_nested_model_config_sections() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config.toml.example"
    config_data = load_toml_config(str(config_path))

    deriver_config = ConfiguredModelSettings.model_validate(
        config_data["deriver"]["model_config"]
    )
    minimal_level = DialecticLevelSettings.model_validate(
        config_data["dialectic"]["levels"]["minimal"]
    )
    max_level = DialecticLevelSettings.model_validate(
        config_data["dialectic"]["levels"]["max"]
    )
    summary_config = ConfiguredModelSettings.model_validate(
        config_data["summary"]["model_config"]
    )
    dream_model_config = ConfiguredModelSettings.model_validate(
        config_data["dream"]["model_config"]
    )
    deduction_model_config = ConfiguredModelSettings.model_validate(
        config_data["dream"]["deduction_model_config"]
    )
    induction_model_config = ConfiguredModelSettings.model_validate(
        config_data["dream"]["induction_model_config"]
    )
    dream = DreamSettings.model_validate(
        {
            "MODEL_CONFIG": dream_model_config,
            "DEDUCTION_MODEL_CONFIG": deduction_model_config,
            "INDUCTION_MODEL_CONFIG": induction_model_config,
        }
    )

    if dream.MODEL_CONFIG is None:
        raise AssertionError(
            "Expected DREAM MODEL_CONFIG to be resolved from config.toml.example"
        )
    if dream.DEDUCTION_MODEL_CONFIG is None:
        raise AssertionError(
            "Expected DREAM DEDUCTION MODEL_CONFIG to be resolved from config.toml.example"
        )

    assert deriver_config.model == "gemini/gemini-2.5-flash-lite"
    assert deriver_config.thinking_budget_tokens == 1024
    assert minimal_level.MODEL_CONFIG is not None
    assert minimal_level.MODEL_CONFIG.model == ("gemini/gemini-2.5-flash-lite")
    assert max_level.MODEL_CONFIG is not None
    assert max_level.MODEL_CONFIG.model == "anthropic/claude-haiku-4-5"
    assert max_level.MODEL_CONFIG.thinking_budget_tokens == 2048
    assert summary_config.model == "gemini/gemini-2.5-flash"
    assert dream.MODEL_CONFIG.model == "anthropic/claude-sonnet-4-20250514"
    assert dream.DEDUCTION_MODEL_CONFIG.model == "anthropic/claude-haiku-4-5"
    assert dream.DEDUCTION_MODEL_CONFIG.thinking_budget_tokens == 8192
