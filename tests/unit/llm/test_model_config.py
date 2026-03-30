from pathlib import Path

import pytest

from src.config import (
    ConfiguredModelSettings,
    DialecticLevelSettings,
    DreamSettings,
    ModelConfig,
    ModelOverrideSettings,
    SummarySettings,
    load_toml_config,
    resolve_model_config,
)


def test_fallback_base_url_requires_fallback_model() -> None:
    with pytest.raises(ValueError, match="fallback_base_url requires fallback_model"):
        ModelConfig(
            model="claude-haiku-4-5",
            transport="anthropic",
            fallback_base_url="https://example.com/v1",
        )


def test_base_url_is_allowed_for_any_transport() -> None:
    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
        base_url="https://anthropic-proxy.example/v1",
    )

    assert config.base_url == "https://anthropic-proxy.example/v1"


def test_anthropic_thinking_budget_has_minimum() -> None:
    with pytest.raises(ValueError, match="thinking_budget_tokens must be >= 1024"):
        ModelConfig(
            model="claude-haiku-4-5",
            transport="anthropic",
            thinking_budget_tokens=512,
        )


def test_reasoning_effort_alias_populates_generic_thinking_effort() -> None:
    config = ModelConfig.model_validate(
        {
            "model": "gpt-5",
            "transport": "openai",
            "reasoning_effort": "minimal",
        }
    )

    assert config.thinking_effort == "minimal"
    assert config.reasoning_effort == "minimal"


def test_for_model_overrides_model_and_transport() -> None:
    config = ModelConfig(
        model="claude-haiku-4-5",
        transport="anthropic",
    )

    updated = config.for_model(
        "gpt-5-mini",
        transport_override="openai",
    )

    assert updated.model == "gpt-5-mini"
    assert updated.transport == "openai"
    assert config.transport == "anthropic"


def test_configured_model_settings_validate_like_runtime_model_config() -> None:
    with pytest.raises(ValueError, match="thinking_budget_tokens must be >= 1024"):
        ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            thinking_budget_tokens=512,
        )


def test_summary_settings_accept_nested_model_config() -> None:
    settings = SummarySettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            fallback_model="gemini-2.5-pro",
            fallback_transport="gemini",
            thinking_budget_tokens=1024,
        ),
    )

    assert settings.MODEL_CONFIG.model == "claude-haiku-4-5"
    assert settings.MODEL_CONFIG.transport == "anthropic"
    assert settings.MODEL_CONFIG.fallback_model == "gemini-2.5-pro"
    assert settings.MODEL_CONFIG.fallback_transport == "gemini"
    assert settings.MODEL_CONFIG.thinking_budget_tokens == 1024


def test_resolve_model_config_reads_override_env_and_provider_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUMMARY_LOCAL_API_KEY", "test-key")

    configured = ConfiguredModelSettings(
        model="my-local-model",
        transport="openai",
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


def test_dialectic_level_settings_accepts_nested_model_config() -> None:
    settings = DialecticLevelSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            fallback_model="gemini-2.5-pro",
            fallback_transport="gemini",
            thinking_budget_tokens=1024,
        ),
        MAX_TOOL_ITERATIONS=2,
    )

    resolved = resolve_model_config(settings.MODEL_CONFIG)
    assert resolved.model == "claude-haiku-4-5"
    assert resolved.transport == "anthropic"
    assert resolved.fallback_model == "gemini-2.5-pro"
    assert resolved.fallback_transport == "gemini"


def test_dialectic_level_settings_require_nested_model_config() -> None:
    with pytest.raises(ValueError, match="Field required"):
        DialecticLevelSettings.model_validate({"MAX_TOOL_ITERATIONS": 2})


def test_dialectic_level_settings_reject_legacy_flat_model_shape() -> None:
    with pytest.raises(ValueError, match="Field required"):
        DialecticLevelSettings.model_validate(
            {
                "MODEL": "claude-haiku-4-5",
                "THINKING_BUDGET_TOKENS": 1024,
                "MAX_TOOL_ITERATIONS": 2,
            }
        )


def test_legacy_prefixed_model_strings_are_normalized() -> None:
    config = ModelConfig.model_validate({"model": "gemini/gemini-2.5-flash"})
    configured = ConfiguredModelSettings.model_validate(
        {"model": "anthropic/claude-haiku-4-5"}
    )

    assert config.transport == "gemini"
    assert config.model == "gemini-2.5-flash"
    assert configured.transport == "anthropic"
    assert configured.model == "claude-haiku-4-5"


def test_dream_specialist_model_configs_inherit_main_model_defaults() -> None:
    settings = DreamSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-sonnet-4-5",
            transport="anthropic",
            fallback_model="gemini-2.5-pro",
            fallback_transport="gemini",
            thinking_budget_tokens=4096,
            max_output_tokens=12_000,
            overrides=ModelOverrideSettings(
                provider_params={"verbosity": "low"},
            ),
        ),
        DEDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
        ),
        INDUCTION_MODEL_CONFIG=ConfiguredModelSettings(
            model="claude-opus-4-1",
            transport="anthropic",
        ),
    )

    assert settings.DEDUCTION_MODEL_CONFIG.model == "claude-haiku-4-5"
    assert settings.DEDUCTION_MODEL_CONFIG.fallback_model == "gemini-2.5-pro"
    assert settings.DEDUCTION_MODEL_CONFIG.fallback_transport == "gemini"
    assert settings.DEDUCTION_MODEL_CONFIG.thinking_budget_tokens == 4096
    assert settings.DEDUCTION_MODEL_CONFIG.max_output_tokens == 12_000
    assert settings.DEDUCTION_MODEL_CONFIG.overrides.provider_params == {
        "verbosity": "low"
    }
    assert settings.INDUCTION_MODEL_CONFIG.model == "claude-opus-4-1"
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

    assert deriver_config.transport == "gemini"
    assert deriver_config.model == "gemini-2.5-flash-lite"
    assert deriver_config.thinking_budget_tokens == 1024
    assert minimal_level.MODEL_CONFIG.model == "gemini-2.5-flash-lite"
    assert minimal_level.MODEL_CONFIG.transport == "gemini"
    assert max_level.MODEL_CONFIG.model == "claude-haiku-4-5"
    assert max_level.MODEL_CONFIG.transport == "anthropic"
    assert max_level.MODEL_CONFIG.thinking_budget_tokens == 2048
    assert summary_config.model == "gemini-2.5-flash"
    assert summary_config.transport == "gemini"
    assert dream.MODEL_CONFIG.model == "claude-sonnet-4-20250514"
    assert dream.MODEL_CONFIG.transport == "anthropic"
    assert dream.DEDUCTION_MODEL_CONFIG.model == "claude-haiku-4-5"
    assert dream.DEDUCTION_MODEL_CONFIG.thinking_budget_tokens == 8192


def test_env_template_uses_nested_model_config_keys() -> None:
    env_template_path = Path(__file__).resolve().parents[3] / ".env.template"
    env_template = env_template_path.read_text()

    assert "DERIVER_MODEL_CONFIG__MODEL" in env_template
    assert "DIALECTIC_LEVELS__minimal__MODEL_CONFIG__MODEL" in env_template
    assert "SUMMARY_MODEL_CONFIG__MODEL" in env_template
    assert "DREAM_MODEL_CONFIG__MODEL" in env_template
    assert "DREAM_DEDUCTION_MODEL_CONFIG__MODEL" in env_template

    assert "DERIVER_PROVIDER=" not in env_template
    assert "SUMMARY_PROVIDER=" not in env_template
    assert "DIALECTIC_LEVELS__minimal__PROVIDER=" not in env_template
    assert "DREAM_PROVIDER=" not in env_template
    assert "DREAM_DEDUCTION_MODEL=" not in env_template
