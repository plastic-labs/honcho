import pytest

from src.config import (
    ConfiguredModelSettings,
    FallbackModelSettings,
    ModelConfig,
    ModelOverrideSettings,
    ResolvedFallbackConfig,
    resolve_model_config,
)


@pytest.mark.parametrize("api_mode", ["response", "chat_completions", True, 1])
def test_invalid_api_mode_is_rejected_at_config_load(api_mode: object) -> None:
    with pytest.raises(ValueError, match="api_mode must be 'responses'"):
        ConfiguredModelSettings(
            model="gpt-5.4",
            transport="openai",
            overrides=ModelOverrideSettings(
                provider_params={"api_mode": api_mode},
            ),
        )


def test_responses_api_mode_is_rejected_on_non_openai_transport() -> None:
    with pytest.raises(ValueError, match="api_mode is only supported"):
        ConfiguredModelSettings(
            model="claude-haiku-4-5",
            transport="anthropic",
            overrides=ModelOverrideSettings(
                provider_params={"api_mode": "responses"},
            ),
        )


def test_responses_api_mode_is_rejected_on_non_openai_fallback() -> None:
    with pytest.raises(ValueError, match="api_mode is only supported"):
        FallbackModelSettings(
            model="gemini-2.5-pro",
            transport="gemini",
            overrides=ModelOverrideSettings(
                provider_params={"api_mode": "responses"},
            ),
        )


def test_responses_api_mode_resolves_for_openai_transport() -> None:
    configured = ConfiguredModelSettings(
        model="gpt-5.4",
        transport="openai",
        overrides=ModelOverrideSettings(
            provider_params={"api_mode": "responses"},
        ),
    )

    resolved = resolve_model_config(configured)

    assert resolved.provider_params["api_mode"] == "responses"


def test_runtime_model_config_rejects_responses_on_non_openai_transport() -> None:
    with pytest.raises(ValueError, match="api_mode is only supported"):
        ModelConfig(
            model="claude-haiku-4-5",
            transport="anthropic",
            provider_params={"api_mode": "responses"},
        )


def test_resolved_fallback_rejects_responses_on_non_openai_transport() -> None:
    with pytest.raises(ValueError, match="api_mode is only supported"):
        ResolvedFallbackConfig(
            model="gemini-2.5-pro",
            transport="gemini",
            provider_params={"api_mode": "responses"},
        )


def test_for_model_revalidates_responses_api_mode_transport() -> None:
    config = ModelConfig(
        model="gpt-5.4",
        transport="openai",
        provider_params={"api_mode": "responses"},
    )

    with pytest.raises(ValueError, match="api_mode is only supported"):
        config.for_model(
            "claude-haiku-4-5",
            transport_override="anthropic",
        )
