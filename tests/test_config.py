import pytest

from src.config import ConfiguredModelSettings, DeriverSettings


def _make_deriver_settings(
    *,
    MAX_INPUT_TOKENS: int = 25000,
    MAX_CUSTOM_INSTRUCTIONS_TOKENS: int = 2000,
    REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS: int = 512,
    REPRESENTATION_BATCH_TARGET_INPUT_TOKENS: int = 1024,
    REPRESENTATION_BATCH_MAX_AGE_SECONDS: int = 1800,
) -> DeriverSettings:
    return DeriverSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="gpt-5.4-mini",
            transport="openai",
        ),
        MAX_INPUT_TOKENS=MAX_INPUT_TOKENS,
        MAX_CUSTOM_INSTRUCTIONS_TOKENS=MAX_CUSTOM_INSTRUCTIONS_TOKENS,
        REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS=REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS,
        REPRESENTATION_BATCH_TARGET_INPUT_TOKENS=REPRESENTATION_BATCH_TARGET_INPUT_TOKENS,
        REPRESENTATION_BATCH_MAX_AGE_SECONDS=REPRESENTATION_BATCH_MAX_AGE_SECONDS,
    )


def test_deriver_defaults_enable_custom_instructions_at_supported_cap() -> None:
    settings = _make_deriver_settings()

    assert settings.MAX_INPUT_TOKENS == 25000
    assert settings.MAX_CUSTOM_INSTRUCTIONS_TOKENS == 2000
    assert settings.REPRESENTATION_BATCH_MAX_AGE_SECONDS == 1800


def test_custom_instructions_tokens_can_be_disabled_with_zero() -> None:
    settings = _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=0)

    assert settings.MAX_CUSTOM_INSTRUCTIONS_TOKENS == 0


def test_custom_instructions_tokens_cannot_exceed_supported_cap() -> None:
    with pytest.raises(ValueError, match="less than or equal to 2000"):
        _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=2001)


def test_representation_batch_age_can_be_disabled_with_zero() -> None:
    settings = _make_deriver_settings(REPRESENTATION_BATCH_MAX_AGE_SECONDS=0)

    assert settings.REPRESENTATION_BATCH_MAX_AGE_SECONDS == 0


def test_representation_batch_age_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _make_deriver_settings(REPRESENTATION_BATCH_MAX_AGE_SECONDS=-1)


def test_representation_batch_work_unit_target_can_be_disabled_with_zero() -> None:
    settings = _make_deriver_settings(REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS=0)

    assert settings.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS == 0


def test_representation_batch_work_unit_target_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _make_deriver_settings(REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS=-1)


def test_representation_batch_tokens_can_diverge() -> None:
    settings = _make_deriver_settings(
        REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS=4096,
        REPRESENTATION_BATCH_TARGET_INPUT_TOKENS=1024,
    )

    assert settings.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS == 4096
    assert settings.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS == 1024


def test_representation_batch_target_input_cannot_exceed_max_input_tokens() -> None:
    with pytest.raises(ValueError, match="cannot exceed max deriver input tokens"):
        _make_deriver_settings(
            MAX_INPUT_TOKENS=1000,
            REPRESENTATION_BATCH_TARGET_INPUT_TOKENS=2048,
        )


def _configured_with_timeout(timeout: object) -> ConfiguredModelSettings:
    return ConfiguredModelSettings.model_validate(
        {
            "model": "gpt-5.4-mini",
            "transport": "openai",
            "overrides": {"provider_params": {"timeout": timeout}},
        }
    )


@pytest.mark.parametrize("timeout", [30, 42.5, "42.5", " 60 "])
def test_provider_timeout_is_normalized_at_config_load(timeout: object) -> None:
    settings = _configured_with_timeout(timeout)

    normalized = settings.overrides.provider_params["timeout"]
    assert isinstance(normalized, float)
    assert normalized == float(str(timeout).strip())


@pytest.mark.parametrize(
    "timeout",
    ["slow", "", 0, -1, True, float("nan"), float("inf"), "nan", "inf", None, [30]],
)
def test_provider_timeout_is_rejected_at_config_load(timeout: object) -> None:
    with pytest.raises(
        ValueError, match=r"provider_params\.timeout must be a positive number"
    ):
        _configured_with_timeout(timeout)


def test_provider_timeout_on_fallback_overrides_is_validated_at_config_load() -> None:
    with pytest.raises(
        ValueError, match=r"provider_params\.timeout must be a positive number"
    ):
        ConfiguredModelSettings.model_validate(
            {
                "model": "gpt-5.4-mini",
                "transport": "openai",
                "fallback": {
                    "model": "gpt-4.1",
                    "transport": "openai",
                    "overrides": {"provider_params": {"timeout": "slow"}},
                },
            }
        )
