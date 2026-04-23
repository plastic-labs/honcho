import pytest

from src.config import ConfiguredModelSettings, DeriverSettings


def _make_deriver_settings(
    *,
    MAX_INPUT_TOKENS: int = 23000,
    MAX_CUSTOM_INSTRUCTIONS_TOKENS: int | None = None,
    REPRESENTATION_BATCH_MAX_TOKENS: int = 1024,
) -> DeriverSettings:
    return DeriverSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="gpt-5.4-mini",
            transport="openai",
        ),
        MAX_INPUT_TOKENS=MAX_INPUT_TOKENS,
        MAX_CUSTOM_INSTRUCTIONS_TOKENS=MAX_CUSTOM_INSTRUCTIONS_TOKENS,
        REPRESENTATION_BATCH_MAX_TOKENS=REPRESENTATION_BATCH_MAX_TOKENS,
    )


def test_effective_custom_instructions_tokens_requires_explicit_limit() -> None:
    settings = _make_deriver_settings()

    with pytest.raises(
        ValueError,
        match=r"set \[deriver\]\.MAX_CUSTOM_INSTRUCTIONS_TOKENS in config\.toml",
    ):
        _ = settings.effective_max_custom_instructions_tokens


def test_effective_custom_instructions_tokens_uses_explicit_limit() -> None:
    settings = _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=500)

    assert settings.effective_max_custom_instructions_tokens == 500


def test_custom_instructions_tokens_cannot_exceed_input_budget() -> None:
    with pytest.raises(
        ValueError,
        match=r"MAX_CUSTOM_INSTRUCTIONS_TOKENS.*cannot exceed max deriver input tokens",
    ):
        _make_deriver_settings(
            MAX_INPUT_TOKENS=400,
            MAX_CUSTOM_INSTRUCTIONS_TOKENS=500,
            REPRESENTATION_BATCH_MAX_TOKENS=128,
        )


def test_custom_instructions_tokens_cannot_exceed_supported_cap() -> None:
    with pytest.raises(ValueError, match="less than or equal to 500"):
        _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=501)
