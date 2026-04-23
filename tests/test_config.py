import pytest

from src.config import ConfiguredModelSettings, DeriverSettings


def _make_deriver_settings(
    *,
    MAX_INPUT_TOKENS: int = 23000,
    MAX_CUSTOM_INSTRUCTIONS_TOKENS: int | None = None,
) -> DeriverSettings:
    return DeriverSettings(
        MODEL_CONFIG=ConfiguredModelSettings(
            model="gpt-5.4-mini",
            transport="openai",
        ),
        MAX_INPUT_TOKENS=MAX_INPUT_TOKENS,
        MAX_CUSTOM_INSTRUCTIONS_TOKENS=MAX_CUSTOM_INSTRUCTIONS_TOKENS,
    )


def test_effective_custom_instructions_tokens_requires_explicit_limit() -> None:
    settings = _make_deriver_settings()

    with pytest.raises(
        ValueError,
        match=r"set \[deriver\]\.MAX_CUSTOM_INSTRUCTIONS_TOKENS in config\.toml",
    ):
        _ = settings.effective_max_custom_instructions_tokens


def test_effective_custom_instructions_tokens_uses_explicit_limit() -> None:
    settings = _make_deriver_settings(MAX_CUSTOM_INSTRUCTIONS_TOKENS=2048)

    assert settings.effective_max_custom_instructions_tokens == 2048


def test_custom_instructions_tokens_cannot_exceed_input_budget() -> None:
    with pytest.raises(
        ValueError,
        match=r"MAX_CUSTOM_INSTRUCTIONS_TOKENS.*cannot exceed max deriver input tokens",
    ):
        _make_deriver_settings(
            MAX_INPUT_TOKENS=1024,
            MAX_CUSTOM_INSTRUCTIONS_TOKENS=2048,
        )
