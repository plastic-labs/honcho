import pytest
from pydantic import ValidationError

from src.config import DeriverSettings


class TestDeriverSettings:
    def test_requires_explicit_custom_instruction_token_limit(self) -> None:
        settings = DeriverSettings(MAX_INPUT_TOKENS=2000)

        with pytest.raises(ValueError) as exc_info:
            _ = settings.effective_max_custom_instructions_tokens

        assert "MAX_CUSTOM_INSTRUCTIONS_TOKENS" in str(exc_info.value)
        assert "config.toml" in str(exc_info.value)

    def test_uses_explicit_custom_instruction_token_limit(self) -> None:
        settings = DeriverSettings(
            MAX_INPUT_TOKENS=23000,
            MAX_CUSTOM_INSTRUCTIONS_TOKENS=777,
        )

        assert settings.effective_max_custom_instructions_tokens == 777

    def test_rejects_explicit_custom_instruction_token_limit_above_input_budget(
        self,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            DeriverSettings(
                MAX_INPUT_TOKENS=1100,
                REPRESENTATION_BATCH_MAX_TOKENS=512,
                MAX_CUSTOM_INSTRUCTIONS_TOKENS=1101,
            )

        assert "MAX_CUSTOM_INSTRUCTIONS_TOKENS" in str(exc_info.value)
