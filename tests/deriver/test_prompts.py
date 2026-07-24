from collections.abc import AsyncGenerator, Generator
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)
from src.utils.representation import PromptRepresentation


@pytest.fixture(autouse=True)
async def clean_queue_tables() -> AsyncGenerator[None, None]:
    """Prompt-only tests do not need queue table cleanup."""
    yield


@pytest.fixture(autouse=True)
def mock_tracked_db() -> Generator[None, None, None]:
    """Prompt-only tests do not need tracked_db patching."""
    yield


def test_minimal_deriver_prompt_includes_custom_instructions_when_present() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions="Prefer concrete timeline facts.",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "Prefer concrete timeline facts." in prompt


def test_minimal_deriver_prompt_omits_custom_instructions_when_absent() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions=None,
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_minimal_deriver_prompt_describes_explicit_json_object_shape() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: I just had my 25th birthday",
        custom_instructions=None,
    )

    assert '"explicit": [{"content": "fact 1"}' in prompt
    assert 'Each item in "explicit" must be an object with a "content" field' in prompt
    assert "not a bare string" in prompt
    with pytest.raises(ValidationError):
        PromptRepresentation.model_validate_json(
            '{"explicit": ["alice is 25 years old"]}'
        )
    assert PromptRepresentation.model_validate_json(
        '{"explicit": [{"content": "alice is 25 years old"}]}'
    ).explicit[0].content == "alice is 25 years old"
    explicit_schema_description = PromptRepresentation.model_json_schema()["properties"][
        "explicit"
    ]["description"]
    assert '{"content": "The user is 25 years old"}' in explicit_schema_description
    assert "['The user is 25 years old'" not in explicit_schema_description


def test_estimate_deriver_prompt_tokens_increases_with_custom_instructions() -> None:
    base_tokens = estimate_minimal_deriver_prompt_tokens()
    custom_tokens = estimate_deriver_prompt_tokens(
        "Prefer explicit facts with absolute dates and keep the subject precise."
    )

    assert custom_tokens > base_tokens


def test_estimate_deriver_prompt_tokens_propagates_token_estimation_errors() -> None:
    estimate_minimal_deriver_prompt_tokens.cache_clear()

    with patch(
        "src.deriver.prompts.estimate_tokens",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens(None)

        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens("Prefer concrete facts.")
