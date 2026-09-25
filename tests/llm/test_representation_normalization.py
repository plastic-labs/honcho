"""Regression coverage for provider JSON entering the deriver representation model."""

import pytest

from src.llm.structured_output import repair_response_model_json
from src.utils.representation import PromptRepresentation


@pytest.mark.parametrize(
    ("raw_content", "expected"),
    [
        (
            '["I live in Berlin", {"text": "I use Cubase"}]',
            ["I live in Berlin", "I use Cubase"],
        ),
        ('{"result": {"observations": ["I work remotely"]}}', ["I work remotely"]),
        ('{"explicit": "I am a musician"}', ["I am a musician"]),
    ],
)
def test_repaired_provider_json_preserves_usable_observations(
    raw_content: str, expected: list[str]
) -> None:
    """The JSON repair path must retain the model's supported output shapes."""
    result = repair_response_model_json(raw_content, PromptRepresentation, "test-model")

    assert isinstance(result, PromptRepresentation)
    assert [item.content for item in result.explicit] == expected
