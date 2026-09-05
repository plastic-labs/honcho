import hashlib
import traceback

import pytest
from pydantic import BaseModel, ValidationError

from src.llm.structured_output import (
    StructuredOutputError,
    repair_response_model_json,
    validate_structured_output,
)
from src.utils.representation import PromptRepresentation


class OtherResponse(BaseModel):
    answer: str


def test_prompt_representation_malformed_json_raises_safe_error() -> None:
    sentinel = "sentinel-secret-é"
    payload = f"not json {sentinel} {{{{"
    model = "test-model"

    with pytest.raises(StructuredOutputError) as exc_info:
        repair_response_model_json(payload, PromptRepresentation, model)

    error = exc_info.value
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert "JSONDecodeError" in str(error)
    assert model in str(error)
    assert f"payload_bytes={len(payload.encode('utf-8'))}" in str(error)
    assert hashlib.sha256(payload.encode()).hexdigest() in str(error)
    assert payload not in rendered
    assert sentinel not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_prompt_representation_schema_irrelevant_json_raises() -> None:
    with pytest.raises(StructuredOutputError, match="failure_class=ValidationError"):
        repair_response_model_json('{"wrong": 1}', PromptRepresentation, "test-model")


def test_prompt_representation_explicit_empty_is_valid() -> None:
    result = repair_response_model_json(
        '{"explicit": []}', PromptRepresentation, "test-model"
    )

    assert result == PromptRepresentation(explicit=[])


def test_prompt_representation_empty_object_is_valid() -> None:
    expected = PromptRepresentation(explicit=[])

    assert (
        repair_response_model_json("{}", PromptRepresentation, "test-model") == expected
    )
    assert validate_structured_output("{}", PromptRepresentation) == expected
    assert validate_structured_output(expected, PromptRepresentation) == expected


@pytest.mark.parametrize(
    "payload",
    ["```json\n{}\n```", "{}\n\nNothing to extract."],
)
def test_prompt_representation_wrapped_empty_object_is_valid(payload: str) -> None:
    result = repair_response_model_json(payload, PromptRepresentation, "test-model")

    assert result == PromptRepresentation(explicit=[])


def test_prompt_representation_truncated_explicit_json_is_repaired() -> None:
    result = repair_response_model_json(
        '{"explicit":[{"content":"prefers tabs',
        PromptRepresentation,
        "test-model",
    )

    assert isinstance(result, PromptRepresentation)
    assert [item.content for item in result.explicit] == ["prefers tabs"]


def test_other_response_model_preserves_validation_error() -> None:
    with pytest.raises(ValidationError):
        repair_response_model_json("not json", OtherResponse, "test-model")
