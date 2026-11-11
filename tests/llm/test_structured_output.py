import hashlib
import json
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
    # Kept off the message and traceback, but available to telemetry.
    assert error.raw_content == payload


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


_EXPLICIT = '{"explicit": [{"source_indices": [0], "content": "Ada prefers tabs"}]}'


@pytest.mark.parametrize(
    "payload",
    [
        # Shapes a proxy's "json_tool_call" response_format emulation returns.
        f'{{"name": "json_tool_call", "arguments": {_EXPLICIT}}}',
        f'{{"name": "json_tool_call", "parameters": {_EXPLICIT}}}',
        f'{{"content": {_EXPLICIT}}}',
        # OpenAI-style arguments carried as a JSON-encoded string.
        '{"name": "json_tool_call", "arguments": ' + json.dumps(_EXPLICIT) + "}",
        # Two levels of envelope.
        f'{{"name": "json_tool_call", "arguments": {{"content": {_EXPLICIT}}}}}',
    ],
)
def test_prompt_representation_tool_call_envelope_is_unwrapped(payload: str) -> None:
    result = repair_response_model_json(payload, PromptRepresentation, "test-model")

    assert isinstance(result, PromptRepresentation)
    assert [item.content for item in result.explicit] == ["Ada prefers tabs"]
    assert [item.source_indices for item in result.explicit] == [[0]]


def test_prompt_representation_tool_call_envelope_with_empty_explicit() -> None:
    result = repair_response_model_json(
        '{"name": "json_tool_call", "parameters": {"explicit": []}}',
        PromptRepresentation,
        "test-model",
    )

    assert result == PromptRepresentation(explicit=[])


@pytest.mark.parametrize(
    "payload",
    [
        # Wrong key, no envelope: still a failure.
        '{"all": []}',
        # Envelope around nothing schema-relevant.
        '{"name": "json_tool_call", "arguments": {"wrong": 1}}',
        # Extra non-metadata keys: not a tool-call envelope.
        f'{{"arguments": {_EXPLICIT}, "note": "x"}}',
        # Two payload keys: ambiguous, refuse.
        f'{{"arguments": {_EXPLICIT}, "parameters": {_EXPLICIT}}}',
        # Deeper than the unwrap limit.
        f'{{"content": {{"content": {{"content": {_EXPLICIT}}}}}}}',
    ],
)
def test_prompt_representation_non_envelope_still_raises(payload: str) -> None:
    with pytest.raises(StructuredOutputError, match="failure_class=ValidationError"):
        repair_response_model_json(payload, PromptRepresentation, "test-model")
