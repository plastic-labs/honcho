from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from typing import cast

from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import PromptRepresentation


class StructuredOutputError(ValueError):
    """Raised when structured output cannot be validated or repaired.

    ``raw_content`` is the model's unparsed output, when the raising site has
    it. It stays out of the message (which reaches logs and the queue item's
    error column) and is read only by telemetry, so a failed generation can be
    inspected in the same sinks that already hold successful ones.
    """

    def __init__(self, message: str, *, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content: str | None = raw_content
        # What the provider reported for the failed response, when the raising
        # backend had one: a "length" finish with output at max_tokens means
        # truncation, a "stop" finish means the model answered the wrong shape.
        self.finish_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None


@contextmanager
def response_details_on_error(
    *,
    finish_reason: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Generator[None]:
    """Attach the provider response's finish reason and usage to a
    StructuredOutputError raised inside, for telemetry; the backend raises
    before it builds a CompletionResult, so this is the only record of them."""
    try:
        yield
    except StructuredOutputError as exc:
        exc.finish_reason = exc.finish_reason or finish_reason
        if exc.input_tokens is None:
            exc.input_tokens = input_tokens
        if exc.output_tokens is None:
            exc.output_tokens = output_tokens
        raise


def schema_instruction(response_format: type[BaseModel], *, tools_present: bool) -> str:
    """Structured-output instruction appended to the conversation for
    providers without native (or tools-compatible) schema enforcement.

    When tools are in play the wording is conditional so the model remains
    free to emit tool calls; validation then relies on parse + repair.
    """
    schema_json = json.dumps(response_format.model_json_schema(), indent=2)
    if tools_present:
        return (
            "\n\nIf not responding with a tool call, respond with valid JSON "
            f"matching this schema:\n{schema_json}"
        )
    return f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"


def repair_response_model_json(
    raw_content: str,
    response_model: type[BaseModel],
    model: str,
) -> BaseModel:
    """Repair truncated or malformed JSON and validate against the response model."""

    failure_class = "ValidationError"
    try:
        final = validate_and_repair_json(raw_content)
        repaired_data = cast(object, json.loads(final))
        repaired_mapping = (
            cast(dict[str, object], repaired_data)
            if isinstance(repaired_data, dict)
            else None
        )
        raw_is_empty_mapping = False
        if response_model is PromptRepresentation and repaired_mapping == {}:
            raw_is_empty_mapping = _starts_with_empty_json_object(raw_content)
            if not raw_is_empty_mapping:
                failure_class = "JSONDecodeError"

        if response_model is PromptRepresentation and (
            repaired_mapping is None
            or (
                not raw_is_empty_mapping
                and not any(
                    field in repaired_mapping for field in response_model.model_fields
                )
            )
        ):
            try:
                json.loads(raw_content)
            except json.JSONDecodeError:
                failure_class = "JSONDecodeError"
            else:
                failure_class = "ValidationError"
            final = ""
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        failure_class = type(exc).__name__
        final = ""

    try:
        return response_model.model_validate_json(final)
    except ValidationError:
        if response_model is not PromptRepresentation:
            raise

    payload = raw_content.encode("utf-8")
    details = f"failure_class={failure_class} model={model}"
    payload_details = (
        f"payload_bytes={len(payload)} payload_sha256={sha256(payload).hexdigest()}"
    )
    raise StructuredOutputError(
        f"PromptRepresentation structured output failed {details} {payload_details}",
        raw_content=raw_content,
    ) from None


def validate_structured_output(
    content: object,
    response_model: type[BaseModel],
) -> BaseModel:
    if isinstance(content, response_model):
        if not _is_schema_relevant(content, response_model):
            raise StructuredOutputError("Structured output has no schema fields")
        return content
    if isinstance(content, str):
        if response_model is PromptRepresentation:
            parsed = cast(object, json.loads(content))
            if not _is_schema_relevant(parsed, response_model):
                raise StructuredOutputError("Structured output has no schema fields")
        return response_model.model_validate_json(content)
    if isinstance(content, dict):
        if not _is_schema_relevant(cast(object, content), response_model):
            raise StructuredOutputError("Structured output has no schema fields")
        return response_model.model_validate(content)
    raise StructuredOutputError(
        f"Unsupported structured output payload: {type(content).__name__}"
    )


def _is_schema_relevant(content: object, response_model: type[BaseModel]) -> bool:
    if response_model is not PromptRepresentation:
        return True
    if isinstance(content, BaseModel):
        return True
    return isinstance(content, dict) and (
        not content or any(field in content for field in response_model.model_fields)
    )


def _starts_with_empty_json_object(raw_content: str) -> bool:
    candidate = raw_content.strip()
    if candidate.startswith("```"):
        _, separator, candidate = candidate.partition("\n")
        if not separator:
            return False
        candidate = candidate.lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and not value


def empty_structured_output(response_model: type[BaseModel]) -> BaseModel:
    if response_model is PromptRepresentation:
        return PromptRepresentation(explicit=[])
    return response_model.model_validate({})
