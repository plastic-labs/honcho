from __future__ import annotations

import json
import logging
from hashlib import sha256
from typing import cast

from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import PromptRepresentation

logger = logging.getLogger(__name__)


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
        if response_model is PromptRepresentation:
            unwrapped = _unwrap_tool_call_envelope(repaired_data)
            if unwrapped is not None:
                logger.info(
                    "Unwrapped a tool-call envelope around PromptRepresentation "
                    + "structured output from model %s",
                    model,
                )
                repaired_data = unwrapped
                final = json.dumps(unwrapped)
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


# Keys under which a model nests its real answer when it writes the answer out
# as a tool call. An OpenAI-compatible proxy in front of Anthropic implements a
# json_schema response_format as a forced tool named "json_tool_call", and the
# model sometimes returns {"name": "json_tool_call", "arguments": {...}} (or
# "parameters", or a bare {"content": {...}}) instead of the object itself.
_ENVELOPE_PAYLOAD_KEYS = frozenset({"arguments", "parameters", "input", "content"})
_ENVELOPE_META_KEYS = frozenset({"name", "type", "id"})
_MAX_ENVELOPE_DEPTH = 2


def _unwrap_tool_call_envelope(data: object) -> dict[str, object] | None:
    """Return the PromptRepresentation payload nested in a tool-call envelope.

    Only unwraps an object whose keys are exactly one payload key plus
    optional tool-call metadata (``name``/``type``/``id``), and only when the
    nested object carries a PromptRepresentation field. A payload given as a
    JSON-encoded string (the OpenAI ``arguments`` convention) is decoded.
    Anything else returns ``None`` and keeps the normal failure path.
    """
    fields = PromptRepresentation.model_fields
    current: object = data
    for _ in range(_MAX_ENVELOPE_DEPTH):
        if not isinstance(current, dict):
            return None
        mapping = cast(dict[str, object], current)
        if any(field in mapping for field in fields):
            return None
        payload_keys = [key for key in mapping if key in _ENVELOPE_PAYLOAD_KEYS]
        if len(payload_keys) != 1:
            return None
        if not set(mapping) - {payload_keys[0]} <= _ENVELOPE_META_KEYS:
            return None
        inner = mapping[payload_keys[0]]
        if isinstance(inner, str):
            try:
                inner = cast(object, json.loads(inner))
            except json.JSONDecodeError:
                return None
        if isinstance(inner, dict) and any(
            field in cast(dict[str, object], inner) for field in fields
        ):
            return cast(dict[str, object], inner)
        current = cast(object, inner)
    return None


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
