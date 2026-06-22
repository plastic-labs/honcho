from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import PromptRepresentation


class StructuredOutputError(ValueError):
    """Raised when structured output cannot be validated or repaired."""


def repair_response_model_json(
    raw_content: str,
    response_model: type[BaseModel],
    _model: str,
) -> BaseModel:
    """Repair truncated or malformed JSON and validate against the response model."""

    try:
        final = validate_and_repair_json(raw_content)
        repaired_data = json.loads(final)

        if (
            response_model is PromptRepresentation
            and "deductive" in repaired_data
            and isinstance(repaired_data["deductive"], list)
        ):
            for item in repaired_data["deductive"]:
                if isinstance(item, dict):
                    if "conclusion" not in item and "premises" in item:
                        if item["premises"]:
                            item["conclusion"] = (
                                f"[Incomplete reasoning from premises: {item['premises'][0][:100]}...]"
                            )
                        else:
                            item["conclusion"] = (
                                "[Incomplete reasoning - conclusion missing]"
                            )
                    if "premises" not in item:
                        item["premises"] = []

        final = json.dumps(repaired_data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        final = ""

    try:
        return response_model.model_validate_json(final)
    except ValidationError:
        if response_model is PromptRepresentation:
            return PromptRepresentation(explicit=[])
        raise


def validate_structured_output(
    content: object,
    response_model: type[BaseModel],
) -> BaseModel:
    if isinstance(content, response_model):
        return content
    if isinstance(content, str):
        return response_model.model_validate_json(content)
    if isinstance(content, dict):
        return response_model.model_validate(content)
    raise StructuredOutputError(
        f"Unsupported structured output payload: {type(content).__name__}"
    )


def empty_structured_output(response_model: type[BaseModel]) -> BaseModel:
    if response_model is PromptRepresentation:
        return PromptRepresentation(explicit=[])
    return response_model.model_validate({})
