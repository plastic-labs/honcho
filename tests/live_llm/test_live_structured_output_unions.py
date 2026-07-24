"""Live coverage for union-bearing structured output without tools.

The dialectic's final synthesis call carries response_format but tools=None
(see src/llm/tool_loop.py), and the model class is created dynamically from a
caller-supplied JSON Schema (src/utils/schema_conversion.py). That call shape
differs from test_live_tools_structured_output.py in one important way: with
no tools attached, the Gemini backend uses its NATIVE response_schema config
instead of injecting a schema instruction — and Gemini's response_schema
historically rejected anyOf. These tests drive a schema that exercises every
union-ish construct the converter supports (anyOf with null, a type list, an
enum, a $defs reference) through that exact call shape on all three
providers.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.exceptions import LLMError
from src.llm.backend import CompletionResult
from src.llm.request_builder import execute_completion
from src.llm.structured_output import StructuredOutputError
from src.utils.schema_conversion import json_response_schema_to_pydantic

from .conftest import make_backend, require_provider_key, wrap_async_method
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm]

_USER_FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "favorite_food": {"$ref": "#/$defs/Food"},
        "sentiment": {"enum": ["loves", "likes", "neutral", "dislikes", "hates"]},
        "years_vegetarian": {"type": ["integer", "null"]},
        "salient_fact": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "One short salient fact about the user, or null",
        },
    },
    "required": ["favorite_food", "sentiment", "years_vegetarian"],
    "$defs": {
        "Food": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
    },
}

_PROMPT = (
    "The user said: 'I love sushi. I've been vegetarian for 3 years.' "
    "Report the user's favorite food, their sentiment toward it, how many "
    "years they have been vegetarian, and optionally one salient fact."
)


async def run_union_structured_flow(backend: Any, config: Any) -> CompletionResult:
    """One no-tools turn that must return a schema-conforming answer.

    A fresh model class is created per call, matching how the dialectic
    converts the caller's schema on every request. Retries mirror
    test_live_tools_structured_output.py: empty/unparseable candidates are
    absorbed by the executor's retry layer in production, but these tests
    call the backend directly.
    """
    response_model = json_response_schema_to_pydantic(
        _USER_FACTS_SCHEMA, model_name="UserFactsReport"
    )

    result: CompletionResult | None = None
    last_error: Exception | None = None
    for _ in range(3):
        try:
            result = await execute_completion(
                backend,
                config,
                messages=[{"role": "user", "content": _PROMPT}],
                max_tokens=4096,
                response_format=response_model,
            )
        except (ValidationError, LLMError, StructuredOutputError) as exc:
            last_error = exc
            continue
        break
    if result is None:
        raise AssertionError(
            "union structured turn failed on all attempts"
        ) from last_error

    content = result.content
    assert isinstance(content, BaseModel), f"expected parsed model, got {content!r}"
    # The model class is dynamic, so field access is untyped by construction.
    report: Any = content
    assert "sushi" in report.favorite_food.name.lower()
    assert report.sentiment == "loves"
    assert report.years_vegetarian == 3
    assert report.salient_fact is None or isinstance(report.salient_fact, str)
    return result


@pytest.mark.asyncio
@pytest.mark.requires_anthropic
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="anthropic", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_anthropic_union_structured_output(
    model_spec: LiveModelSpec,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    await run_union_structured_flow(backend, config)


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="openai", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_openai_union_structured_output(
    model_spec: LiveModelSpec,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    await run_union_structured_flow(backend, config)


@pytest.mark.asyncio
@pytest.mark.requires_gemini
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="gemini", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_gemini_union_structured_output(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    generate_calls = wrap_async_method(
        monkeypatch,
        backend._client.aio.models,
        "generate_content",
    )

    await run_union_structured_flow(backend, config)

    # The point of this test: with no tools, Gemini must take the NATIVE
    # response_schema path (the one that historically rejected anyOf), not
    # the prompt-injection workaround used when tools are attached.
    assert generate_calls
    gen_config = generate_calls[-1]["kwargs"]["config"]
    assert "response_schema" in gen_config
    assert gen_config["response_mime_type"] == "application/json"
