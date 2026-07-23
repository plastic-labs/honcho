"""Live coverage for combining tool calling with structured output.

Each provider needs a different workaround when a request carries both
function tools and a response_format (see src/llm/backends/):

- OpenAI: parse() rejects non-strict function tools with a 500, so
  tool-carrying structured requests must go through create() with an
  explicit json_schema response_format.
- Anthropic: the '{' assistant prefill suppresses tool_use blocks, so it
  must be skipped when tools are present (conditional instruction +
  parse/repair instead).
- Gemini: native response_schema + function calling is rejected before
  Gemini 3, so a schema instruction is injected into the final turn.

The flow below drives both halves of the combination against real APIs:
a first turn that must produce a tool call (structured parsing skipped),
and a replay turn that must produce a schema-conforming final answer
while tools are still attached.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.exceptions import LLMError
from src.llm.backend import CompletionResult
from src.llm.history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    HistoryAdapter,
    OpenAIHistoryAdapter,
)
from src.llm.request_builder import execute_completion
from src.llm.structured_output import StructuredOutputError

from .conftest import (
    execute_local_tool,
    favorite_prime_tools,
    make_backend,
    require_provider_key,
    wrap_async_method,
)
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm]

_OPENAI_TOOL_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="openai", feature="structured_output")
    if spec.family in {"gpt_4_class", "gpt_5_class"}
)


class FavoritePrimeReport(BaseModel):
    number: int
    is_prime: bool
    summary: str


_INITIAL_PROMPT = (
    "Before answering, call the get_favorite_prime tool exactly once. "
    "Do not answer with plain text or JSON on this turn. "
    "After you receive the tool result, answer with JSON where number is "
    "the tool's number, is_prime says whether that number is prime, and "
    "summary is one short sentence."
)


async def run_tool_then_structured_flow(
    backend: Any,
    config: Any,
    adapter: HistoryAdapter,
) -> tuple[CompletionResult, CompletionResult]:
    """First turn must tool-call (parsing skipped); replay turn must return
    a schema-conforming answer with tools still attached."""
    initial_messages = [{"role": "user", "content": _INITIAL_PROMPT}]
    tools = favorite_prime_tools()

    first = await execute_completion(
        backend,
        config,
        messages=initial_messages,
        max_tokens=4096,
        tools=tools,
        tool_choice="required",
        response_format=FavoritePrimeReport,
    )

    assert first.tool_calls, "first turn should issue a tool call"
    assert not isinstance(
        first.content, FavoritePrimeReport
    ), "tool-call turns carry no consumable content and must not be parsed"

    tool_call = first.tool_calls[0]
    tool_result = execute_local_tool(tool_call.name, tool_call.input)
    replay_messages = initial_messages + [
        adapter.format_assistant_tool_message(first),
        *adapter.format_tool_results(
            [
                {
                    "tool_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "result": tool_result,
                }
            ]
        ),
    ]

    # tool_choice stays "auto" on the replay turn — the production tool loop
    # (dialectic) never forces "none", and gemini-2.5-flash is prone to
    # returning empty candidates under NONE mode. Retry the turn on empty /
    # unparseable candidates (in production the executor's retry layer
    # absorbs those; this calls the backend directly) and on the rare run
    # where the model chooses to tool-call again instead of answering.
    second: CompletionResult | None = None
    last_error: Exception | None = None
    for _ in range(3):
        try:
            candidate = await execute_completion(
                backend,
                config,
                messages=replay_messages,
                max_tokens=4096,
                tools=tools,
                tool_choice="auto",
                response_format=FavoritePrimeReport,
            )
        except (ValidationError, LLMError, StructuredOutputError) as exc:
            last_error = exc
            continue
        if candidate.tool_calls:
            last_error = AssertionError(
                "model issued another tool call instead of answering"
            )
            continue
        second = candidate
        break
    if second is None:
        raise AssertionError(
            "structured replay turn failed on all attempts"
        ) from last_error

    assert isinstance(second.content, FavoritePrimeReport)
    assert second.content.number == 13
    assert second.content.is_prime is True
    return first, second


@pytest.mark.asyncio
@pytest.mark.requires_anthropic
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="anthropic", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_anthropic_tools_with_structured_output(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec)
    create_calls = wrap_async_method(monkeypatch, backend._client.messages, "create")

    await run_tool_then_structured_flow(backend, config, AnthropicHistoryAdapter())

    assert len(create_calls) >= 2
    for call in create_calls:
        messages = call["kwargs"]["messages"]
        assert messages[-1] != {
            "role": "assistant",
            "content": "{",
        }, "the '{' prefill would suppress tool_use blocks"
        assert "If not responding with a tool call" in str(
            messages[-1]
        ), "schema instruction should use the conditional wording with tools"


@pytest.mark.asyncio
@pytest.mark.requires_openai
@pytest.mark.parametrize("model_spec", _OPENAI_TOOL_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_tools_with_structured_output(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    # gpt-5.4 rejects function tools combined with any explicit
    # reasoning_effort other than 'none' on /v1/chat/completions, so leave
    # the parameter unset and let the server default apply.
    backend, config = make_backend(model_spec)
    parse_calls = wrap_async_method(
        monkeypatch, backend._client.chat.completions, "parse"
    )
    create_calls = wrap_async_method(
        monkeypatch, backend._client.chat.completions, "create"
    )

    await run_tool_then_structured_flow(backend, config, OpenAIHistoryAdapter())

    # favorite_prime_tools() is deliberately non-strict, the exact shape
    # parse() refuses with a 500.
    assert not parse_calls, "tool-carrying structured requests must avoid parse()"
    assert len(create_calls) >= 2
    for call in create_calls:
        response_format = call["kwargs"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "FavoritePrimeReport"


@pytest.mark.asyncio
@pytest.mark.requires_gemini
@pytest.mark.parametrize(
    "model_spec",
    get_live_model_specs(provider="gemini", feature="structured_output"),
    ids=lambda spec: spec.id,
)
async def test_live_gemini_tools_with_structured_output(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_provider_key(model_spec)
    # No temperature pin: gemini-2.5-flash occasionally returns an empty
    # candidate on the replay turn, and at temperature=0 the retry re-sends
    # a deterministic request — default sampling gives retries a real chance.
    backend, config = make_backend(model_spec)
    generate_calls = wrap_async_method(
        monkeypatch,
        backend._client.aio.models,
        "generate_content",
    )

    await run_tool_then_structured_flow(backend, config, GeminiHistoryAdapter())

    assert len(generate_calls) >= 2
    for call in generate_calls:
        gen_config = call["kwargs"]["config"]
        assert (
            "response_schema" not in gen_config
        ), "native response_schema + function calling is rejected pre-Gemini 3"
        assert "response_mime_type" not in gen_config
        contents = call["kwargs"]["contents"]
        assert "matching this schema" in str(
            contents[-1]
        ), "schema instruction should be injected into the final turn"
