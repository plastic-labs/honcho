"""Live checks for Claude models that bind replayed thinking to the request.

On these models (Opus/Sonnet/Haiku 5.5, Fable 5.1) a replayed thinking block is
rejected once the tool set or anything before it changed, and Opus/Sonnet 5.5
reject a forced tool_choice. Each tool-loop test opts into enforcement with
`thinking.block_binding.prefix_mismatch_behavior: "error"`, so an edit the loop
fails to account for is a 400 here even on an account that is not enforced by
default, and runs with retries off so the 400 is not retried or masked.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from src.config import ModelConfig
from src.llm import client_for_model_config, honcho_llm_call
from src.llm.capabilities import anthropic_capabilities
from src.llm.request_builder import execute_completion
from src.utils.json_coerce import as_dict, as_list

from .conftest import (
    REASONING_TOOL_PROMPT,
    favorite_prime_tools,
    make_backend,
    require_provider_key,
    wrap_async_method,
)
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_anthropic]

PRESERVED_THINKING_SPECS = tuple(
    spec
    for spec in get_live_model_specs(provider="anthropic")
    if anthropic_capabilities(spec.model).preserved_thinking
)

CHECK_NUMBER_TOOL = {
    "name": "check_number",
    "description": "Look up the notes on file for one number.",
    "input_schema": {
        "type": "object",
        "properties": {"number": {"type": "integer"}},
        "required": ["number"],
    },
}
LOOP_PROMPT = (
    "For each of 91, 97 and 119, in that order: first work out whether it is "
    "prime by checking its divisors, then call check_number once for it. Make "
    "one call per turn. After all three calls, report which of them are prime."
)
# One tool round is enough for the response_model test to replay thinking, and
# asking for one call per turn makes some models end a turn with prose instead.
REPORT_PROMPT = (
    "For each of 4087, 4093 and 4097, work out whether it is prime by checking "
    "its divisors, without writing out your work, and call check_number once for "
    "it. Then report which of them are prime."
)
# ~1.5k tokens per result, so the loop crosses the input cap used below.
_FILLER = " ".join(f"note{i}" for i in range(1500))


class PrimeReport(BaseModel):
    primes: list[int]
    summary: str


async def _check_number(_name: str, tool_input: dict[str, Any]) -> str:
    return f"{tool_input.get('number')}: notes on file. {_FILLER}"


def _enforced_config(model_spec: LiveModelSpec) -> ModelConfig:
    return ModelConfig(
        model=model_spec.model,
        transport="anthropic",
        provider_params={
            "extra_body": {
                "thinking": {
                    "type": "adaptive",
                    "block_binding": {"prefix_mismatch_behavior": "error"},
                }
            },
            "extra_headers": {"anthropic-beta": "thinking-binding-controls-2026-08-01"},
        },
    )


def _thinking_produced(responses: list[Any]) -> int:
    """Thinking blocks the model produced across every response."""
    return sum(
        1
        for response in responses
        for block in response.content
        if getattr(block, "type", None) == "thinking"
    )


def _thinking_replayed(requests: list[dict[str, Any]]) -> int:
    """Thinking blocks sent back to the model across every request."""
    return sum(
        1
        for request in requests
        for message in cast(list[dict[str, Any]], request["messages"])
        for block in as_list(message.get("content")) or ()
        if (entry := as_dict(block)) is not None and entry.get("type") == "thinking"
    )


def _record_client(
    monkeypatch: pytest.MonkeyPatch, config: ModelConfig
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Record every request and raw response of the Anthropic client for `config`."""
    client: Any = client_for_model_config("anthropic", config)
    messages_api = client.messages
    original = messages_api.create
    requests: list[dict[str, Any]] = []
    responses: list[Any] = []

    async def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        response = await original(**kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(messages_api, "create", create)
    return requests, responses


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", PRESERVED_THINKING_SPECS, ids=lambda s: s.id)
async def test_live_forced_tool_choice_follows_capabilities(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced tool_choice is relaxed where the model rejects it (Sonnet/Opus
    5.5) and kept where it is accepted (Haiku 5.5); either way, no 400."""
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, thinking_budget_tokens=1024)
    create_calls = wrap_async_method(monkeypatch, backend._client.messages, "create")

    result = await execute_completion(
        backend,
        config,
        messages=[{"role": "user", "content": REASONING_TOOL_PROMPT}],
        max_tokens=4096,
        tools=favorite_prime_tools(),
        tool_choice="required",
    )

    forced = anthropic_capabilities(model_spec.model).forced_tool_choice
    expected = {"type": "auto"} if forced == "rejected" else {"type": "any"}
    assert create_calls[0]["kwargs"]["tool_choice"] == expected
    assert result.tool_calls, "the model should still call the tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", PRESERVED_THINKING_SPECS, ids=lambda s: s.id)
async def test_live_response_model_tool_loop_keeps_replayed_thinking_valid(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With response_model in a tool loop, the schema instruction must not move
    between rounds, or every replayed thinking block is a 400."""
    require_provider_key(model_spec)
    config = _enforced_config(model_spec)
    requests, responses = _record_client(monkeypatch, config)

    # Now and then the model ends with prose instead of the JSON, or the API
    # returns an empty turn. Retry only that; an API error such as the 400 this
    # test is for fails on the first attempt.
    response = None
    for attempt in range(3):
        requests.clear()
        responses.clear()
        try:
            response = await honcho_llm_call(
                model_config=config,
                prompt=REPORT_PROMPT,
                max_tokens=4096,
                tools=[CHECK_NUMBER_TOOL],
                tool_choice="auto",
                tool_executor=_check_number,
                max_tool_iterations=6,
                response_model=PrimeReport,
                enable_retry=False,
            )
            break
        except ValidationError:
            if attempt == 2:
                raise
    assert response is not None

    # Nothing truncates here, so every thinking block is replayed in later rounds.
    if not _thinking_replayed(requests[1:]):
        pytest.skip(f"{model_spec.model} produced no thinking to replay")
    assert _thinking_produced(responses)
    assert isinstance(response.content, PrimeReport)
    assert response.content.primes == [4093]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", PRESERVED_THINKING_SPECS, ids=lambda s: s.id)
async def test_live_truncation_and_synthesis_keep_replayed_thinking_valid(
    model_spec: LiveModelSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncation rewrites earlier turns and the synthesis call drops the tools;
    thinking produced before either must not be replayed after it."""
    require_provider_key(model_spec)
    config = _enforced_config(model_spec)
    requests, responses = _record_client(monkeypatch, config)

    response = await honcho_llm_call(
        model_config=config,
        prompt="",
        messages=[
            {"role": "system", "content": "You check numbers with check_number."},
            {"role": "user", "content": LOOP_PROMPT},
        ],
        max_tokens=4096,
        tools=[CHECK_NUMBER_TOOL],
        tool_choice="auto",
        tool_executor=_check_number,
        max_tool_iterations=3,
        max_input_tokens=2600,
        enable_retry=False,
    )

    # Thinking produced before a truncation or the tool-set change has to be
    # dropped from later requests rather than replayed, so what matters is that
    # the model produced some and every request still went through.
    if not _thinking_produced(responses[:-1]):
        pytest.skip(f"{model_spec.model} produced no thinking before the last call")
    assert response.hit_input_token_cap, "the loop should have truncated history"
    assert response.hit_iteration_limit, "the loop should have reached synthesis"
    assert requests[-1].get("tools") is None
    assert _thinking_replayed(requests[-1:]) == 0, (
        "the synthesis call changes the tool set, so no earlier thinking is replayed"
    )
