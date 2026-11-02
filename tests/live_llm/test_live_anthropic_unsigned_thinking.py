"""Replaying a thinking block that carries no signature.

A non-Anthropic fallback behind a proxy (MiniMax on 2026-09-21) can serve one
round of an Anthropic-transport tool loop and return a `thinking` block with
`signature: null`. Honcho can't see the proxy switched models, so it replays
that block, and Anthropic rejects the whole next request at
`messages.N.content.0.thinking.signature`. The Anthropic backend removes every
thinking block from such a request, and turns thinking off on budget models.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import pytest

from src.llm.capabilities import anthropic_capabilities
from src.llm.codecs import RenderTarget
from src.llm.messages import ModelOrigin, ToolMessage
from src.llm.profiles import profile_for
from src.llm.request_builder import execute_completion, execute_stream

from .conftest import (
    execute_local_tool,
    favorite_prime_tools,
    make_backend,
    require_provider_key,
)
from .model_matrix import LiveModelSpec, get_live_model_specs

pytestmark = [pytest.mark.live_llm, pytest.mark.requires_anthropic]

ANTHROPIC_THINKING_SPECS = get_live_model_specs(
    provider="anthropic", feature="thinking"
)

TOPICS = ("first", "second", "third")


def prompt(calls: int) -> str:
    topics = ", then ".join(f"topic='{topic}'" for topic in TOPICS[:calls])
    return (
        f"Call the get_favorite_prime tool exactly {calls} times, one call per "
        f"turn, in this order: {topics}. After all results are back, answer in "
        "one sentence that includes the number and the word 'prime'."
    )


def thinking_on(model: str) -> dict[str, Any]:
    """Request kwargs that make `model` think on round 1.

    Adaptive models get max effort, the level at which Claude Haiku 5.5 and
    Claude Sonnet 5.5 reliably think on this prompt.
    """
    if anthropic_capabilities(model).thinking == "adaptive":
        return {
            "thinking": {"type": "adaptive"},
            "extra_body": {"output_config": {"effort": "max"}},
        }
    return {"thinking": {"type": "enabled", "budget_tokens": 2048}}


def thinking_off(model: str) -> dict[str, Any]:
    # Claude 5.x thinks by default, so omitting `thinking` leaves it on. Claude
    # Sonnet 5.5 rejects `disabled`; `between_tools` is its lowest setting.
    if "sonnet-5-5" in model:
        return {"type": "between_tools"}
    return {"type": "disabled"}


def thinking_config(model_spec: LiveModelSpec) -> dict[str, Any]:
    """ModelConfig overrides that turn thinking on through Honcho's backend."""
    if anthropic_capabilities(model_spec.model).thinking == "adaptive":
        return {"thinking_effort": "low"}
    return {"thinking_budget_tokens": 2048}


async def build_fallback_history(
    client: Any, model: str, calls: int = 2
) -> list[dict[str, Any]]:
    """Round 1 from the real model, round 2 as a fallback provider would return it.

    Round 1 runs with thinking on, so the history carries a real signed block.
    Round 2 is the shape the proxy fallback handed back on 2026-09-21: an
    unsigned thinking block ahead of the tool call.
    """
    tools = favorite_prime_tools()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt(calls)}]
    first = await client.messages.create(
        model=model,
        max_tokens=8192,
        tools=tools,
        messages=messages,
        **thinking_on(model),
    )
    tool_use = next((b for b in first.content if b.type == "tool_use"), None)
    assert tool_use is not None, "The model should call the tool in round 1"
    assert any(b.type == "thinking" and b.signature for b in first.content), (
        "Round 1 should emit a signed thinking block"
    )
    messages += [
        {
            "role": "assistant",
            "content": [b.model_dump(exclude_none=True) for b in first.content],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": execute_local_tool(tool_use.name, tool_use.input),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I have the first result. Now call it for 'second'.",
                    "signature": None,
                },
                {
                    "type": "tool_use",
                    "id": "toolu_fallback_01",
                    "name": "get_favorite_prime",
                    "input": {"topic": "second"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_fallback_01",
                    "content": "13",
                }
            ],
        },
    ]
    return messages


def strip_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content: Any = message["content"]
        if message["role"] == "assistant" and isinstance(content, list):
            blocks = cast(list[dict[str, Any]], content)
            content = [
                block
                for block in blocks
                if block["type"] not in {"thinking", "redacted_thinking"}
            ]
        stripped.append({**message, "content": content})
    return stripped


def final_text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def assert_answer(text: str) -> None:
    assert "13" in text
    assert "prime" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["default", "on"])
@pytest.mark.parametrize(
    "model_spec", ANTHROPIC_THINKING_SPECS, ids=lambda spec: spec.id
)
async def test_live_anthropic_rejects_unsigned_thinking_replay(
    model_spec: LiveModelSpec,
    thinking: str,
) -> None:
    """Canary: Anthropic still rejects the unsigned block.

    If this starts passing, the API stopped validating replayed signatures and
    the backend's strip can go.
    """
    require_provider_key(model_spec)
    backend, _ = make_backend(model_spec)
    client = backend._client
    messages = await build_fallback_history(client, model_spec.model)
    extra = thinking_on(model_spec.model) if thinking == "on" else {}

    with pytest.raises(anthropic.BadRequestError, match="signature"):
        await client.messages.create(
            model=model_spec.model,
            max_tokens=8192,
            tools=favorite_prime_tools(),
            messages=messages,
            **extra,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", ["default", "off"])
@pytest.mark.parametrize(
    "model_spec", ANTHROPIC_THINKING_SPECS, ids=lambda spec: spec.id
)
async def test_live_anthropic_accepts_fallback_history_with_thinking_stripped(
    model_spec: LiveModelSpec,
    thinking: str,
) -> None:
    """Round 1 thought, and the model accepts a continuation of that tool loop
    with no thinking left in the history: with thinking at the model default
    (on for Claude 5.x, off for budget models), and turned off."""
    require_provider_key(model_spec)
    backend, _ = make_backend(model_spec)
    client = backend._client
    messages = strip_thinking(await build_fallback_history(client, model_spec.model))
    extra: dict[str, Any] = (
        {"thinking": thinking_off(model_spec.model)} if thinking == "off" else {}
    )

    response = await client.messages.create(
        model=model_spec.model,
        max_tokens=4096,
        tools=favorite_prime_tools(),
        messages=messages,
        **extra,
    )

    assert_answer(final_text(response))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec", ANTHROPIC_THINKING_SPECS, ids=lambda spec: spec.id
)
async def test_live_anthropic_backend_continues_a_tool_loop_after_unsigned_thinking(
    model_spec: LiveModelSpec,
) -> None:
    """Honcho's backend sends the fallback history without a 400, runs one more
    tool round, then answers with the unsigned block still in the history."""
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, **thinking_config(model_spec))
    messages = await build_fallback_history(backend._client, model_spec.model, calls=3)

    third = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=8192,
        tools=favorite_prime_tools(),
    )
    assert third.tool_calls, "The model should call the tool for 'third'"
    # The history so far is already in wire form (it carries the unsigned
    # block), so only the new round is rendered, by the Anthropic codec.
    codec = profile_for("anthropic", model_spec.model).codec
    origin = ModelOrigin(transport="anthropic", model=model_spec.model)
    messages += codec.render(
        [
            codec.parse(third, origin),
            *(
                ToolMessage(
                    tool_call_id=call.id,
                    name=call.name,
                    content=str(execute_local_tool(call.name, call.input)),
                )
                for call in third.tool_calls
            ),
        ],
        RenderTarget("anthropic", model_spec.model),
    ).messages

    answer = await execute_completion(
        backend,
        config,
        messages=messages,
        max_tokens=8192,
        tools=favorite_prime_tools(),
    )

    assert isinstance(answer.content, str)
    assert_answer(answer.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_spec", ANTHROPIC_THINKING_SPECS, ids=lambda spec: spec.id
)
async def test_live_anthropic_backend_streams_after_unsigned_thinking(
    model_spec: LiveModelSpec,
) -> None:
    require_provider_key(model_spec)
    backend, config = make_backend(model_spec, **thinking_config(model_spec))
    messages = await build_fallback_history(backend._client, model_spec.model)

    stream = await execute_stream(
        backend,
        config,
        messages=messages,
        max_tokens=8192,
        tools=favorite_prime_tools(),
    )
    chunks = [chunk async for chunk in stream]

    assert chunks[-1].is_done
    assert_answer("".join(chunk.content for chunk in chunks))
