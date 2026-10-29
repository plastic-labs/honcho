"""Golden render tests for the provider codecs.

Each transport renders one tool round (system, user, an assistant turn with
reasoning and two tool calls, both results, a budget line) to the exact dicts
its backend receives. A turn produced on one transport and rendered for another
carries no reasoning and no foreign block types.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, ToolCallResult
from src.llm.codecs import RenderTarget
from src.llm.messages import (
    AIMessage,
    ChatMessage,
    ModelOrigin,
    SystemMessage,
    ToolMessage,
    UserMessage,
    coerce_messages,
)
from src.llm.profiles import profile_for

ANTHROPIC = ModelOrigin(transport="anthropic", model="claude-sonnet-5-5")
OPENAI = ModelOrigin(
    transport="openai", model="google/gemini-3-pro", base_url="https://openrouter.ai"
)
GEMINI = ModelOrigin(transport="gemini", model="gemini-3-pro")

THINKING = {"type": "thinking", "thinking": "Check both.", "signature": "sig-1"}
REDACTED = {"type": "redacted_thinking", "data": "opaque"}
DETAIL = {"type": "reasoning.encrypted", "data": "enc", "index": 0}

CALLS = [
    ToolCallResult(id="call_a", name="search_memory", input={"query": "tea"}),
    ToolCallResult(
        id="call_b",
        name="search_messages",
        input={"query": "coffee", "limit": 2},
        thought_signature=b"\x01sig",
    ),
]


def _target(origin: ModelOrigin) -> RenderTarget:
    return RenderTarget(origin.transport, origin.model, origin.base_url)


def _round(result: CompletionResult, origin: ModelOrigin) -> list[ChatMessage]:
    ai = profile_for(origin.transport, origin.model).codec.parse(result, origin)
    return [
        SystemMessage(content="You recall."),
        UserMessage(content="What does she drink?"),
        ai,
        ToolMessage(tool_call_id="call_a", name="search_memory", content="green tea"),
        ToolMessage(
            tool_call_id="call_b",
            name="search_messages",
            content="Error: timeout",
            is_error=True,
        ),
        UserMessage(content="1 of 3 rounds used."),
    ]


def _render(messages: list[ChatMessage], origin: ModelOrigin) -> list[dict[str, Any]]:
    return profile_for(origin.transport, origin.model).codec.render(
        messages, _target(origin)
    )


HEAD = [
    {"role": "system", "content": "You recall."},
    {"role": "user", "content": "What does she drink?"},
]
BUDGET = {"role": "user", "content": "1 of 3 rounds used."}


def test_anthropic_round_renders_native_blocks() -> None:
    result = CompletionResult(
        content="Looking.",
        tool_calls=CALLS,
        thinking_content="Check both.",
        thinking_blocks=[THINKING, REDACTED],
    )
    assert _render(_round(result, ANTHROPIC), ANTHROPIC) == [
        *HEAD,
        {
            "role": "assistant",
            "content": [
                THINKING,
                REDACTED,
                {"type": "text", "text": "Looking."},
                {
                    "type": "tool_use",
                    "id": "call_a",
                    "name": "search_memory",
                    "input": {"query": "tea"},
                },
                {
                    "type": "tool_use",
                    "id": "call_b",
                    "name": "search_messages",
                    "input": {"query": "coffee", "limit": 2},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": "green tea",
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_b",
                    "content": "Error: timeout",
                    "is_error": True,
                },
            ],
        },
        BUDGET,
    ]


def test_openai_round_renders_tool_calls_and_reasoning_details() -> None:
    result = CompletionResult(
        content=None,
        tool_calls=CALLS,
        thinking_content="summary",
        reasoning_details=[DETAIL],
    )
    assert _render(_round(result, OPENAI), OPENAI) == [
        *HEAD,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "search_memory",
                        "arguments": '{"query": "tea"}',
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "search_messages",
                        "arguments": '{"query": "coffee", "limit": 2}',
                    },
                },
            ],
            "reasoning_details": [DETAIL],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "green tea"},
        {"role": "tool", "tool_call_id": "call_b", "content": "Error: timeout"},
        BUDGET,
    ]


def test_openai_reasoning_content_replays_without_details() -> None:
    result = CompletionResult(
        content="", tool_calls=CALLS[:1], thinking_content="Let me look."
    )
    rendered = _render(_round(result, OPENAI), OPENAI)
    assert rendered[2]["content"] == ""
    assert rendered[2]["reasoning_content"] == "Let me look."
    assert "reasoning_details" not in rendered[2]


def test_gemini_round_renders_parts_with_signatures() -> None:
    result = CompletionResult(content="Looking.", tool_calls=CALLS)
    assert _render(_round(result, GEMINI), GEMINI) == [
        *HEAD,
        {
            "role": "model",
            "parts": [
                {"text": "Looking."},
                {"function_call": {"name": "search_memory", "args": {"query": "tea"}}},
                {
                    "function_call": {
                        "name": "search_messages",
                        "args": {"query": "coffee", "limit": 2},
                    },
                    "thought_signature": b"\x01sig",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "search_memory",
                        "response": {"result": "green tea"},
                    }
                },
                {
                    "function_response": {
                        "name": "search_messages",
                        "response": {"result": "Error: timeout"},
                    }
                },
            ],
        },
        BUDGET,
    ]


def test_anthropic_turn_renders_for_other_transports_without_reasoning() -> None:
    result = CompletionResult(
        content="Looking.", tool_calls=CALLS, thinking_blocks=[THINKING, REDACTED]
    )
    history = _round(result, ANTHROPIC)

    openai = _render(history, OPENAI)
    assert openai[2]["content"] == "Looking."
    assert "reasoning_details" not in openai[2]
    assert "reasoning_content" not in openai[2]
    assert [m["role"] for m in openai] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]

    gemini = _render(history, GEMINI)
    assert gemini[2]["role"] == "model"
    assert all("thought_signature" not in part for part in gemini[2]["parts"])


def test_openai_turn_renders_for_anthropic_as_blocks() -> None:
    result = CompletionResult(
        content=None, tool_calls=CALLS[:1], reasoning_details=[DETAIL]
    )
    rendered = _render(_round(result, OPENAI), ANTHROPIC)
    assert rendered[2] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_a",
                "name": "search_memory",
                "input": {"query": "tea"},
            }
        ],
    }
    assert [block["type"] for block in rendered[3]["content"]] == [
        "tool_result",
        "tool_result",
    ]


def test_anthropic_thinking_replays_to_another_claude_model() -> None:
    result = CompletionResult(tool_calls=CALLS[:1], thinking_blocks=[THINKING])
    other = ModelOrigin(transport="anthropic", model="claude-opus-5-5")
    assert _render(_round(result, ANTHROPIC), other)[2]["content"][0] == THINKING


@pytest.mark.parametrize(
    "target",
    [
        ModelOrigin(transport="openai", model="other", base_url=OPENAI.base_url),
        ModelOrigin(transport="openai", model=OPENAI.model, base_url=None),
    ],
)
def test_openai_reasoning_stays_with_its_endpoint_and_model(
    target: ModelOrigin,
) -> None:
    result = CompletionResult(tool_calls=CALLS[:1], reasoning_details=[DETAIL])
    assert "reasoning_details" not in _render(_round(result, OPENAI), target)[2]


def test_gemini_signature_dropped_for_other_transports() -> None:
    result = CompletionResult(tool_calls=CALLS)
    rendered = _render(_round(result, GEMINI), ANTHROPIC)
    assert [block["type"] for block in rendered[2]["content"]] == [
        "tool_use",
        "tool_use",
    ]


@pytest.mark.parametrize("origin", [ANTHROPIC, OPENAI, GEMINI])
def test_caller_assistant_turn_stays_a_string(origin: ModelOrigin) -> None:
    history = coerce_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert _render(history, origin) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_coerce_messages_types_caller_dicts() -> None:
    history = coerce_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert history == [
        SystemMessage(content="s"),
        UserMessage(content="u"),
        AIMessage(text="a"),
    ]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "x", "tool_call_id": "1"},
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "model", "parts": [{"text": "x"}]},
    ],
)
def test_coerce_messages_rejects_provider_shapes(message: dict[str, Any]) -> None:
    with pytest.raises(ValidationException):
        coerce_messages([message])
