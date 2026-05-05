"""Tests for the LiteLLM provider backend."""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest import mock

import pytest

from src.llm.backend import CompletionResult, ToolCallResult


def _install_litellm_stub():
    fake = types.ModuleType("litellm")
    fake.acompletion = mock.AsyncMock(name="litellm.acompletion")
    sys.modules["litellm"] = fake
    return fake


@pytest.fixture(autouse=True)
def litellm_stub():
    fake = _install_litellm_stub()
    yield fake
    sys.modules.pop("litellm", None)


def _mock_response(content: str = "Hello!", tool_calls: Any = None):
    from types import SimpleNamespace

    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=usage,
    )


@pytest.mark.asyncio
async def test_complete_calls_acompletion(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response("test reply")

    from src.llm.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend(api_key="sk-test")
    result = await backend.complete(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )

    litellm_stub.acompletion.assert_called_once()
    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["drop_params"] is True
    assert isinstance(result, CompletionResult)
    assert result.content == "test reply"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


@pytest.mark.asyncio
async def test_complete_omits_blank_credentials(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.llm.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend()
    await backend.complete(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


@pytest.mark.asyncio
async def test_complete_forwards_tools(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.llm.backends.litellm import LiteLLMBackend

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    backend = LiteLLMBackend(api_key="k")
    await backend.complete(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Find info"}],
        max_tokens=100,
        tools=tools,
        tool_choice="auto",
    )

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["tool_choice"] == "auto"
    assert len(kwargs["tools"]) == 1


@pytest.mark.asyncio
async def test_complete_parses_tool_calls(litellm_stub):
    from types import SimpleNamespace

    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search", arguments=json.dumps({"q": "test"})),
    )
    litellm_stub.acompletion.return_value = _mock_response("", tool_calls=[tc])

    from src.llm.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend(api_key="k")
    result = await backend.complete(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].input == {"q": "test"}
    assert isinstance(result.tool_calls[0], ToolCallResult)


@pytest.mark.asyncio
async def test_complete_forwards_temperature(litellm_stub):
    litellm_stub.acompletion.return_value = _mock_response()

    from src.llm.backends.litellm import LiteLLMBackend

    backend = LiteLLMBackend(api_key="k")
    await backend.complete(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
        temperature=0.7,
    )

    kwargs = litellm_stub.acompletion.call_args.kwargs
    assert kwargs["temperature"] == 0.7


def test_model_transport_includes_litellm():
    from src.config import ModelTransport
    from typing import get_args

    assert "litellm" in get_args(ModelTransport)
