"""Regression tests for https://github.com/plastic-labs/honcho/issues/676.

OpenAI-compatible gateways can return an HTTP-successful response without a
usable shape: empty ``choices``, ``None`` choices, a ``None`` message, or a
missing ``usage`` attribute. The backend must surface a controlled ``LLMError``
(or degrade gracefully for usage) instead of crashing with a raw
``IndexError``/``AttributeError``/``TypeError``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.exceptions import LLMError
from src.llm.backends.openai import OpenAIBackend


def _client_returning(response: object) -> Mock:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


async def _complete(client: Mock, **overrides: object):
    backend = OpenAIBackend(client)
    kwargs = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        **overrides,
    }
    return await backend.complete(**kwargs)


async def test_empty_choices_raises_llm_error() -> None:
    client = _client_returning(SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMError):
        await _complete(client)


async def test_none_choices_raises_llm_error() -> None:
    client = _client_returning(SimpleNamespace(choices=None, usage=None))
    with pytest.raises(LLMError):
        await _complete(client)


async def test_none_message_raises_llm_error() -> None:
    client = _client_returning(
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=None)],
            usage=None,
        )
    )
    with pytest.raises(LLMError):
        await _complete(client)


async def test_non_sdk_object_raises_llm_error() -> None:
    client = _client_returning({"choices": [], "usage": {}})
    with pytest.raises(LLMError):
        await _complete(client)


async def test_truthy_non_indexable_choices_raises_llm_error() -> None:
    client = _client_returning({"choices": {"0": {"message": {}}}, "usage": {}})
    with pytest.raises(LLMError):
        await _complete(client)


async def test_truthy_scalar_choices_raises_llm_error() -> None:
    client = _client_returning(SimpleNamespace(choices=5, usage=None))
    with pytest.raises(LLMError):
        await _complete(client)


async def test_missing_usage_attribute_degrades_to_zero_tokens() -> None:
    message = SimpleNamespace(content="Hello", tool_calls=None)
    client = _client_returning(
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
        )
    )
    result = await _complete(client)
    assert result.content == "Hello"
    assert result.input_tokens == 0
    assert result.output_tokens == 0


async def test_json_object_path_with_empty_choices_raises_llm_error() -> None:
    client = _client_returning(SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMError):
        await _complete(
            client,
            response_format={"type": "json_object"},
            extra_params={"json_mode": True},
        )
