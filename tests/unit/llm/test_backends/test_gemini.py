from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from src.exceptions import LLMError
from src.llm.backends.gemini import GeminiBackend
from src.llm.caching import PromptCachePolicy, gemini_cache_store


@pytest.mark.asyncio
async def test_gemini_backend_preserves_thought_signature() -> None:
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Hello from Gemini"),
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    name="search",
                                    args={"query": "honcho"},
                                ),
                                thought_signature="sig_gemini",
                            ),
                        ]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=6,
            ),
            parsed=None,
        )
    )

    backend = GeminiBackend(client)
    result = await backend.complete(
        model="gemini/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=100,
        thinking_budget_tokens=256,
    )

    assert result.content == "Hello from Gemini"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].thought_signature == "sig_gemini"

    await_args = client.aio.models.generate_content.await_args
    if await_args is None:
        raise AssertionError("Expected Gemini generate_content call")
    call = await_args.kwargs
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"]["system_instruction"] == "System prompt"
    assert call["config"]["thinking_config"] == {"thinking_budget": 256}


@pytest.mark.asyncio
async def test_gemini_backend_maps_thinking_effort_to_thinking_level() -> None:
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=SimpleNamespace(parts=[SimpleNamespace(text="ok")]),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=6,
            ),
            parsed=None,
        )
    )

    backend = GeminiBackend(client)
    await backend.complete(
        model="gemini/gemini-3-pro-preview",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        thinking_effort="low",
    )

    await_args = client.aio.models.generate_content.await_args
    if await_args is None:
        raise AssertionError("Expected Gemini generate_content call")
    call = await_args.kwargs
    assert call["config"]["thinking_config"] == {"thinking_level": "low"}


@pytest.mark.asyncio
async def test_gemini_backend_rejects_budget_and_effort_together() -> None:
    backend = GeminiBackend(Mock())

    with pytest.raises(
        ValueError,
        match="does not support sending both thinking_budget_tokens and thinking_effort",
    ):
        await backend.complete(
            model="gemini/gemini-3-pro-preview",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            thinking_budget_tokens=256,
            thinking_effort="low",
        )


@pytest.mark.asyncio
async def test_gemini_backend_raises_on_blocked_response() -> None:
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="SAFETY"),
                    content=SimpleNamespace(parts=[]),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=0,
            ),
            parsed=None,
        )
    )

    backend = GeminiBackend(client)

    with pytest.raises(LLMError, match="Gemini response blocked"):
        await backend.complete(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )


class StructuredResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_gemini_backend_validates_dict_parsed_payload() -> None:
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=SimpleNamespace(parts=[]),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=6,
            ),
            parsed={"answer": "ok"},
            text=None,
            function_calls=None,
        )
    )

    backend = GeminiBackend(client)
    result = await backend.complete(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        response_format=StructuredResponse,
    )

    assert isinstance(result.content, StructuredResponse)
    assert result.content.answer == "ok"


@pytest.mark.asyncio
async def test_gemini_backend_falls_back_to_response_text_and_function_calls() -> None:
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=SimpleNamespace(parts=None),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=6,
            ),
            parsed=None,
            text="13 is prime.",
            function_calls=[
                SimpleNamespace(name="get_favorite_prime", args={"topic": "test"})
            ],
        )
    )

    backend = GeminiBackend(client)
    result = await backend.complete(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    assert result.content == "13 is prime."
    assert result.tool_calls[0].name == "get_favorite_prime"


@pytest.mark.asyncio
async def test_gemini_backend_ignores_mock_text_and_function_call_placeholders() -> (
    None
):
    client = Mock()
    client.aio.models.generate_content = AsyncMock(
        return_value=Mock(
            candidates=[
                Mock(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=None,
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=0,
            ),
            parsed=None,
        )
    )

    backend = GeminiBackend(client)
    result = await backend.complete(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    assert result.content == ""
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_gemini_backend_strips_system_and_tools_when_using_cached_content() -> (
    None
):
    gemini_cache_store._handles.clear()  # pyright: ignore[reportPrivateUsage]
    client = Mock()
    client.aio.caches.create = AsyncMock(
        return_value=SimpleNamespace(
            name="cachedContents/abc123",
            expire_time=datetime.now(timezone.utc),
        )
    )
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="STOP"),
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text="cached result")]
                    ),
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=6,
            ),
            parsed=None,
        )
    )

    backend = GeminiBackend(client)
    result = await backend.complete(
        model="gemini/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=100,
        tools=[
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
        tool_choice="required",
        extra_params={
            "cache_policy": PromptCachePolicy(
                mode="gemini_cached_content",
                ttl_seconds=300,
            )
        },
    )

    assert result.content == "cached result"
    await_args = client.aio.models.generate_content.await_args
    if await_args is None:
        raise AssertionError("Expected Gemini generate_content call")
    call = await_args.kwargs
    assert call["config"]["cached_content"] == "cachedContents/abc123"
    assert "system_instruction" not in call["config"]
    assert "tools" not in call["config"]
    assert "tool_config" not in call["config"]
