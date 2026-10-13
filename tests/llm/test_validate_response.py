"""honcho_llm_call's validate_response hook runs inside the retried call."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.config import ModelConfig, ResolvedFallbackConfig
from src.exceptions import ValidationException
from src.llm import HonchoLLMCallResponse, honcho_llm_call

_CONFIG = ModelConfig(
    model="primary-model",
    transport="anthropic",
    api_key="test-key",
    fallback=ResolvedFallbackConfig(
        model="fallback-model", transport="openai", api_key="test-key"
    ),
)


class Refused(Exception):
    pass


def _noop_tool(_name: str, _args: dict[str, Any]) -> None:
    return None


def _response(content: str) -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content=content, output_tokens=1, finish_reasons=["stop"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_input_tokens",
    [
        None,
        # Takes the message-truncating path the deriver uses.
        1000,
    ],
)
async def test_refusal_is_retried_through_to_the_fallback(
    max_input_tokens: int | None,
) -> None:
    """A response the hook refuses consumes an attempt, so the final attempt
    goes to the fallback model and its accepted response is returned."""
    models: list[str] = []

    async def inner(_provider: str, model: str, *_args: Any, **_kwargs: Any) -> Any:
        models.append(model)
        return _response("good" if model == "fallback-model" else "")

    def validate(response: HonchoLLMCallResponse[Any]) -> None:
        if not response.content:
            raise Refused

    with (
        patch("src.llm.api.honcho_llm_call_inner", side_effect=inner),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await honcho_llm_call(
            model_config=_CONFIG,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            max_input_tokens=max_input_tokens,
            validate_response=validate,
        )

    assert models == ["primary-model", "primary-model", "fallback-model"]
    assert result.content == "good"


@pytest.mark.asyncio
async def test_refusal_on_every_attempt_raises_the_hook_error() -> None:
    inner = AsyncMock(return_value=_response(""))

    def validate(_response: HonchoLLMCallResponse[Any]) -> None:
        raise Refused

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(Refused),
    ):
        await honcho_llm_call(
            model_config=_CONFIG,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            validate_response=validate,
        )

    assert inner.await_count == 3


_UNVALIDATED_CALLS: list[dict[str, Any]] = [
    {"stream": True},
    {
        "tools": [{"name": "noop", "parameters": {}}],
        "tool_executor": _noop_tool,
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_kwargs", _UNVALIDATED_CALLS, ids=["streaming", "tool-loop"]
)
async def test_rejected_where_it_would_never_run(call_kwargs: dict[str, Any]) -> None:
    """Streaming and tool-loop calls don't go through the hook, so refuse the
    combination rather than silently skipping validation."""
    with pytest.raises(ValidationException, match="validate_response"):
        await honcho_llm_call(
            model_config=_CONFIG,
            prompt="hi",
            max_tokens=10,
            validate_response=lambda _response: None,
            **call_kwargs,
        )
