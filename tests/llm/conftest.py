from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from src.config import ModelTransport
from src.llm.backend import (
    CompletionResult,
    ProviderBackend,
    StreamChunk,
    ToolCallResult,
)
from src.llm.messages import ModelOrigin
from src.llm.profiles import profile_for
from src.llm.types import HonchoLLMCallResponse


class FakeBackend(ProviderBackend):
    """Simple backend for request-builder and orchestration tests."""

    def __init__(self, responses: list[CompletionResult] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: Iterator[CompletionResult] = iter(
            responses or [CompletionResult(content="ok")]
        )

    async def complete(self, **kwargs: Any) -> CompletionResult:
        self.calls.append(kwargs)
        return next(self._responses)

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        self.calls.append(kwargs)
        result = next(self._responses)
        yield StreamChunk(content=result.content, is_done=True)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


def with_assistant_turn(
    response: HonchoLLMCallResponse[Any],
    transport: ModelTransport = "anthropic",
    model: str = "claude-sonnet-4-5",
    base_url: str | None = None,
) -> HonchoLLMCallResponse[Any]:
    """Attach the assistant turn the executor would parse from this response.

    For tests that stand in for `honcho_llm_call_inner`, whose real responses
    carry the turn the tool loop appends to its history.
    """
    result = CompletionResult(
        content=response.content,
        tool_calls=[ToolCallResult(**call) for call in response.tool_calls_made],
        thinking_content=response.thinking_content,
        thinking_blocks=response.thinking_blocks,
        reasoning_details=response.reasoning_details,
    )
    origin = ModelOrigin(transport=transport, model=model, base_url=base_url)
    response.message = profile_for(transport, model).codec.parse(result, origin)
    return response
