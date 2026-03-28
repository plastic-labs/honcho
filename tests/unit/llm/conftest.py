from collections.abc import AsyncIterator
from typing import Any

import pytest

from src.llm.backend import CompletionResult, ProviderBackend, StreamChunk


class FakeBackend(ProviderBackend):
    """Simple backend for request-builder and orchestration tests."""

    def __init__(self, responses: list[CompletionResult] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = iter(responses or [CompletionResult(content="ok")])

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
