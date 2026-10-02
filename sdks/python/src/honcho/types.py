"""Shared types for the Honcho SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

from honcho.api_types import Evidence

__all__ = [
    "ChatResponse",
    "DialecticStreamResponse",
    "AsyncDialecticStreamResponse",
]

TContent = TypeVar("TContent")


@dataclass(frozen=True)
class ChatResponse(Generic[TContent]):
    """An answer together with what it was built from.

    Returned by `chat(..., include_evidence=True)`. Without that, `chat`
    returns the answer on its own.

    `evidence` lists what the dialectic read while answering, collated from its
    own reads rather than reported by the model. That makes it deterministic
    but broader than a citation list: a conclusion appears because the agent
    saw it, which is not proof the answer relied on it. It is empty rather than
    None when a run genuinely read nothing.
    """

    content: TContent | None
    evidence: Evidence | None = None


class DialecticStreamResponse:
    """
    Sync streaming response for dialectic queries.

    Allows iterating over chunks as they arrive and accessing the final
    accumulated response after streaming completes.

    Example:
        ```python
        stream = peer.chat_stream("Hello")

        # Stream chunks
        for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```
    """

    _iterator: Iterator[str]
    _accumulated_content: list[str]
    _is_complete: bool
    _evidence_source: Callable[[], Evidence | None] | None

    def __init__(
        self,
        iterator: Iterator[str],
        evidence_source: Callable[[], Evidence | None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False
        self._evidence_source = evidence_source

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._iterator)
            self._accumulated_content.append(chunk)
            return chunk
        except StopIteration:
            self._is_complete = True
            raise

    def get_final_response(self) -> dict[str, str]:
        """
        Get the final accumulated response after streaming completes.

        Returns:
            A dictionary with the full content: {"content": "full accumulated text"}

        Note:
            This should be called after the stream has been fully consumed.
            If called before completion, it returns the content accumulated so far.
        """
        return {"content": "".join(self._accumulated_content)}

    @property
    def evidence(self) -> Evidence | None:
        """What the answer was built from, once the stream has finished.

        The server can only know this after the answer is complete, so it
        arrives on the stream's final event. Reading it before the stream is
        fully consumed returns None, as does a request that did not ask for
        evidence.
        """
        if self._evidence_source is None:
            return None
        return self._evidence_source()

    @property
    def is_complete(self) -> bool:
        """Check if the stream has finished."""
        return self._is_complete


class AsyncDialecticStreamResponse:
    """
    Async streaming response for dialectic queries.

    Allows iterating over chunks as they arrive and accessing the final
    accumulated response after streaming completes.

    Example:
        ```python
        stream = await peer.aio.chat_stream("Hello")

        # Stream chunks
        async for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```
    """

    _iterator: AsyncIterator[str]
    _accumulated_content: list[str]
    _is_complete: bool
    _evidence_source: Callable[[], Evidence | None] | None

    def __init__(
        self,
        iterator: AsyncIterator[str],
        evidence_source: Callable[[], Evidence | None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False
        self._evidence_source = evidence_source

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str:
        try:
            chunk = await self._iterator.__anext__()
            self._accumulated_content.append(chunk)
            return chunk
        except StopAsyncIteration:
            self._is_complete = True
            raise

    def get_final_response(self) -> dict[str, str]:
        """
        Get the final accumulated response after streaming completes.

        Returns:
            A dictionary with the full content: {"content": "full accumulated text"}

        Note:
            This should be called after the stream has been fully consumed.
            If called before completion, it returns the content accumulated so far.
        """
        return {"content": "".join(self._accumulated_content)}

    @property
    def evidence(self) -> Evidence | None:
        """What the answer was built from, once the stream has finished.

        See :attr:`DialecticStreamResponse.evidence`.
        """
        if self._evidence_source is None:
            return None
        return self._evidence_source()

    @property
    def is_complete(self) -> bool:
        """Check if the stream has finished."""
        return self._is_complete
