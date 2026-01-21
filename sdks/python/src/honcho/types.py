"""Shared types for the Honcho SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Self

__all__ = [
    "DialecticStreamResponse",
    "AsyncDialecticStreamResponse",
]


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

    def __init__(self, iterator: Iterator[str]) -> None:
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False

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

    def __init__(self, iterator: AsyncIterator[str]) -> None:
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False

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
    def is_complete(self) -> bool:
        """Check if the stream has finished."""
        return self._is_complete
