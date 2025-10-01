"""Shared types for the Honcho SDK."""

from __future__ import annotations

from collections.abc import Iterator, AsyncIterator


class DialecticStreamResponse:
    """
    Iterator for streaming dialectic responses with utilities for accessing the final response.

    Similar to OpenAI and Anthropic streaming patterns, this allows you to:
    - Iterate over chunks as they arrive
    - Access the final accumulated response after streaming completes

    Works with both sync and async iterators.

    Example (sync):
        ```python
        stream = peer.chat("Hello", stream=True)

        # Stream chunks
        for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```

    Example (async):
        ```python
        stream = await peer.chat("Hello", stream=True)

        # Stream chunks
        async for chunk in stream:
            print(chunk, end="", flush=True)

        # Get final response object
        final = stream.get_final_response()
        print(f"\\nFull content: {final['content']}")
        ```
    """

    _iterator: Iterator[str] | AsyncIterator[str]
    _accumulated_content: list[str]
    _is_complete: bool

    def __init__(self, iterator: Iterator[str] | AsyncIterator[str]):
        self._iterator = iterator
        self._accumulated_content = []
        self._is_complete = False

    # Sync iterator protocol
    def __iter__(self):
        if isinstance(self._iterator, Iterator):
            return self
        else:
            raise TypeError("iterator must be an sync iterator, got async iterator")

    def __next__(self) -> str:
        try:
            if not isinstance(self._iterator, Iterator):
                raise TypeError("iterator must be an sync iterator, got async iterator")
            chunk = next(self._iterator)  # type: ignore
            self._accumulated_content.append(chunk)
            return chunk
        except StopIteration:
            self._is_complete = True
            raise

    # Async iterator protocol
    def __aiter__(self):
        if isinstance(self._iterator, AsyncIterator):
            return self
        else:
            raise TypeError("iterator must be an async iterator, got sync iterator")

    async def __anext__(self) -> str:
        try:
            if not isinstance(self._iterator, AsyncIterator):
                raise TypeError("iterator must be an async iterator, got sync iterator")
            chunk = await self._iterator.__anext__()  # type: ignore
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
