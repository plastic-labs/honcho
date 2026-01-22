"""SSE (Server-Sent Events) parsing utilities."""

from __future__ import annotations

import codecs
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Generator, Iterable
from typing import Any, cast

logger = logging.getLogger(__name__)


class SSEStreamParser:
    """
    Incrementally parse an SSE byte stream into content strings.

    This parser is designed for real network streaming where byte chunks may split UTF-8
    codepoints and/or split lines arbitrarily. It maintains:

    - An incremental UTF-8 decoder to safely decode across chunk boundaries.
    - A text buffer to safely assemble complete lines across chunk boundaries.

    The Honcho streaming format is expected to include lines in the form `data:<json>` or
    `data: <json>`. Each data line should contain a JSON object with:

    - `done: true` to indicate stream completion.
    - `delta.content` containing incremental text.

    Any JSON decoding failures are logged with the same warning format as the legacy
    parser, including a preview of the data payload.
    """

    def __init__(self) -> None:
        self._decoder: codecs.IncrementalDecoder = codecs.getincrementaldecoder(
            "utf-8"
        )(errors="replace")
        self._text_buffer: str = ""
        self._done: bool = False

    @property
    def done(self) -> bool:
        """Whether the stream has emitted a `done: true` message."""
        return self._done

    def feed(self, chunk: bytes) -> Generator[str, None, None]:
        """
        Feed the next bytes from the SSE stream and yield any newly available content.

        Args:
            chunk: Raw bytes from the SSE stream.

        Yields:
            Content strings extracted from any complete `data:` lines decoded from this
            chunk (and any previously buffered partial data).
        """
        if self._done or not chunk:
            return

        decoded = self._decoder.decode(chunk, final=False)
        if decoded:
            self._text_buffer += decoded

        yield from self._drain_complete_lines()

    def finalize(self) -> Generator[str, None, None]:
        """
        Finalize the stream and yield any remaining content.

        This should be called once the underlying byte stream is finished to flush any
        remaining decoder/buffer state.
        """
        if self._done:
            return

        decoded = self._decoder.decode(b"", final=True)
        if decoded:
            self._text_buffer += decoded

        yield from self._drain_complete_lines(flush_partial=True)

    def _drain_complete_lines(
        self, *, flush_partial: bool = False
    ) -> Generator[str, None, None]:
        while not self._done:
            line = self._pop_line(flush_partial=flush_partial)
            if line is None:
                return
            yield from self._handle_line(line)

    def _pop_line(self, *, flush_partial: bool) -> str | None:
        if not self._text_buffer:
            return None

        idx_n = self._text_buffer.find("\n")
        idx_r = self._text_buffer.find("\r")

        if idx_n == -1 and idx_r == -1:
            if not flush_partial:
                return None
            line = self._text_buffer
            self._text_buffer = ""
            return line

        if idx_n == -1:
            idx = idx_r
        elif idx_r == -1:
            idx = idx_n
        else:
            idx = min(idx_n, idx_r)

        sep = self._text_buffer[idx]
        if sep == "\n":
            line = self._text_buffer[:idx]
            self._text_buffer = self._text_buffer[idx + 1 :]
            if line.endswith("\r"):
                line = line[:-1]
            return line

        if idx == len(self._text_buffer) - 1 and not flush_partial:
            return None

        if idx + 1 < len(self._text_buffer) and self._text_buffer[idx + 1] == "\n":
            line = self._text_buffer[:idx]
            self._text_buffer = self._text_buffer[idx + 2 :]
            return line

        line = self._text_buffer[:idx]
        self._text_buffer = self._text_buffer[idx + 1 :]
        return line

    def _handle_line(self, line: str) -> Generator[str, None, None]:
        if not line.startswith("data:"):
            return

        json_str = line[len("data:") :].lstrip(" ")
        if not json_str:
            return

        try:
            parsed: object = json.loads(json_str)
            if not isinstance(parsed, dict):
                return

            chunk_data = cast(dict[str, Any], parsed)
            if chunk_data.get("done"):
                self._done = True
                return

            delta_obj = chunk_data.get("delta", {})
            if not isinstance(delta_obj, dict):
                return

            delta_data = cast(dict[str, Any], delta_obj)
            content = delta_data.get("content")
            if isinstance(content, str) and content:
                yield content
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to decode streaming chunk: %s (data: %s)",
                e,
                json_str[:100],
            )


def parse_sse_chunk(
    chunk: bytes, *, parser: SSEStreamParser | None = None
) -> Generator[str, None, None]:
    """
    Parse bytes from an SSE stream and yield content strings.

    For correct handling of UTF-8 and line boundaries across network chunks, construct
    one `SSEStreamParser` per stream and pass it for each call:

    - `yield from parse_sse_chunk(chunk, parser=parser)`

    Args:
        chunk: Raw bytes from the SSE stream.
        parser: Optional persistent parser instance. If omitted, a temporary parser is
            created and finalized for this single chunk.

    Yields:
        Content strings extracted from delta objects.
    """
    if parser is None:
        tmp = SSEStreamParser()
        yield from tmp.feed(chunk)
        yield from tmp.finalize()
        return
    yield from parser.feed(chunk)


def parse_sse_stream(chunks: Iterable[bytes]) -> Generator[str, None, None]:
    """
    Parse an SSE byte stream and yield content strings.

    Args:
        chunks: An iterable of raw byte chunks from an SSE stream.

    Yields:
        Content strings extracted from delta objects, in order.
    """
    parser = SSEStreamParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
        if parser.done:
            return
    yield from parser.finalize()


async def parse_sse_astream(chunks: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
    """
    Parse an async SSE byte stream and yield content strings.

    Args:
        chunks: An async iterable of raw byte chunks from an SSE stream.

    Yields:
        Content strings extracted from delta objects, in order.
    """
    parser = SSEStreamParser()
    async for chunk in chunks:
        for content in parser.feed(chunk):
            yield content
        if parser.done:
            return
    for content in parser.finalize():
        yield content
