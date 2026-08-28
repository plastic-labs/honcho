"""Disconnect handling for the non-streaming dialectic chat route (#1050).

Uvicorn reports a client disconnect on the ASGI receive channel but does not
cancel the running handler, so a non-streaming `/chat` request could keep its
(expensive) LLM call alive for minutes after the caller had already timed out.
These exercise the race helper directly — no DB, no LLM — so the cancellation
contract is pinned without standing up the full stack.
"""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from starlette.requests import Request

from src.routers.peers import _ClientDisconnected, _run_until_disconnect


def _request(receive: Callable[[], Awaitable[dict[str, object]]]) -> Request:
    """A minimal ASGI request whose receive channel is scripted per test."""
    return Request({"type": "http", "method": "POST", "headers": []}, receive)


async def test_disconnect_cancels_inflight_work():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_work() -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "should not get here"

    async def receive() -> dict[str, object]:
        # Let the work start, then report the client hanging up.
        await started.wait()
        return {"type": "http.disconnect"}

    with pytest.raises(_ClientDisconnected):
        await asyncio.wait_for(
            _run_until_disconnect(_request(receive), slow_work()), timeout=1
        )

    assert cancelled.is_set()


async def test_completed_work_returns_and_stops_watching():
    async def work() -> str:
        return "answer"

    # A receive that never yields a disconnect — the watcher must be torn down
    # once the work finishes rather than leaking a pending task.
    async def receive() -> dict[str, object]:
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    before = len(asyncio.all_tasks())
    result = await asyncio.wait_for(
        _run_until_disconnect(_request(receive), work()), timeout=1
    )

    assert result == "answer"
    # give the cancelled watcher a tick to unwind
    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) <= before


async def test_non_disconnect_receive_messages_are_ignored():
    # A stray http.request frame must not be mistaken for a disconnect.
    frames: list[dict[str, object]] = [
        {"type": "http.request", "body": b"", "more_body": False}
    ]
    messages = iter(frames)

    async def receive() -> dict[str, object]:
        try:
            return next(messages)
        except StopIteration:
            await asyncio.Event().wait()  # then block
            raise  # pragma: no cover

    async def work() -> str:
        await asyncio.sleep(0)
        return "done"

    result = await asyncio.wait_for(
        _run_until_disconnect(_request(receive), work()), timeout=1
    )
    assert result == "done"
