from __future__ import annotations

import asyncio

import pytest
import uvloop
from redis import exceptions as redis_exc
from redis.asyncio.connection import Connection
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

import src.cache.client  # noqa: F401  # pyright: ignore[reportUnusedImport]


async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(65536):
        for command in data.split(b"*")[1:]:
            writer.write(b"+PONG\r\n" if b"PING" in command.upper() else b"+OK\r\n")
        await writer.drain()
    writer.close()


async def _send_on_peer_closed_connection(health_check: bool) -> None:
    server = await asyncio.start_server(_serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    connection = Connection(
        host="127.0.0.1",
        port=port,
        socket_timeout=1,
        health_check_interval=30 if health_check else 0,
        retry=Retry(NoBackoff(), 0, supported_errors=(redis_exc.ConnectionError,)),
    )
    try:
        await connection.connect()
        assert connection._writer is not None  # pyright: ignore[reportPrivateUsage]
        connection._writer.transport.abort()  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0.05)
        connection.next_health_check = -1
        await connection.send_command("GET", "key")
    finally:
        await connection.disconnect()
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("health_check", [False, True])
def test_closed_transport_raises_retryable_connection_error(health_check: bool):
    with pytest.raises(redis_exc.ConnectionError) as excinfo:
        uvloop.run(_send_on_peer_closed_connection(health_check))
    assert type(excinfo.value) is redis_exc.ConnectionError


async def _ping_fresh_connection() -> None:
    server = await asyncio.start_server(_serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    connection = Connection(host="127.0.0.1", port=port, socket_timeout=1)
    try:
        await connection.send_command("PING")
        assert await connection.read_response() == b"PONG"
    finally:
        await connection.disconnect()
        server.close()
        await server.wait_closed()


def test_live_transport_still_sends():
    uvloop.run(_ping_fresh_connection())
