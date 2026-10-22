"""A singleton lease for loops that must run on one replica: a Postgres session advisory lock."""

import asyncio
from logging import getLogger

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from src.db import service_read_engine
from src.migrate import advisory_lock_key

logger = getLogger(__name__)

# Bounds each round trip on the lease connection, so a half-open connection
# can't wedge the holder's loop.
LEASE_CHECK_TIMEOUT_SECONDS = 10
# region ai
# Server-side TCP keepalives for the held connection only: if the holder's pod
# vanishes without closing the socket, the server notices within about a minute
# (idle + interval * count) and frees the lock, instead of after the OS default
# of two hours. Against a proxy they reach the proxy, not the client.
# endregion
_KEEPALIVE_SETTINGS = (
    "SET tcp_keepalives_idle = 30",
    "SET tcp_keepalives_interval = 10",
    "SET tcp_keepalives_count = 3",
)


class AdvisoryLease:
    """Holds a named session-level advisory lock on one dedicated connection."""

    # region ai
    # Postgres releases a session lock when its connection drops, so a pod that
    # dies gives the lease up with no expiry or heartbeat to tune. The lock lives
    # on a connection this object keeps checked out of the service engine's pool
    # for as long as it holds the lease (one pool connection, permanently):
    # returned to the pool with the lock still held, it would hand the lease to
    # the pool's next user. Releasing therefore invalidates the connection
    # instead of returning it. AUTOCOMMIT (service_read_engine) keeps the held
    # connection out of any transaction, so it never pins a snapshot (a
    # concurrent index build would wait on one).
    #
    # Needs a session-mode or direct connection: behind a transaction-mode
    # pooler the session lock would stay on a shared backend. Under MULTI_TENANT
    # validate_tenant_isolation refuses DB_POOLER_MODE=transaction/statement; a
    # flag-off DERIVER_SCHEDULER=api deployment must not put the service
    # connection behind one either.
    # endregion

    def __init__(self, name: str) -> None:
        self.name: str = name
        self._key: int = advisory_lock_key(name)
        self._connection: AsyncConnection | None = None

    @property
    def held(self) -> bool:
        return self._connection is not None

    async def ensure(self) -> bool:
        """Keep or try to take the lease; True when this replica holds it."""
        if self._connection is not None:
            try:
                async with asyncio.timeout(LEASE_CHECK_TIMEOUT_SECONDS):
                    await self._connection.execute(text("SELECT 1"))
                return True
            except Exception:
                logger.warning(
                    "Lost the connection holding lease %s; stepping down",
                    self.name,
                    exc_info=True,
                )
                await self._discard()
                return False

        async with asyncio.timeout(LEASE_CHECK_TIMEOUT_SECONDS):
            connection = await service_read_engine.connect()
        try:
            async with asyncio.timeout(LEASE_CHECK_TIMEOUT_SECONDS):
                acquired = bool(
                    await connection.scalar(
                        text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}
                    )
                )
                if acquired:
                    for statement in _KEEPALIVE_SETTINGS:
                        await connection.execute(text(statement))
        # ai: BaseException: a cancel mid-acquire must not return a locked connection to the pool
        except BaseException:
            await _invalidate(connection)
            raise
        if not acquired:
            await connection.close()
            return False
        self._connection = connection
        logger.info("Acquired lease %s", self.name)
        return True

    async def release(self) -> None:
        """Give the lease up, if held."""
        if self._connection is None:
            return
        try:
            async with asyncio.timeout(LEASE_CHECK_TIMEOUT_SECONDS):
                await self._connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"), {"key": self._key}
                )
        except Exception:
            logger.warning("Failed to unlock lease %s", self.name, exc_info=True)
        await self._discard()
        logger.info("Released lease %s", self.name)

    async def _discard(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            await _invalidate(connection)


async def _invalidate(connection: AsyncConnection) -> None:
    """Close the connection without returning it to the pool."""
    try:
        await connection.invalidate()
        await connection.close()
    except Exception:
        logger.debug("Closing the lease connection failed", exc_info=True)
