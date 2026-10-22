"""A singleton lease for loops that must run on one replica: a Postgres session advisory lock."""

from logging import getLogger

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from src.db import service_engine
from src.migrate import advisory_lock_key

logger = getLogger(__name__)


class AdvisoryLease:
    """Holds a named session-level advisory lock on one dedicated connection."""

    # region ai
    # Postgres releases a session lock when its connection drops, so a pod that
    # dies gives the lease up with no expiry or heartbeat to tune. The lock lives
    # on a connection this object keeps checked out of the service engine's pool
    # for as long as it holds the lease: returned to the pool with the lock still
    # held, it would hand the lease to the pool's next user. Releasing therefore
    # invalidates the connection instead of returning it. AUTOCOMMIT keeps the
    # held connection out of any transaction, so it never pins a snapshot (a
    # concurrent index build would wait on one). Needs a session-mode connection:
    # validate_tenant_isolation already refuses a transaction-mode pooler under
    # MULTI_TENANT, and the lease runs only under DERIVER.SCHEDULER == "api".
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

        connection = await service_engine.connect()
        try:
            await connection.execution_options(isolation_level="AUTOCOMMIT")
            acquired = bool(
                await connection.scalar(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}
                )
            )
        except Exception:
            await connection.invalidate()
            await connection.close()
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
            await self._connection.execute(
                text("SELECT pg_advisory_unlock(:key)"), {"key": self._key}
            )
        except Exception:
            logger.warning("Failed to unlock lease %s", self.name, exc_info=True)
        await self._discard()
        logger.info("Released lease %s", self.name)

    async def _discard(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            await connection.invalidate()
            await connection.close()
        except Exception:
            logger.debug("Closing the lease connection failed", exc_info=True)
