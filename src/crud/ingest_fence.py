"""The ingest fence: advisory locks that serialize ingest writes with deletion.

Every transaction that writes the ingest tables, and every deletion that removes
their rows, takes these locks before any row lock:

- the workspace fence, shared by ingest and session deletion, exclusive for
  workspace deletion;
- then a session fence per session it touches, in sorted order, exclusive.

Taking them first means ingest and deletion serialize per session and per
workspace without being able to deadlock on row locks, and a deletion's predicate
delete can't miss an ingest row that another transaction hasn't committed yet.
The ingest enqueuer try-locks instead of waiting, and skips a session it can't
take.

Every fence key is a single 64-bit key. Postgres keeps single-key advisory locks in
a separate space from the two-key form, which is what the message route's
per-session sequence lock uses (``_ingest_lock`` in ``src/crud/message.py``), so a
fence can never collide with it, and holding one never blocks the message route's
sequence lock.
All of them are transaction-scoped (``pg_*_xact_*``) and release at commit or
rollback.
"""

from collections.abc import Iterable
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.db import tenant_context

# The workspace and session fences hash different prefixes, so a workspace key
# and a session key never share an input string.
_WORKSPACE_PREFIX = "ingest-fence:workspace"
_SESSION_PREFIX = "ingest-fence:session"


def _workspace_key(workspace_name: str) -> str:
    return f"{_WORKSPACE_PREFIX}:{_tenant_part()}{workspace_name}"


def _session_key(workspace_name: str, session_name: str) -> str:
    # The workspace name's length goes first so a ":" inside either name can't
    # make two different (workspace, session) pairs read as the same string.
    return (
        f"{_SESSION_PREFIX}:{_tenant_part()}"
        + f"{len(workspace_name)}:{workspace_name}:{session_name}"
    )


def _tenant_part() -> str:
    """The tenant segment of a fence key: "<tenant>:" flag-on, "" flag-off.

    Raises:
        ValueError: MULTI_TENANT is on and no tenant is bound.
    """
    # region ai
    # Advisory locks are database-wide, not RLS-scoped, and workspace names repeat
    # across tenants ("default"), so flag-on the key carries the tenant. Flag-off
    # it doesn't, matching _ingest_lock. Flag-on with no tenant bound fails closed,
    # as every tenant-scoped write does.
    # endregion
    if not settings.MULTI_TENANT:
        return ""
    tenant = tenant_context.get()
    if not tenant:
        raise ValueError(
            "cannot take the ingest fence without a tenant when MULTI_TENANT is on"
        )
    return f"{tenant}:"


async def take_workspace_fence(
    db: AsyncSession,
    workspace_name: str,
    *,
    mode: Literal["shared", "exclusive"],
    try_lock: bool = False,
) -> bool:
    """Take the workspace fence in the given mode.

    Returns:
        True when the lock is held. With ``try_lock=False`` it waits and always
        returns True; with ``try_lock=True`` it returns False instead of waiting.
    """
    function = {
        ("shared", False): "pg_advisory_xact_lock_shared",
        ("shared", True): "pg_try_advisory_xact_lock_shared",
        ("exclusive", False): "pg_advisory_xact_lock",
        ("exclusive", True): "pg_try_advisory_xact_lock",
    }[(mode, try_lock)]
    result = await db.execute(
        text(f"SELECT {function}(hashtextextended(:key, 0))"),
        {"key": _workspace_key(workspace_name)},
    )
    acquired = result.scalar()
    # The waiting forms return void (None); only the try forms return a boolean.
    return True if not try_lock else bool(acquired)


async def take_session_fences(
    db: AsyncSession,
    workspace_name: str,
    session_names: Iterable[str],
    *,
    try_lock: bool = False,
) -> bool:
    """Take the exclusive session fence for each session, in sorted order.

    Call after ``take_workspace_fence`` in the same transaction.

    Returns:
        True when every lock is held. With ``try_lock=True`` it stops at the first
        lock it can't take and returns False; the caller ends its transaction,
        which releases whatever was taken.
    """
    function = "pg_try_advisory_xact_lock" if try_lock else "pg_advisory_xact_lock"
    for session_name in sorted(set(session_names)):
        result = await db.execute(
            text(f"SELECT {function}(hashtextextended(:key, 0))"),
            {"key": _session_key(workspace_name, session_name)},
        )
        if try_lock and not result.scalar():
            return False
    return True
