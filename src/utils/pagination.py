"""
Offset-or-cursor pagination for list endpoints.

Plain offset pagination (`page`/`size`) runs a COUNT and an OFFSET scan on
every request, so a client walking a large collection page by page pays O(n)
per page. Two things here remove that cost:

- Cursor mode is opt-in: a request that sends a `cursor` query parameter (an
  empty value means "first page") gets a keyset-paginated `CursorPage` back,
  one seek per page and no count. Any other request gets the existing `Page`
  shape. Swapping the response in place would break installed SDKs, which stop
  iterating when `pages` is missing from the response. Servers that predate
  cursor mode ignore the unknown `cursor` parameter and answer with a `Page`,
  so clients can detect support from the response shape.
- Offset requests (`_paginate_offset`) run through the same keyset machinery,
  so each page knows whether a next one exists, and `total` is counted through
  a subquery that stops at `PAGINATION_COUNT_LIMIT` + 1 rows: exact up to the
  limit, the limit plus one beyond it, and raised past the current page while
  a next page exists so `pages` never ends a walk early.
- The offset shim makes offset requests cheap without changing their response,
  for clients that keep sending `?page=N`: after each page the position of its
  last row is cached, so the next page seeks instead of scanning. It needs the
  cache, and turns off with `CACHE_PAGINATION_OFFSET_SHIM`.
- With the cache enabled, a list past the limit reports a full count instead
  of the limit plus one. The full count runs in the background after the
  response, at most once per `CACHE_PAGINATION_ESTIMATE_REFRESH_SECONDS` per
  statement, so a large list's `total` keeps growing at that resolution.
- An offset page's `total` is a headline number, not a way to enumerate: it
  carries `total_kind` so a display can tell a count from an estimate or a
  bound, and
  `next_cursor` so a client can read the rest of the list in cursor mode,
  which stays exact under concurrent writes where offsets skip and repeat rows.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Coroutine
from typing import Any, ClassVar, Generic, Literal, TypeVar

from fastapi import HTTPException, Query, status
from fastapi_pagination import Page as _Page
from fastapi_pagination import Params
from fastapi_pagination.bases import AbstractParams, CursorRawParams
from fastapi_pagination.config import Config
from fastapi_pagination.cursor import CursorPage as _CursorPage
from fastapi_pagination.cursor import CursorParams as _CursorParams
from fastapi_pagination.cursor import decode_cursor
from fastapi_pagination.ext.sqlalchemy import apaginate
from fastapi_pagination.types import GreaterEqualZero
from pydantic import Field
from sqlakeyset import InvalidPage, unserialize_bookmark
from sqlalchemy import Select, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DataError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from src.cache.client import cache, cache_key_namespace, safe_cache_get
from src.config import settings
from src.dependencies import tracked_db
from src.telemetry.prometheus.metrics import PaginationShimOutcomes, prometheus_metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ai: the shim's cache traffic is an optimization on the request path, so a slow cache costs at most this per call, then counts as a miss
_CACHE_TIMEOUT_SECONDS = 0.25
# ai: the full count of a list past the limit runs off the request path; one that takes longer than this is abandoned and the list keeps its last count
_ESTIMATE_COUNT_TIMEOUT_SECONDS = 30
# ai: longer than the whole refresh (checkout, count, store), which _ESTIMATE_COUNT_TIMEOUT_SECONDS bounds, so the lock never lapses under a running count
_ESTIMATE_LOCK_SECONDS = 2 * _ESTIMATE_COUNT_TIMEOUT_SECONDS
# ai: far longer than the refresh interval, so a list that stays large keeps reporting its last count while the next one runs
_ESTIMATE_TTL_SECONDS = 86_400
# Strong references to running background counts, so they are not collected mid-run.
_estimate_tasks: set[asyncio.Task[None]] = set()


TotalKind = Literal["exact", "estimate", "at_least", "at_most"]


class Page(_Page[T], Generic[T]):
    """Offset page whose `total` is counted up to `PAGINATION_COUNT_LIMIT`."""

    total: GreaterEqualZero = Field(
        description=(
            "Number of matching rows, exact up to the server's "
            "PAGINATION_COUNT_LIMIT (default 100000). Past the limit it is the "
            "limit plus one, or, with the server's cache enabled, the list's last "
            "full count, which may be up to "
            "CACHE_PAGINATION_ESTIMATE_REFRESH_SECONDS (default 600) old. While a "
            "next page exists it is at least one more than this page's last row, "
            "so `pages` always reaches the end of the list. On the page that ends "
            "the list it is the number of rows up to that page's last row; an "
            "empty page after the first reports at most the rows before it, so "
            "`pages` ends a walk there. `total_kind` says which of these applies."
        )
    )
    total_kind: TotalKind = Field(
        default="exact",
        description=(
            'How to read `total`. "exact": a count of the matching rows (pages '
            "after the first may reuse one taken up to "
            'CACHE_PAGINATION_COUNT_TTL_SECONDS, default 60, earlier). "estimate": '
            "a full count taken in the background, up to "
            'CACHE_PAGINATION_ESTIMATE_REFRESH_SECONDS (default 600) old. "at_least": '
            "a lower bound, either the limit plus one or one more than this "
            'page\'s last row (display as "100,000+"). "at_most": an upper bound, '
            "reported by an empty page after the first."
        ),
    )
    next_cursor: str | None = Field(
        default=None,
        description=(
            "Send as `cursor` to read the rest of the list in cursor mode, which "
            "seeks past this page's last row and runs no count. Null on the page "
            "that ends the list."
        ),
    )


class CursorParams(_CursorParams):
    """Cursor params that skip the COUNT query.

    fastapi-pagination's cursor flow still counts the whole result set by
    default, which would keep half the cost cursor mode exists to remove.
    """

    def to_raw_params(self) -> CursorRawParams:
        raw = super().to_raw_params()
        raw.include_total = False
        return raw


class CursorPage(_CursorPage[T], Generic[T]):
    """Cursor page whose `total` is always null (see `CursorParams`)."""

    # ai: widened from the base's required int, as fastapi-pagination's own UseIncludeTotal(False) customizer does at runtime
    total: int | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    __params_type__: ClassVar[type[AbstractParams]] = CursorParams


def _invalid_cursor() -> HTTPException:
    # ai: HTTPException, not a HonchoException: the HonchoException handler logs a traceback at ERROR, and this is a client mistake
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor"
    )


def pagination_params(
    # ai: bounded so (page - 1) * size stays inside Postgres' bigint OFFSET
    page: int = Query(1, ge=1, le=2**31 - 1, description="Page number (offset mode)"),
    size: int = Query(50, ge=1, le=100, description="Page size"),
    cursor: str | None = Query(
        None,
        description=(
            "Opt in to cursor pagination. Send an empty value for the first "
            "page, then the `next_page` value from each response. When set, "
            "`page` is ignored and the response is a cursor page with no `total`."
        ),
    ),
) -> AbstractParams:
    """Resolve the request's pagination mode from its query parameters."""
    if cursor is None:
        return Params(page=page, size=size)

    # ai: validated here, before route code runs: some routes map a ValueError raised while paginating to a 404
    if cursor:
        try:
            unserialize_bookmark(decode_cursor(cursor) or "")
        except Exception:
            raise _invalid_cursor() from None
    return CursorParams(cursor=cursor, size=size)


async def _apaginate_cursor(
    db: AsyncSession, stmt: Select[Any], params: CursorParams, **apaginate_kwargs: Any
) -> CursorPage[Any]:
    # ai: the routes' response model is a union, so fastapi-pagination can't infer the page class from it
    return await apaginate(
        db, stmt, params=params, config=Config(page_cls=CursorPage), **apaginate_kwargs
    )


def _statement_digest(stmt: Select[Any]) -> str:
    """Identify a statement by its SQL and bound values.

    The digest covers everything that selects or orders rows: the workspace,
    every filter, the auth scope and the direction. Two requests share stored
    positions only if they would run the same query.
    """
    compiled = stmt.compile(dialect=postgresql.dialect())
    material = repr((compiled.string, sorted(compiled.params.items())))
    return hashlib.sha256(material.encode()).hexdigest()


async def _cache_set_once(key: str, value: Any, expire: int) -> None:
    # ai: one attempt under the shim's budget rather than safe_cache_set's retries: the request is waiting, and a lost write only costs a later miss
    try:
        async with asyncio.timeout(_CACHE_TIMEOUT_SECONDS):
            await cache.set(key, value, expire=expire)
    except Exception:
        logger.warning("Pagination cache set failed for key %s", key, exc_info=True)


def _capped_count_query(stmt: Select[Any]) -> Select[Any]:
    """Count `stmt`'s rows, stopping one past `PAGINATION_COUNT_LIMIT`.

    A plain COUNT walks every row the statement selects, so a list call on a
    large workspace pays for the whole workspace on every page. Counting
    through a LIMITed subquery lets Postgres stop once the limit is passed, and
    a result of limit + 1 tells the client there are more.
    """
    bounded = (
        stmt.order_by(None)
        .options(noload("*"))
        .limit(settings.PAGINATION_COUNT_LIMIT + 1)
    )
    return select(func.count()).select_from(bounded.subquery())


async def _read_estimate(key: str) -> tuple[int, float]:
    """Return the stored full count under `key` and when it was taken (0, 0.0 if none)."""
    stored = await safe_cache_get(key, timeout=_CACHE_TIMEOUT_SECONDS)
    if not isinstance(stored, dict):
        return 0, 0.0
    count = int(stored.get("count", 0))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    taken_at = float(stored.get("at", 0.0))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    return count, taken_at


def _is_fresh(taken_at: float) -> bool:
    return time.time() - taken_at < settings.CACHE.PAGINATION_ESTIMATE_REFRESH_SECONDS


async def _refresh_estimate(stmt: Select[Any], key: str) -> None:
    """Take a full count of `stmt` and store it under `key`.

    A cache lock, held under a token of this task's own, lets one process per
    statement count at a time; a holder that finds a fresh count already
    stored skips its own. The lock outlives the whole refresh, connection
    checkout included, so it cannot expire under a running count. A count that
    succeeds releases the lock; one that fails leaves it to expire, so a count
    that keeps timing out runs at most once per lock period.
    """
    lock_key, token = f"{key}:lock", uuid.uuid4().hex
    try:
        async with asyncio.timeout(_CACHE_TIMEOUT_SECONDS):
            if not await cache.set_lock(lock_key, token, expire=_ESTIMATE_LOCK_SECONDS):
                return
    except Exception:
        # ai: the cache is down or slow; requests carry on without an estimate, so this is not worth a warning per request
        logger.debug("Could not take the estimate lock for %s", key, exc_info=True)
        return
    try:
        _, taken_at = await _read_estimate(key)
        if not _is_fresh(taken_at):
            full = stmt.order_by(None).options(noload("*"))
            async with (
                asyncio.timeout(_ESTIMATE_COUNT_TIMEOUT_SECONDS),
                tracked_db("pagination.estimate") as db,
            ):
                await db.execute(
                    text(
                        "SET LOCAL statement_timeout = "
                        + str(_ESTIMATE_COUNT_TIMEOUT_SECONDS * 1000)
                    )
                )
                count = await db.scalar(
                    select(func.count()).select_from(full.subquery())
                )
            await _cache_set_once(
                key,
                {"count": int(count or 0), "at": time.time()},
                _ESTIMATE_TTL_SECONDS,
            )
    except Exception:
        logger.warning("Background count of a large list failed", exc_info=True)
        return
    try:
        async with asyncio.timeout(_CACHE_TIMEOUT_SECONDS):
            await cache.unlock(lock_key, token)
    except Exception:
        logger.debug("Could not release the estimate lock for %s", key, exc_info=True)


async def _fresh_estimate_past_limit(prefix: str) -> int | None:
    """Return the stored full count if it is fresh and past the limit, else None.

    Such a list's capped count would only return the limit plus one, so a
    request can report the stored count without running it.
    """
    count, taken_at = await _read_estimate(f"{prefix}:estimate")
    if count > settings.PAGINATION_COUNT_LIMIT and _is_fresh(taken_at):
        return count
    return None


async def _estimated_total(stmt: Select[Any], prefix: str) -> int:
    """Return the last full count of a list past the limit (0 if none yet).

    Starts a background recount when the stored one is older than
    `CACHE_PAGINATION_ESTIMATE_REFRESH_SECONDS`. The request never waits for it.
    """
    key = f"{prefix}:estimate"
    count, taken_at = await _read_estimate(key)
    if not _is_fresh(taken_at):
        task = asyncio.create_task(_refresh_estimate(stmt, key))
        _estimate_tasks.add(task)
        task.add_done_callback(_estimate_tasks.discard)
    return count


async def _paginate_offset(
    db: AsyncSession,
    stmt: Select[Any],
    params: Params,
    *,
    use_cache: bool,
    estimate: bool = False,
    **apaginate_kwargs: Any,
) -> Page[Any]:
    """
    Serve an offset request through a keyset seek wherever possible.

    Every page runs through the cursor machinery, so it knows whether a next
    page exists. A page that ends the list (a non-empty page with no next
    page, or an empty page 1) reports the rows up to its end as `total`
    without counting, so page 1 of a short list runs no count, and a cached
    count that is too high cannot send a walk past the last page. Otherwise
    `total` comes from `_capped_count_query`, and while a next page exists it
    is kept large enough that `pages` reaches past this page, so neither the
    cap nor a stale cached count can end a walk early.

    With `use_cache`, after serving page N the position of its last row is
    stored under page N+1, so a sequential walk seeks instead of scanning past
    every earlier row. Requests with no stored position (page 1, random
    access, expired entries, the page after a last page) run the OFFSET query,
    but still record a position for the page after them. On pages after the
    first, `total` comes from the cache and is recounted when it expires, so a
    long walk counts about once per TTL instead of once per page. Page 1
    never reads the cached count, so a list read right after a write is exact.

    With `estimate`, a list whose count reaches the limit reports the larger
    of the limit plus one and its last full count (see `_estimated_total`),
    and while that full count is fresh the capped count is skipped.

    An empty page after the first reports at most the rows before it, so a
    walk whose tail was deleted stops there instead of following a stale
    total through empty pages. `total_kind` says whether `total` is a count, a
    full count from the background, or one of these bounds (the limit plus one
    is a lower bound). Every page carries `next_cursor`, the cursor-mode position
    after its last row, so a client can read the rest of the list without
    offsets or counts.

    Positions are shared by every caller that runs the same query. A client
    that jumps ahead instead of walking can get a page positioned by another
    client's earlier walk, so rows written in between may be skipped or
    repeated, for up to the position TTL. A sequential walker re-seeds every
    position it uses.
    """
    prefix = count_key = position_key = ""
    if use_cache or estimate:
        try:
            prefix = f"{cache_key_namespace()}:v1:pagination:{_statement_digest(stmt)}"
            count_key = f"{prefix}:count"
            position_key = f"{prefix}:size={params.size}:page="
        except Exception:
            logger.warning("Pagination shim could not key a statement", exc_info=True)
            use_cache = estimate = False

    position, cached_total = None, None
    outcome = PaginationShimOutcomes.FIRST
    if use_cache and params.page > 1:
        position, cached_total = await asyncio.gather(
            safe_cache_get(
                f"{position_key}{params.page}", timeout=_CACHE_TIMEOUT_SECONDS
            ),
            safe_cache_get(count_key, timeout=_CACHE_TIMEOUT_SECONDS),
        )
        outcome = (
            PaginationShimOutcomes.HIT if position else PaginationShimOutcomes.MISS
        )

    stmt_for_page = stmt if position else stmt.offset((params.page - 1) * params.size)
    cursor_page = await _apaginate_cursor(
        db,
        stmt_for_page,
        CursorParams(cursor=position, size=params.size),
        **apaginate_kwargs,
    )
    if use_cache:
        prometheus_metrics.record_pagination_offset_shim(outcome)

    writes: list[Coroutine[Any, Any, None]] = []
    if use_cache and cursor_page.next_page:
        # ai: nothing is stored after a last page: a position taken there would skip rows appended later that OFFSET puts on the next page
        writes.append(
            _cache_set_once(
                f"{position_key}{params.page + 1}",
                cursor_page.next_page,
                settings.CACHE.PAGINATION_POSITION_TTL_SECONDS,
            )
        )
    items = list(cursor_page.items)
    rows_before = (params.page - 1) * params.size
    seen = rows_before + len(items) if items else 0
    # ai: an empty page past page 1 doesn't show where the list ends, so it keeps the count (clamped below)
    ends_list = not cursor_page.next_page and (bool(items) or params.page == 1)
    total: int | None = seen if ends_list else cached_total
    estimated = False
    if total is None and estimate:
        fresh = await _fresh_estimate_past_limit(prefix)
        if fresh is not None:
            total, estimated = fresh, True
    counted = total is None
    if total is None:
        total = await db.scalar(_capped_count_query(stmt))
    total = total or 0
    if use_cache and not estimated and (counted or params.page == 1):
        writes.append(
            _cache_set_once(
                count_key, total, settings.CACHE.PAGINATION_COUNT_TTL_SECONDS
            )
        )
    await asyncio.gather(*writes)

    kind: TotalKind = "exact"
    if estimated:
        kind = "estimate"
    elif not ends_list and total > settings.PAGINATION_COUNT_LIMIT:
        kind = "at_least"
        if estimate:
            full = await _estimated_total(stmt, prefix)
            if full > total:
                total, kind = full, "estimate"
    floor = seen + 1 if cursor_page.next_page else seen
    if total < floor:
        total, kind = floor, "at_least"
    if not items and params.page > 1 and total > rows_before:
        # ai: no row sits at or past this page's start, so a higher total (stale, estimated, or counted after rows moved) would walk clients through empty pages
        total, kind = rows_before, "at_most"
    return Page[Any].create(
        items,
        params=params,
        total=total,
        total_kind=kind,
        next_cursor=cursor_page.next_page,
    )


async def paginate_offset_or_cursor(
    db: AsyncSession,
    stmt: Select[Any],
    params: AbstractParams,
    **apaginate_kwargs: Any,
) -> Any:
    """
    Paginate `stmt` in whichever mode `params` selects.

    `stmt` must have a deterministic ORDER BY whose last column is unique (e.g.
    `created_at, id`); otherwise rows sharing a sort key can be skipped or
    repeated across pages, in either mode.
    """
    if isinstance(params, CursorParams):
        try:
            return await _apaginate_cursor(db, stmt, params, **apaginate_kwargs)
        except (InvalidPage, DataError, ProgrammingError):
            if not params.cursor:
                raise
            # ai: a well-formed cursor whose values don't fit this query (another endpoint's cursor, or a hand-edited one) fails in sqlakeyset or in Postgres
            logger.info("Rejected a cursor that doesn't fit its query", exc_info=True)
            raise _invalid_cursor() from None

    if not isinstance(params, Params):
        raise TypeError(f"Unsupported pagination params: {type(params).__name__}")
    return await _paginate_offset(
        db,
        stmt,
        params,
        use_cache=settings.CACHE.ENABLED and settings.CACHE.PAGINATION_OFFSET_SHIM,
        estimate=settings.CACHE.ENABLED,
        **apaginate_kwargs,
    )


__all__ = [
    "CursorPage",
    "CursorParams",
    "Page",
    "pagination_params",
    "paginate_offset_or_cursor",
]
