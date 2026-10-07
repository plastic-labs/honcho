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
- The offset shim (`_paginate_offset_via_keyset`) makes offset requests cheap
  without changing their response, for clients that keep sending `?page=N`.
  It needs the cache, and turns off with `CACHE_PAGINATION_OFFSET_SHIM`.
"""

import asyncio
import hashlib
import logging
from collections.abc import Coroutine
from typing import Any, ClassVar, Generic, TypeVar, cast

from fastapi import HTTPException, Query, status
from fastapi_pagination import Page, Params
from fastapi_pagination.bases import AbstractParams, CursorRawParams
from fastapi_pagination.config import Config
from fastapi_pagination.cursor import CursorPage as _CursorPage
from fastapi_pagination.cursor import CursorParams as _CursorParams
from fastapi_pagination.cursor import decode_cursor
from fastapi_pagination.ext.sqlalchemy import apaginate, create_count_query
from sqlakeyset import InvalidPage, unserialize_bookmark
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DataError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.client import cache, cache_key_namespace, safe_cache_get
from src.config import settings
from src.telemetry.prometheus.metrics import PaginationShimOutcomes, prometheus_metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ai: the shim's cache traffic is an optimization on the request path, so a slow cache costs at most this per call, then counts as a miss
_CACHE_TIMEOUT_SECONDS = 0.25


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
    page: int = Query(1, ge=1, description="Page number (offset mode)"),
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


async def _paginate_offset_via_keyset(
    db: AsyncSession,
    stmt: Select[Any],
    params: Params,
    **apaginate_kwargs: Any,
) -> Page[Any]:
    """
    Serve an offset request with a keyset seek wherever possible.

    After serving page N, the position of its last row is stored under page
    N+1, so a sequential walk seeks instead of scanning past every earlier row.
    Requests with no stored position (page 1, random access, expired entries,
    the page after a last page) run the OFFSET query, but through the same
    keyset machinery, so they still record a position for the page after them.
    The response is the unchanged offset `Page`.

    On pages after the first, `total` comes from the cache and is recounted
    when it expires, so a long walk counts about once per TTL instead of once
    per page. Page 1 always counts, so a list read right after a write is
    exact. While a next page exists, `total` is kept large enough that `pages`
    reaches past this page, so a cached count from before rows were appended
    can't end a walk early.

    Positions are shared by every caller that runs the same query. A client
    that jumps ahead instead of walking can get a page positioned by another
    client's earlier walk, so rows written in between may be skipped or
    repeated, for up to the position TTL. A sequential walker re-seeds every
    position it uses.
    """
    try:
        prefix = f"{cache_key_namespace()}:v1:pagination:{_statement_digest(stmt)}"
    except Exception:
        logger.warning("Pagination shim could not key a statement", exc_info=True)
        return await apaginate(
            db, stmt, params=params, config=Config(page_cls=Page), **apaginate_kwargs
        )
    count_key = f"{prefix}:count"
    position_key = f"{prefix}:size={params.size}:page="

    if params.page == 1:
        outcome = PaginationShimOutcomes.FIRST
        position, cached_total = None, None
    else:
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
    prometheus_metrics.record_pagination_offset_shim(outcome)

    writes: list[Coroutine[Any, Any, None]] = []
    if cursor_page.next_page:
        # ai: nothing is stored after a last page: a position taken there would skip rows appended later that OFFSET puts on the next page
        writes.append(
            _cache_set_once(
                f"{position_key}{params.page + 1}",
                cursor_page.next_page,
                settings.CACHE.PAGINATION_POSITION_TTL_SECONDS,
            )
        )
    total = cached_total
    if total is None:
        total = await db.scalar(cast(Select[Any], create_count_query(stmt)))
        writes.append(
            _cache_set_once(
                count_key, total, settings.CACHE.PAGINATION_COUNT_TTL_SECONDS
            )
        )
    await asyncio.gather(*writes)

    if cursor_page.next_page:
        total = max(total or 0, params.page * params.size + 1)
    return Page.create(list(cursor_page.items), params=params, total=total)


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

    if (
        isinstance(params, Params)
        and settings.CACHE.ENABLED
        and settings.CACHE.PAGINATION_OFFSET_SHIM
    ):
        return await _paginate_offset_via_keyset(db, stmt, params, **apaginate_kwargs)
    return await apaginate(
        db, stmt, params=params, config=Config(page_cls=Page), **apaginate_kwargs
    )


__all__ = [
    "CursorPage",
    "CursorParams",
    "Page",
    "pagination_params",
    "paginate_offset_or_cursor",
]
