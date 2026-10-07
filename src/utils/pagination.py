"""
Offset-or-cursor pagination for list endpoints.

Offset pagination (`page`/`size`) runs a COUNT and an OFFSET scan on every
request, so a client walking a large collection page by page pays O(n) per
page. Cursor (keyset) pagination seeks straight to the next row and skips the
count.

Cursor mode is opt-in: a request that sends a `cursor` query parameter (an
empty value means "first page") gets a `CursorPage` back; any other request
gets the existing `Page` shape. Swapping the response in place would break
installed SDKs, which stop iterating when `pages` is missing from the response.

Servers that predate cursor mode ignore the unknown `cursor` parameter and
answer with a `Page`, so clients can detect support from the response shape.

Offset requests are made cheap too, without changing their response, by the
offset shim (see `_paginate_offset_via_keyset`): installed SDKs keep sending
`?page=N` and will never switch to cursors on their own.
"""

import hashlib
import logging
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
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.client import cache, cache_key_namespace, safe_cache_set
from src.config import settings
from src.telemetry.prometheus.metrics import pagination_offset_shim_counter

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Stored in place of a position when the page before came back short, so a
# walker paging past the end gets an empty page without a query.
_END = "end"


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

    # Widened from the base's required int, as fastapi-pagination's own
    # UseIncludeTotal(False) customizer does at runtime.
    total: int | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    __params_type__: ClassVar[type[AbstractParams]] = CursorParams


def _invalid_cursor() -> HTTPException:
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

    # Validate the cursor before any route code runs: a malformed one is a
    # 400, and some routes map a ValueError raised while paginating to a 404.
    if cursor:
        try:
            unserialize_bookmark(decode_cursor(cursor) or "")
        except InvalidPage:
            raise _invalid_cursor() from None
    return CursorParams(cursor=cursor, size=size)


def _statement_digest(stmt: Select[Any]) -> str:
    """Identify a statement by its SQL and bound values.

    The digest covers everything that selects or orders rows: the workspace,
    every filter, the auth scope and the direction. Two requests share stored
    positions only if they would run the same query.
    """
    compiled = stmt.compile(dialect=postgresql.dialect())
    material = repr((compiled.string, sorted(compiled.params.items())))
    return hashlib.sha256(material.encode()).hexdigest()


async def _cache_get(key: str) -> Any:
    """Read the cache, treating any failure as a miss."""
    try:
        return await cache.get(key)
    except Exception:
        logger.warning("Pagination cache read failed for key %s", key, exc_info=True)
        return None


async def _paginate_offset_via_keyset(
    db: AsyncSession,
    stmt: Select[Any],
    params: Params,
    **apaginate_kwargs: Any,
) -> Page[Any]:
    """
    Serve an offset request with a keyset seek wherever possible.

    Installed SDKs walk pages in order. After serving page N, the position of
    its last row is stored under page N+1, so a sequential walk seeks from
    there instead of scanning past every earlier row. Requests with no stored
    position (page 1, random access, expired entries) run the OFFSET query,
    but through the same keyset machinery, so they still record a position
    for the page after them. The response is the unchanged offset `Page`.

    On pages after the first, `total` comes from the cache and is recounted
    when it expires, so a long walk counts about once per TTL instead of once
    per page. Page 1 always counts, so a list read right after a write is exact.
    """
    prefix = f"{cache_key_namespace()}:pagination:{_statement_digest(stmt)}"
    count_key = f"{prefix}:count"
    position_key = f"{prefix}:size={params.size}:page="

    position = (
        None if params.page == 1 else await _cache_get(f"{position_key}{params.page}")
    )

    items: list[Any] = []
    if position == _END:
        outcome = "end"
    else:
        outcome = "hit" if position else "miss"
        if not position:
            stmt_for_page = stmt.offset((params.page - 1) * params.size)
        else:
            stmt_for_page = stmt
        cursor_page = await apaginate(
            db,
            stmt_for_page,
            params=CursorParams(cursor=position, size=params.size),
            config=Config(page_cls=CursorPage),
            **apaginate_kwargs,
        )
        items = list(cursor_page.items)
        await safe_cache_set(
            f"{position_key}{params.page + 1}",
            cursor_page.next_page or _END,
            expire=settings.PAGINATION_SHIM_POSITION_TTL_SECONDS,
        )
    pagination_offset_shim_counter.labels(outcome=outcome).inc()

    total = None if params.page == 1 else await _cache_get(count_key)
    if total is None:
        total = await db.scalar(cast(Select[Any], create_count_query(stmt)))
        await safe_cache_set(
            count_key, total, expire=settings.PAGINATION_SHIM_COUNT_TTL_SECONDS
        )

    return Page.create(items, params=params, total=total)


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
    try:
        if isinstance(params, CursorParams):
            # The route's response model is a union, so fastapi-pagination
            # can't infer the page class from it; name it explicitly.
            return await apaginate(
                db,
                stmt,
                params=params,
                config=Config(page_cls=CursorPage),
                **apaginate_kwargs,
            )
        if (
            isinstance(params, Params)
            and settings.PAGINATION_OFFSET_SHIM
            and settings.CACHE.ENABLED
        ):
            return await _paginate_offset_via_keyset(
                db, stmt, params, **apaginate_kwargs
            )
        return await apaginate(
            db, stmt, params=params, config=Config(page_cls=Page), **apaginate_kwargs
        )
    except InvalidPage:
        # A well-formed cursor that doesn't fit this query, e.g. one taken
        # from a different list endpoint.
        raise _invalid_cursor() from None


__all__ = [
    "CursorPage",
    "CursorParams",
    "Page",
    "pagination_params",
    "paginate_offset_or_cursor",
]
