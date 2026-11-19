"""Pagination wrapper for Honcho SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Generic, Literal, get_args

from pydantic import BaseModel
from typing_extensions import TypeVar

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", default=T)

__all__ = ["SyncPage", "AsyncPage", "TotalKind"]

TotalKind = Literal["exact", "estimate", "at_least", "at_most"]
_TOTAL_KINDS: tuple[TotalKind, ...] = get_args(TotalKind)

# The server's default PAGINATION_COUNT_LIMIT. A server that predates
# `total_kind` reports a total past it as the limit plus one or as a bound.
_DEFAULT_COUNT_LIMIT = 100_000


def _total_kind(data: dict[str, Any]) -> TotalKind:
    kind = data.get("total_kind")
    for known in _TOTAL_KINDS:
        if kind == known:
            return known
    total = data.get("total")
    if isinstance(total, int) and total > _DEFAULT_COUNT_LIMIT:
        return "at_least"
    return "exact"


def cursor_query(query: dict[str, Any], cursor: str) -> dict[str, Any]:
    """Turn an offset list request's query into the cursor-mode request that reads on from `cursor`."""
    return {**{k: v for k, v in query.items() if k != "page"}, "cursor": cursor}


def _cursor_after(data: dict[str, Any]) -> str | None:
    """The cursor that reads on after a cursor-mode page; raises if the server sent an offset page."""
    if "next_page" not in data:
        raise RuntimeError(
            "The server answered a cursor-mode list request with an offset page"
        )
    cursor = data["next_page"]
    return cursor if isinstance(cursor, str) else None


class SyncPage(Generic[T, U]):
    """
    Paginated result wrapper that transforms objects from type T to type U.

    Provides iteration and transformation capabilities for paginated API responses.
    """

    def __init__(
        self,
        data: dict[str, Any],
        item_type: type[T],
        transform_func: Callable[[T], U] | None = None,
        fetch_next: Callable[[int], "SyncPage[T, U]"] | None = None,
        fetch_cursor: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the page.

        Args:
            data: Raw paginated response data with items, page, size, total, pages
            item_type: Type to parse items as
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
            fetch_next: Optional callback to fetch the next page. Takes page number.
            fetch_cursor: Optional callback that repeats the list request in cursor
                          mode from the given cursor and returns the raw response.
                          Iteration uses it to read past this page.
        """
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[int], "SyncPage[T, U]"] | None = fetch_next
        self._fetch_cursor: Callable[[str], dict[str, Any]] | None = fetch_cursor

        # Parse items
        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    def __iter__(self) -> Iterator[U] | Iterator[T]:
        """
        Iterate over all transformed items across all pages.

        When the server sends `next_cursor`, the pages after this one are read in
        cursor mode: each request seeks past the last row read and runs no count,
        and rows added or removed during the walk do not make it skip or repeat
        rows. Servers that predate `next_cursor` are walked by page number.

        Warning:
            This iterator automatically fetches ALL subsequent pages as you iterate.
            For large datasets, this may result in many API calls. If you only need
            the current page, use the `items` property instead.
        """
        for item in self._walk():
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item

    def _walk(self) -> Iterator[T]:
        """Yield the untransformed items of this page and every page after it."""
        yield from self._raw_items
        if self._fetch_cursor is not None and "next_cursor" in self._data:
            cursor = self.next_cursor
            while cursor is not None:
                data = self._fetch_cursor(cursor)
                for raw in data.get("items", []):
                    yield self._item_type.model_validate(raw)
                cursor = _cursor_after(data)
            return
        page = self.get_next_page()
        while page is not None:
            yield from page._raw_items
            page = page.get_next_page()

    def __getitem__(self, index: int) -> U | T:
        """Get a transformed item by index on the current page."""
        item = self._raw_items[index]
        if self._transform_func is not None:
            return self._transform_func(item)
        return item

    def __len__(self) -> int:
        """Get the number of items on the current page."""
        return len(self._raw_items)

    @property
    def items(self) -> list[U] | list[T]:
        """Get all transformed items on the current page."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._raw_items]
        return list(self._raw_items)

    @property
    def total(self) -> int | None:
        """Get the total number of items across all pages."""
        return self._data.get("total")

    @property
    def page(self) -> int | None:
        """Get the current page number."""
        return self._data.get("page")

    @property
    def size(self) -> int | None:
        """Get the page size."""
        return self._data.get("size")

    @property
    def pages(self) -> int | None:
        """Get the total number of pages."""
        return self._data.get("pages")

    @property
    def total_kind(self) -> TotalKind:
        """
        What `total` is: "exact" (a count), "estimate" (a full count taken in the
        background, up to about 10 minutes old), "at_least" (a lower bound, such
        as the server's count limit plus one) or "at_most" (an upper bound, on an
        empty page after the first). Display anything but "exact" as approximate.
        For servers that predate this field, totals past 100,000 (the server's
        default count limit) are "at_least" and smaller ones "exact".
        """
        return _total_kind(self._data)

    @property
    def next_cursor(self) -> str | None:
        """The cursor that reads the rest of the list in cursor mode (None at the end, or from servers that predate cursor mode)."""
        cursor = self._data.get("next_cursor")
        return cursor if isinstance(cursor, str) else None

    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        current_page = self.page
        total_pages = self.pages
        if current_page is None or total_pages is None:
            return False
        return current_page < total_pages

    def get_next_page(self) -> "SyncPage[T, U] | None":
        """
        Fetch the next page of results.

        Returns None if there are no more pages or no fetch callback.
        """
        if not self.has_next_page():
            return None
        if self._fetch_next is None:
            return None

        current_page = self.page
        if current_page is None:
            return None

        return self._fetch_next(current_page + 1)


class AsyncPage(Generic[T, U]):
    """
    Async paginated result wrapper that transforms objects from type T to type U.

    Provides async iteration and transformation capabilities for paginated API responses.
    """

    def __init__(
        self,
        data: dict[str, Any],
        item_type: type[T],
        transform_func: Callable[[T], U] | None = None,
        fetch_next: Callable[[int], Awaitable["AsyncPage[T, U]"]] | None = None,
        fetch_cursor: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        """
        Initialize the async page.

        Args:
            data: Raw paginated response data with items, page, size, total, pages
            item_type: Type to parse items as
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
            fetch_next: Optional async callback to fetch the next page. Takes page number.
            fetch_cursor: Optional async callback that repeats the list request in
                          cursor mode from the given cursor and returns the raw
                          response. Iteration uses it to read past this page.
        """
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[int], Awaitable["AsyncPage[T, U]"]] | None = (
            fetch_next
        )
        self._fetch_cursor: Callable[[str], Awaitable[dict[str, Any]]] | None = (
            fetch_cursor
        )

        # Parse items
        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    async def __aiter__(self) -> AsyncIterator[U | T]:
        """
        Async iterate over all transformed items across all pages.

        When the server sends `next_cursor`, the pages after this one are read in
        cursor mode: each request seeks past the last row read and runs no count,
        and rows added or removed during the walk do not make it skip or repeat
        rows. Servers that predate `next_cursor` are walked by page number.

        Warning:
            This iterator automatically fetches ALL subsequent pages as you iterate.
            For large datasets, this may result in many API calls. If you only need
            the current page, use the `items` property instead.
        """
        async for item in self._walk():
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item

    async def _walk(self) -> AsyncIterator[T]:
        """Yield the untransformed items of this page and every page after it."""
        for item in self._raw_items:
            yield item
        if self._fetch_cursor is not None and "next_cursor" in self._data:
            cursor = self.next_cursor
            while cursor is not None:
                data = await self._fetch_cursor(cursor)
                for raw in data.get("items", []):
                    yield self._item_type.model_validate(raw)
                cursor = _cursor_after(data)
            return
        page = await self.get_next_page()
        while page is not None:
            for item in page._raw_items:
                yield item
            page = await page.get_next_page()

    def __getitem__(self, index: int) -> U | T:
        """Get a transformed item by index on the current page."""
        item = self._raw_items[index]
        if self._transform_func is not None:
            return self._transform_func(item)
        return item

    def __len__(self) -> int:
        """Get the number of items on the current page."""
        return len(self._raw_items)

    @property
    def items(self) -> list[U] | list[T]:
        """Get all transformed items on the current page."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._raw_items]
        return list(self._raw_items)

    @property
    def total(self) -> int | None:
        """Get the total number of items across all pages."""
        return self._data.get("total")

    @property
    def page(self) -> int | None:
        """Get the current page number."""
        return self._data.get("page")

    @property
    def size(self) -> int | None:
        """Get the page size."""
        return self._data.get("size")

    @property
    def pages(self) -> int | None:
        """Get the total number of pages."""
        return self._data.get("pages")

    @property
    def total_kind(self) -> TotalKind:
        """
        What `total` is: "exact" (a count), "estimate" (a full count taken in the
        background, up to about 10 minutes old), "at_least" (a lower bound, such
        as the server's count limit plus one) or "at_most" (an upper bound, on an
        empty page after the first). Display anything but "exact" as approximate.
        For servers that predate this field, totals past 100,000 (the server's
        default count limit) are "at_least" and smaller ones "exact".
        """
        return _total_kind(self._data)

    @property
    def next_cursor(self) -> str | None:
        """The cursor that reads the rest of the list in cursor mode (None at the end, or from servers that predate cursor mode)."""
        cursor = self._data.get("next_cursor")
        return cursor if isinstance(cursor, str) else None

    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        current_page = self.page
        total_pages = self.pages
        if current_page is None or total_pages is None:
            return False
        return current_page < total_pages

    async def get_next_page(self) -> "AsyncPage[T, U] | None":
        """
        Fetch the next page of results.

        Returns None if there are no more pages or no fetch callback.
        """
        if not self.has_next_page():
            return None
        if self._fetch_next is None:
            return None

        current_page = self.page
        if current_page is None:
            return None

        return await self._fetch_next(current_page + 1)
