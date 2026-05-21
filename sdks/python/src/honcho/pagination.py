"""Pagination wrappers for Honcho SDK.

Two flavors live here:
    - ``SyncPage`` / ``AsyncPage`` — offset/page-number pagination
    - ``SyncCursorPage`` / ``AsyncCursorPage`` — opaque-cursor pagination

Use whichever matches the endpoint's response envelope.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", default=T)

__all__ = ["SyncPage", "AsyncPage", "SyncCursorPage", "AsyncCursorPage"]


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
    ) -> None:
        """
        Initialize the page.

        Args:
            data: Raw paginated response data with items, page, size, total, pages
            item_type: Type to parse items as
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
            fetch_next: Optional callback to fetch the next page. Takes page number.
        """
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[int], "SyncPage[T, U]"] | None = fetch_next

        # Parse items
        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    def __iter__(self) -> Iterator[U] | Iterator[T]:
        """
        Iterate over all transformed items across all pages.

        Warning:
            This iterator automatically fetches ALL subsequent pages as you iterate.
            For large datasets, this may result in many API calls. If you only need
            the current page, use the `items` property instead.
        """
        page: SyncPage[T, U] | None = self
        while page is not None:
            for item in page._raw_items:
                if self._transform_func is not None:
                    yield self._transform_func(item)
                else:
                    yield item
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
    ) -> None:
        """
        Initialize the async page.

        Args:
            data: Raw paginated response data with items, page, size, total, pages
            item_type: Type to parse items as
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
            fetch_next: Optional async callback to fetch the next page. Takes page number.
        """
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[int], Awaitable["AsyncPage[T, U]"]] | None = (
            fetch_next
        )

        # Parse items
        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    async def __aiter__(self) -> AsyncIterator[U] | AsyncIterator[T]:
        """
        Async iterate over all transformed items across all pages.

        Warning:
            This iterator automatically fetches ALL subsequent pages as you iterate.
            For large datasets, this may result in many API calls. If you only need
            the current page, use the `items` property instead.
        """
        page: AsyncPage[T, U] | None = self
        while page is not None:
            for item in page._raw_items:
                if self._transform_func is not None:
                    yield self._transform_func(item)
                else:
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


class SyncCursorPage(Generic[T, U]):
    """
    Cursor-paginated result wrapper that transforms objects from type T to type U.

    Cursor pagination uses opaque tokens (``next_page`` / ``previous_page``) instead
    of page numbers; the navigation is stable across concurrent mutations of the
    underlying data, unlike offset pagination.

    Note:
        The underlying server data may still mutate between page fetches (rows
        appearing or being processed away). Iterating across pages snapshots
        what the server returns per page, but pages may be inconsistent with
        each other under concurrent processing. Use ``.items`` to read just
        the current page if you need a stable view.
    """

    def __init__(
        self,
        data: dict[str, Any],
        item_type: type[T],
        transform_func: Callable[[T], U] | None = None,
        fetch_next: Callable[[str], "SyncCursorPage[T, U]"] | None = None,
        fetch_previous: Callable[[str], "SyncCursorPage[T, U]"] | None = None,
    ) -> None:
        """
        Initialize a cursor page.

        Args:
            data: Raw paginated response — ``items``, ``total``, ``current_page``,
                ``current_page_backwards``, ``next_page``, ``previous_page``.
            item_type: Pydantic type to parse each item as.
            transform_func: Optional mapper from raw item type to a user-facing type.
            fetch_next: Optional callback that takes a cursor token and returns
                the page at that cursor.
            fetch_previous: Optional callback for backwards navigation.
        """
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[str], "SyncCursorPage[T, U]"] | None = fetch_next
        self._fetch_previous: Callable[[str], "SyncCursorPage[T, U]"] | None = (
            fetch_previous
        )

        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    def __iter__(self) -> Iterator[U] | Iterator[T]:
        """
        Iterate over all transformed items across all pages.

        Warning:
            Automatically chases ``next_page`` tokens until exhausted. For
            a mutating queue this may yield a different set of items than
            a single-shot snapshot. Use ``items`` for the current page only.
        """
        page: SyncCursorPage[T, U] | None = self
        while page is not None:
            for item in page._raw_items:
                if self._transform_func is not None:
                    yield self._transform_func(item)
                else:
                    yield item
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
        """Get all transformed items on the current page (snapshot only)."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._raw_items]
        return list(self._raw_items)

    @property
    def total(self) -> int | None:
        """Total items across all pages, when the server populates it."""
        return self._data.get("total")

    @property
    def current_page(self) -> str | None:
        """Cursor token that re-fetches the current page."""
        return self._data.get("current_page")

    @property
    def current_page_backwards(self) -> str | None:
        """Cursor token to re-fetch the current page starting from the last item."""
        return self._data.get("current_page_backwards")

    @property
    def next_page(self) -> str | None:
        """Cursor token for the next page, or None if no more pages."""
        return self._data.get("next_page")

    @property
    def previous_page(self) -> str | None:
        """Cursor token for the previous page, or None if at the start."""
        return self._data.get("previous_page")

    def has_next_page(self) -> bool:
        """True if there's a cursor for the next page."""
        return self.next_page is not None

    def has_previous_page(self) -> bool:
        """True if there's a cursor for the previous page."""
        return self.previous_page is not None

    def get_next_page(self) -> "SyncCursorPage[T, U] | None":
        """Fetch the next page; returns None if at the end or no fetch callback."""
        if self.next_page is None or self._fetch_next is None:
            return None
        return self._fetch_next(self.next_page)

    def get_previous_page(self) -> "SyncCursorPage[T, U] | None":
        """Fetch the previous page; returns None if at the start or no callback."""
        if self.previous_page is None or self._fetch_previous is None:
            return None
        return self._fetch_previous(self.previous_page)


class AsyncCursorPage(Generic[T, U]):
    """
    Async cursor-paginated result wrapper. See ``SyncCursorPage`` for semantics.
    """

    def __init__(
        self,
        data: dict[str, Any],
        item_type: type[T],
        transform_func: Callable[[T], U] | None = None,
        fetch_next: (Callable[[str], Awaitable["AsyncCursorPage[T, U]"]] | None) = None,
        fetch_previous: (
            Callable[[str], Awaitable["AsyncCursorPage[T, U]"]] | None
        ) = None,
    ) -> None:
        self._data: dict[str, Any] = data
        self._item_type: type[T] = item_type
        self._transform_func: Callable[[T], U] | None = transform_func
        self._fetch_next: Callable[[str], Awaitable["AsyncCursorPage[T, U]"]] | None = (
            fetch_next
        )
        self._fetch_previous: (
            Callable[[str], Awaitable["AsyncCursorPage[T, U]"]] | None
        ) = fetch_previous

        raw_items = data.get("items", [])
        self._raw_items: list[T] = [
            item_type.model_validate(item) for item in raw_items
        ]

    async def __aiter__(self) -> AsyncIterator[U] | AsyncIterator[T]:
        """
        Async iterate over all transformed items across all pages.

        Warning:
            Automatically chases ``next_page`` tokens until exhausted. For
            a mutating queue this may yield a different set of items than
            a single-shot snapshot. Use ``items`` for the current page only.
        """
        page: AsyncCursorPage[T, U] | None = self
        while page is not None:
            for item in page._raw_items:
                if self._transform_func is not None:
                    yield self._transform_func(item)
                else:
                    yield item
            page = await page.get_next_page()

    def __getitem__(self, index: int) -> U | T:
        item = self._raw_items[index]
        if self._transform_func is not None:
            return self._transform_func(item)
        return item

    def __len__(self) -> int:
        return len(self._raw_items)

    @property
    def items(self) -> list[U] | list[T]:
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._raw_items]
        return list(self._raw_items)

    @property
    def total(self) -> int | None:
        return self._data.get("total")

    @property
    def current_page(self) -> str | None:
        return self._data.get("current_page")

    @property
    def current_page_backwards(self) -> str | None:
        return self._data.get("current_page_backwards")

    @property
    def next_page(self) -> str | None:
        return self._data.get("next_page")

    @property
    def previous_page(self) -> str | None:
        return self._data.get("previous_page")

    def has_next_page(self) -> bool:
        return self.next_page is not None

    def has_previous_page(self) -> bool:
        return self.previous_page is not None

    async def get_next_page(self) -> "AsyncCursorPage[T, U] | None":
        if self.next_page is None or self._fetch_next is None:
            return None
        return await self._fetch_next(self.next_page)

    async def get_previous_page(self) -> "AsyncCursorPage[T, U] | None":
        if self.previous_page is None or self._fetch_previous is None:
            return None
        return await self._fetch_previous(self.previous_page)
