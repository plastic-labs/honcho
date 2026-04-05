"""Pagination wrapper for Honcho SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", default=T)

__all__ = ["SyncPage", "AsyncPage"]


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
