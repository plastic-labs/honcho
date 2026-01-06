"""Pagination classes for the Honcho API."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Generic, Iterator, TypeVar

if TYPE_CHECKING:
    pass

T = TypeVar("T")
U = TypeVar("U")


class SyncPage(Generic[T, U]):
    """Paginated result wrapper with sync iteration support.

    Supports transformation of items and on-demand page fetching.

    Type Parameters:
        T: The raw item type from the API
        U: The transformed item type (defaults to T if no transform)
    """

    def __init__(
        self,
        items: list[T],
        total: int | None,
        page: int,
        size: int,
        pages: int | None,
        transform_func: Callable[[T], U] | None = None,
        fetch_next: Callable[[], "SyncPage[T, U]"] | None = None,
    ):
        self._items = items
        self._total = total
        self._page = page
        self._size = size
        self._pages = pages
        self._transform_func = transform_func
        self._fetch_next = fetch_next

    @property
    def items(self) -> list[U]:
        """Get transformed items on the current page."""
        if self._transform_func:
            return [self._transform_func(item) for item in self._items]
        return self._items  # type: ignore

    @property
    def total(self) -> int | None:
        """Total number of items across all pages."""
        return self._total

    @property
    def page(self) -> int:
        """Current page number (1-indexed)."""
        return self._page

    @property
    def size(self) -> int:
        """Number of items per page."""
        return self._size

    @property
    def pages(self) -> int | None:
        """Total number of pages."""
        return self._pages

    def __len__(self) -> int:
        """Number of items on the current page."""
        return len(self._items)

    def __getitem__(self, index: int) -> U:
        """Get item by index on the current page."""
        item = self._items[index]
        if self._transform_func:
            return self._transform_func(item)
        return item  # type: ignore

    @property
    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        if self._pages is not None:
            return self._page < self._pages
        # If we don't know total pages, check if we got a full page
        return len(self._items) >= self._size

    def get_next_page(self) -> "SyncPage[T, U] | None":
        """Fetch the next page of results."""
        if not self.has_next_page or not self._fetch_next:
            return None
        return self._fetch_next()

    def __iter__(self) -> Iterator[U]:
        """Iterate over all items across all pages."""
        current: SyncPage[T, U] | None = self
        while current is not None:
            for item in current._items:
                if self._transform_func:
                    yield self._transform_func(item)
                else:
                    yield item  # type: ignore
            current = current.get_next_page()


class AsyncPage(Generic[T, U]):
    """Paginated result wrapper with async iteration support.

    Supports transformation of items and on-demand page fetching.

    Type Parameters:
        T: The raw item type from the API
        U: The transformed item type (defaults to T if no transform)
    """

    def __init__(
        self,
        items: list[T],
        total: int | None,
        page: int,
        size: int,
        pages: int | None,
        transform_func: Callable[[T], U] | None = None,
        fetch_next: Callable[[], Awaitable["AsyncPage[T, U]"]] | None = None,
    ):
        self._items = items
        self._total = total
        self._page = page
        self._size = size
        self._pages = pages
        self._transform_func = transform_func
        self._fetch_next = fetch_next

    @property
    def items(self) -> list[U]:
        """Get transformed items on the current page."""
        if self._transform_func:
            return [self._transform_func(item) for item in self._items]
        return self._items  # type: ignore

    @property
    def total(self) -> int | None:
        """Total number of items across all pages."""
        return self._total

    @property
    def page(self) -> int:
        """Current page number (1-indexed)."""
        return self._page

    @property
    def size(self) -> int:
        """Number of items per page."""
        return self._size

    @property
    def pages(self) -> int | None:
        """Total number of pages."""
        return self._pages

    def __len__(self) -> int:
        """Number of items on the current page."""
        return len(self._items)

    def __getitem__(self, index: int) -> U:
        """Get item by index on the current page."""
        item = self._items[index]
        if self._transform_func:
            return self._transform_func(item)
        return item  # type: ignore

    @property
    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        if self._pages is not None:
            return self._page < self._pages
        # If we don't know total pages, check if we got a full page
        return len(self._items) >= self._size

    async def get_next_page(self) -> "AsyncPage[T, U] | None":
        """Fetch the next page of results."""
        if not self.has_next_page or not self._fetch_next:
            return None
        return await self._fetch_next()

    async def __aiter__(self) -> AsyncIterator[U]:
        """Async iterate over all items across all pages."""
        current: AsyncPage[T, U] | None = self
        while current is not None:
            for item in current._items:
                if self._transform_func:
                    yield self._transform_func(item)
                else:
                    yield item  # type: ignore
            current = await current.get_next_page()
