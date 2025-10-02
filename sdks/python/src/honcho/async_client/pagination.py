from collections.abc import AsyncIterator, Callable

from honcho_core.pagination import AsyncPage as AsyncPageCore
from typing_extensions import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U", default=T)


class AsyncPage(Generic[T, U]):
    """
    Async paginated result wrapper that transforms objects from type T to type U.

    Provides async iteration and transformation capabilities while preserving
    pagination functionality from the underlying core AsyncPage.
    """

    _original_page: AsyncPageCore[T]
    _transform_func: Callable[[T], U] | None

    def __init__(
        self,
        original_page: AsyncPageCore[T],
        transform_func: Callable[[T], U] | None = None,
    ) -> None:
        """
        Initialize the transformed async page.

        Args:
            original_page: The original AsyncPage to wrap
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
        """
        self._original_page = original_page
        self._transform_func = transform_func

    async def __aiter__(self) -> AsyncIterator[U] | AsyncIterator[T]:
        """Async iterate over all transformed items across all pages."""
        async for item in self._original_page:
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item

    def __getitem__(self, index: int) -> U | T:
        """Get a transformed item by index on the current page."""
        items = self._original_page.items or []
        item = items[index]
        if self._transform_func is not None:
            return self._transform_func(item)
        return item

    def __len__(self) -> int:
        """Get the number of items on the current page."""
        items = self._original_page.items or []
        return len(items)

    @property
    def items(self) -> list[U] | list[T]:
        """Get all transformed items on the current page."""
        items = self._original_page.items or []
        if self._transform_func is not None:
            return [self._transform_func(item) for item in items]
        return items

    @property
    def total(self) -> int | None:
        """Get the total number of items across all pages."""
        return self._original_page.total

    @property
    def page(self) -> int | None:
        """Get the current page number."""
        return self._original_page.page

    @property
    def size(self) -> int | None:
        """Get the page size."""
        return self._original_page.size

    @property
    def pages(self) -> int | None:
        """Get the total number of pages."""
        return self._original_page.pages

    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        return self._original_page.has_next_page()

    async def get_next_page(self) -> "AsyncPage[T, U] | None":
        """
        Fetch the next page of results.

        Returns None if there are no more pages.
        """
        if not hasattr(self._original_page, "get_next_page"):
            return None

        next_original_page = await self._original_page.get_next_page()
        if not next_original_page:
            return None

        return AsyncPage(next_original_page, self._transform_func)
