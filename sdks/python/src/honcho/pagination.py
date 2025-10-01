from collections.abc import Callable, Iterator
from typing import Generic

from honcho_core.pagination import SyncPage as SyncPageCore
from typing_extensions import TypeVar

T = TypeVar("T")
U = TypeVar("U", default=T)


class SyncPage(Generic[T, U]):
    """
    Paginated result wrapper that transforms objects from type T to type U.

    Provides iteration and transformation capabilities while preserving
    pagination functionality from the underlying core SyncPage.
    """

    _original_page: SyncPageCore[T]
    _transform_func: Callable[[T], U] | None

    def __init__(
        self,
        original_page: SyncPageCore[T],
        transform_func: Callable[[T], U] | None = None,
    ) -> None:
        """
        Initialize the transformed page.

        Args:
            original_page: The original SyncPage to wrap
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
        """
        self._original_page = original_page
        self._transform_func = transform_func

    def __iter__(self) -> Iterator[U] | Iterator[T]:
        """Iterate over all transformed items across all pages."""
        for item in self._original_page:
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

    def get_next_page(self) -> "SyncPage[T, U] | None":
        """
        Fetch the next page of results.

        Returns None if there are no more pages.
        """
        if not hasattr(self._original_page, "get_next_page"):
            return None

        next_original_page = self._original_page.get_next_page()
        if not next_original_page:
            return None

        return SyncPage(next_original_page, self._transform_func)
