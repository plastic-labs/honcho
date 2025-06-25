from collections.abc import Iterator
from typing import Callable, Optional, TypeVar

from honcho_core.pagination import SyncPage as SyncPageCore
from pydantic import Field, validate_call

T = TypeVar("T")
U = TypeVar("U")


class SyncPage(SyncPageCore[U]):
    """
    Paginated result wrapper that transforms objects from type T to type U.

    Provides iteration and transformation capabilities while preserving
    pagination functionality from the underlying core SyncPage.
    """

    @validate_call
    def __init__(
        self,
        original_page: SyncPageCore[T] = Field(
            ..., description="The original SyncPage to wrap"
        ),
        transform_func: Optional[Callable[[T], U]] = Field(
            None,
            description="Optional function to transform objects from type T to type U",
        ),
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

    def __iter__(self) -> Iterator[U]:
        """Iterate over optionally transformed objects."""
        for item in self._original_page:
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item  # type: ignore

    def __getitem__(self, index: int) -> U:
        """Get an optionally transformed object by index."""
        item = self._original_page[index]
        if self._transform_func is not None:
            return self._transform_func(item)
        else:
            return item  # type: ignore

    def __len__(self) -> int:
        """Get the length of the page."""
        return len(self._original_page)

    @property
    def data(self) -> list[U]:
        """Get all optionally transformed data as a list."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._original_page.data]
        else:
            return self._original_page.data  # type: ignore

    @property
    def object(self) -> str:
        """Get the object type."""
        return self._original_page.object

    @property
    def has_next_page(self) -> bool:
        """Check if there's a next page."""
        return self._original_page.has_next_page

    def next_page(self) -> Optional["SyncPage[U]"]:
        """Get the next page with optional transformation applied."""
        next_page = self._original_page.next_page()
        if next_page is None:
            return None
        return SyncPage(next_page, self._transform_func)
