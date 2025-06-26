from collections.abc import Callable, Iterator
from typing import Optional, TypeVar  # pyright: ignore

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
    def __init__(  # pyright: ignore
        self,
        original_page: SyncPageCore[T] = Field(
            ..., description="The original SyncPage to wrap"
        ),
        transform_func: Callable[[T], U] | None = Field(
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
        self._original_page = original_page  # pyright: ignore
        self._transform_func = transform_func  # pyright: ignore

    def __iter__(self) -> Iterator[U]:
        """Iterate over optionally transformed objects."""
        for item in self._original_page:
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item  # pyright: ignore

    def __getitem__(self, index: int) -> U:
        """Get an optionally transformed object by index."""
        item = self._original_page[index]  # pyright: ignore
        if self._transform_func is not None:
            return self._transform_func(item)  # pyright: ignore
        return item  # pyright: ignore

    def __len__(self) -> int:
        """Get the length of the page."""
        return len(self._original_page)  # pyright: ignore

    @property
    def data(self) -> list[U]:
        """Get all optionally transformed data as a list."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._original_page.data]  # pyright: ignore
        return self._original_page.data  # pyright: ignore

    @property
    def object(self) -> str:
        """Get the object type."""
        return self._original_page.object  # pyright: ignore

    @property
    def has_next_page(self) -> bool:  # pyright: ignore
        """Check if there's a next page."""
        return self._original_page.has_next_page  # pyright: ignore

    def next_page(self) -> Optional["SyncPage[U]"]:  # pyright: ignore
        """Get the next page with optional transformation applied."""
        next_page = self._original_page.next_page()  # pyright: ignore
        if next_page is None:
            return None
        return SyncPage(next_page, self._transform_func)  # pyright: ignore
