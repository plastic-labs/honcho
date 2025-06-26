from collections.abc import AsyncIterator, Callable
from typing import TypeVar

from honcho_core.pagination import AsyncPage as AsyncPageCore
from pydantic import Field, validate_call

T = TypeVar("T")
U = TypeVar("U")


class AsyncPage(AsyncPageCore[U]):
    """
    Async paginated result wrapper that transforms objects from type T to type U.

    Provides async iteration and transformation capabilities while preserving
    pagination functionality from the underlying core AsyncPage.
    """

    @validate_call
    def __init__(
        self,
        original_page: AsyncPageCore[T] = Field(
            ..., description="The original AsyncPage to wrap"
        ),
        transform_func: Callable[[T], U] | None = Field(
            None,
            description="Optional function to transform objects from type T to type U",
        ),
    ) -> None:
        """
        Initialize the transformed async page.

        Args:
            original_page: The original AsyncPage to wrap
            transform_func: Optional function to transform objects from type T to type U.
                            If None, objects are passed through unchanged.
        """
        super().__init__(items=original_page.items)  # pyright: ignore
        self._original_page = original_page  # pyright: ignore
        self._transform_func = transform_func  # pyright: ignore

    @property
    def items(self) -> list[U]:  # pyright: ignore
        """Get all optionally transformed items as a list."""
        if self._transform_func is not None:
            return [self._transform_func(item) for item in self._original_page.items]
        return self._original_page.items  # pyright: ignore

    async def __aiter__(self) -> AsyncIterator[U]:
        """Async iterate over optionally transformed objects."""
        async for item in self._original_page:
            if self._transform_func is not None:
                yield self._transform_func(item)
            else:
                yield item  # type: ignore # pyright: ignore

    async def __agetitem__(self, index: int) -> U:
        """Get an optionally transformed object by index."""
        item = await self._original_page.__agetitem__(index)  # type: ignore # pyright: ignore
        if self._transform_func is not None:
            return self._transform_func(item)  # pyright: ignore
        return item  # type: ignore # pyright: ignore

    def __len__(self) -> int:
        """Get the length of the page."""
        return len(self._original_page)  # type: ignore # pyright: ignore

    @property
    async def data(self) -> list[U]:
        """Get all optionally transformed data as a list."""
        data = await self._original_page.data  # type: ignore # pyright: ignore
        if self._transform_func is not None:
            return [self._transform_func(item) for item in data]  # pyright: ignore
        return data  # type: ignore # pyright: ignore

    @property
    def object(self) -> str:
        """Get the object type."""
        return self._original_page.object  # type: ignore # pyright: ignore

    @property
    def has_next_page(self) -> bool:  # pyright: ignore
        """Check if there's a next page."""
        return self._original_page.has_next_page  # type: ignore # pyright: ignore

    async def next_page(self) -> "AsyncPage[U] | None":
        """Get the next page with optional transformation applied."""
        next_page = await self._original_page.next_page()  # type: ignore # pyright: ignore
        if next_page is None:
            return None
        return AsyncPage(next_page, self._transform_func)  # pyright: ignore
