from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

T = TypeVar("T")

# Context variable for tracking current iteration in tool execution loop
# This is used for telemetry to associate tool calls with their iteration
_current_iteration: ContextVar[int] = ContextVar("current_iteration", default=0)


def set_current_iteration(iteration: int) -> None:
    """Set the current iteration number for telemetry context."""
    _current_iteration.set(iteration)


def get_current_iteration() -> int:
    """Get the current iteration number from telemetry context."""
    return _current_iteration.get()


@dataclass
class GetOrCreateResult(Generic[T]):
    """Result of a get_or_create operation indicating whether the resource was created."""

    resource: T
    created: bool
    on_commit: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def post_commit(self) -> None:
        """Run deferred cache operations after the transaction is committed."""
        if self.on_commit is not None:
            await self.on_commit()


SupportedProviders = Literal["anthropic", "openai", "google", "groq", "custom", "vllm"]
TaskType = Literal[
    "webhook", "summary", "representation", "dream", "deletion", "reconciler"
]
VectorSyncState = Literal["synced", "pending", "failed"]
DocumentLevel = Literal["explicit", "deductive", "inductive", "contradiction"]
