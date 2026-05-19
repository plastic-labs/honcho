from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

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


# ordinal of the tool call within its iteration. Two calls
# to the same tool in one iteration (the model can do this) need distinct
# resource ids on AgentToolCallCompletedEvent — seq disambiguates.
_current_tool_call_seq: ContextVar[int] = ContextVar("current_tool_call_seq", default=0)
# Optional provider-supplied tool-call id (e.g. Anthropic's `toolu_*`) so the
# emitted event can be cross-referenced with provider logs.
_current_provider_tool_call_id: ContextVar[str | None] = ContextVar(
    "current_provider_tool_call_id", default=None
)


def set_current_tool_call_seq(seq: int, provider_tool_call_id: str | None) -> None:
    """Set the current tool-call ordinal + provider id for telemetry context.

    Called by tool_loop before invoking the tool_executor closure. The seq
    starts at 0 within each iteration's tool batch and increments per call.
    """
    _current_tool_call_seq.set(seq)
    _current_provider_tool_call_id.set(provider_tool_call_id)


def get_current_tool_call_seq() -> int:
    return _current_tool_call_seq.get()


def get_current_provider_tool_call_id() -> str | None:
    return _current_provider_tool_call_id.get()


# After `execute_tool` finishes, the metadata dict from the handler's
# ToolResult is published here so tool_loop can stash it on the
# `all_tool_calls` entry (which DreamSpecialistEvent reads for rollups).
# Default is None (not {}) per ruff B039 — mutable defaults on
# ContextVars are foot-guns; the getter normalizes None → {}.
_last_tool_metadata: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_tool_metadata", default=None
)


def set_last_tool_metadata(metadata: dict[str, Any]) -> None:
    """Publish the just-finished tool call's ToolResult metadata."""
    _last_tool_metadata.set(metadata)


def get_last_tool_metadata() -> dict[str, Any]:
    """Read the last tool call's metadata. Returns {} when no ToolResult was returned."""
    return _last_tool_metadata.get() or {}


@contextmanager
def iteration_scope() -> Generator[None]:
    """Reset per-tool-loop ContextVars on exit.

    Wrap the body of `tool_loop.run_tool_loop` so a subsequent loop in the
    same asyncio Task (worker batches, tests using TestClient) starts with
    fresh iteration / tool-call state instead of inheriting stale values
    from a prior loop. ContextVars are per-Task in asyncio, so cross-request
    leakage is unlikely under normal FastAPI use — but this is defensive and
    cheap.
    """
    iter_token = _current_iteration.set(0)
    seq_token = _current_tool_call_seq.set(0)
    pid_token = _current_provider_tool_call_id.set(None)
    meta_token = _last_tool_metadata.set(None)
    try:
        yield
    finally:
        _current_iteration.reset(iter_token)
        _current_tool_call_seq.reset(seq_token)
        _current_provider_tool_call_id.reset(pid_token)
        _last_tool_metadata.reset(meta_token)


# embedding-call purpose ContextVar. Callers wrap embedding-driving
# operations in `with embedding_call_purpose("search_memory"): ...` so the
# embedding client can stamp every provider call with the originating intent
# without changing the call signature. None = caller didn't instrument; the
# event still emits but with call_purpose unset.
_embedding_call_purpose: ContextVar[str | None] = ContextVar(
    "embedding_call_purpose", default=None
)
# Companion ContextVars so the event can also carry workspace + run
# correlation without threading kwargs through every embedding call site.
_embedding_workspace_name: ContextVar[str | None] = ContextVar(
    "embedding_workspace_name", default=None
)
_embedding_run_id: ContextVar[str | None] = ContextVar("embedding_run_id", default=None)
# Parent category for joining EmbeddingCallCompletedEvent against the
# workflow that drove the call (e.g. "dialectic", "deriver",
# "reconciliation", "api"). Optional — None when uninstrumented.
_embedding_parent_category: ContextVar[str | None] = ContextVar(
    "embedding_parent_category", default=None
)


def get_embedding_call_purpose() -> str | None:
    """Read the current embedding call purpose. None when uninstrumented."""
    return _embedding_call_purpose.get()


def get_embedding_workspace_name() -> str | None:
    """Read the workspace name attached to the current embedding call scope."""
    return _embedding_workspace_name.get()


def get_embedding_run_id() -> str | None:
    """Read the run_id attached to the current embedding call scope."""
    return _embedding_run_id.get()


def get_embedding_parent_category() -> str | None:
    """Read the parent category attached to the current embedding call scope."""
    return _embedding_parent_category.get()


@contextmanager
def embedding_call_purpose(
    purpose: str,
    *,
    workspace_name: str | None = None,
    run_id: str | None = None,
    parent_category: str | None = None,
) -> Generator[None]:
    """Tag any embedding calls made inside this `with` block.

    `purpose` should match an `EmbeddingCallPurpose` enum value (see
    src/telemetry/events/llm.py). Unknown values pass through silently and
    land as None on the event — the emitter validates against the enum.

    `workspace_name` and `run_id` let the emitted event correlate back to
    a specific workspace and agent run. Both are optional; callers that
    don't have one (or have it set further up the stack via a wider
    `with` block) can omit it.

    `parent_category` joins the event back to the originating workflow —
    typically the same category used by the calling LLM agent ("dialectic",
    "deriver", "reconciliation", "api"). Lets analytics pivot embedding
    cost/latency by workflow without per-purpose joins.
    """
    purpose_token = _embedding_call_purpose.set(purpose)
    workspace_token = (
        _embedding_workspace_name.set(workspace_name)
        if workspace_name is not None
        else None
    )
    run_id_token = _embedding_run_id.set(run_id) if run_id is not None else None
    parent_category_token = (
        _embedding_parent_category.set(parent_category)
        if parent_category is not None
        else None
    )
    try:
        yield
    finally:
        _embedding_call_purpose.reset(purpose_token)
        if workspace_token is not None:
            _embedding_workspace_name.reset(workspace_token)
        if run_id_token is not None:
            _embedding_run_id.reset(run_id_token)
        if parent_category_token is not None:
            _embedding_parent_category.reset(parent_category_token)


@dataclass
class ToolResult:
    """Internal return shape used by tool handlers.

    Handlers may continue to return a plain `str` (existing contract). When
    they need to carry structured metadata for downstream events — search
    `top_k`/`results_count` for AgentToolCallCompletedEvent, or
    `created_count`/`deleted_count` for specialist rollups — they
    return `ToolResult(content=..., metadata={...})` instead. The
    `execute_tool` closure in `create_tool_executor` unwraps the dataclass
    before returning the string to `tool_loop`.

    Treat this as a private contract between agent_tools.py and tool_loop.py.
    Public callers see only the string content.

    The `__contains__` and `__str__` overrides exist so direct handler-unit
    tests that predate the dataclass — e.g. `assert "Created 2" in result`
    — keep working without churning every assertion. Anything beyond
    substring / str() (like `.lower()`) should access `.content` explicitly.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, item: object) -> bool:
        # Only meaningful for substring checks; matches the legacy str-return
        # contract used by handler unit tests.
        if not isinstance(item, str):
            return False
        return item in self.content

    def __str__(self) -> str:
        return self.content


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


TaskType = Literal[
    "webhook", "summary", "representation", "dream", "deletion", "reconciler"
]
VectorSyncState = Literal["synced", "pending", "failed"]
DocumentLevel = Literal["explicit", "deductive", "inductive", "contradiction"]
