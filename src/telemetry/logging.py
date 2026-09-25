"""
Custom utility logging functions for Langfuse integration.
This module provides specialized formatters for all @observe decorated functions
and a conditional observe decorator that only applies when Langfuse is configured.
"""

import datetime
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar, overload

from fastapi import Request
from langfuse import observe
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import settings
from src.telemetry.metrics_collector import append_metrics_to_file
from src.utils.representation import (
    Representation,
)

logger = logging.getLogger(__name__)

# Global console instance for consistent formatting
console = Console(markup=True)

COLLECT_METRICS_LOCAL = settings.COLLECT_METRICS_LOCAL

P = ParamSpec("P")
R = TypeVar("R")

# Langfuse observation types accepted by `@observe(as_type=...)`. Mirrors the
# literal union the SDK exposes; kept local so callers don't import langfuse
# internals just to name an observation type.
ObserveAsType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]


@overload
def conditional_observe(
    func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def conditional_observe(
    *,
    name: str | None = None,
    as_type: ObserveAsType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def conditional_observe(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    as_type: ObserveAsType | None = None,
    capture_input: bool | None = None,
    capture_output: bool | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Conditionally apply the @observe decorator only in legacy inline mode
    (``langfuse_inline_enabled`` — i.e. a key is set AND
    ``LANGFUSE_EXPORTER_MODE == "inline"``). In exporter mode the LangfuseExporter
    rebuilds every observation from the captured trace stream, so a live
    @observe span here would double-emit.

    Can be used in two ways:
    1. As a decorator: @conditional_observe
    2. As a decorator factory: @conditional_observe(name="...", as_type="generation")

    Args:
        func: The function to potentially decorate (when used as @conditional_observe)
        name: Optional name for the observation (when used as @conditional_observe(name="..."))
        as_type: Optional Langfuse observation type (e.g. "generation", "tool"). When
            omitted, Langfuse infers a default span.
        capture_input: When ``False``, Langfuse does NOT auto-serialize the
            function's arguments into the span input. Set this on functions that
            receive live SDK clients or secret-bearing config as parameters
            (e.g. the LLM executor): auto-capture would deep-copy those clients
            into throwaway, half-constructed objects whose GC raises
            ``AsyncHttpxClientWrapper ... no attribute '_state'`` /
            ``BaseApiClient ... no attribute '_http_options'`` (see HONCHO-4HA),
            and would also leak ``ModelConfig.api_key`` into traces. Pair with an
            explicit ``update_current_generation(input=...)`` call to keep
            full-fidelity input. ``None`` leaves the SDK default (capture on).
        capture_output: When ``False``, Langfuse does NOT auto-serialize the
            return value. Pair with an explicit
            ``update_current_generation(output=...)``. ``None`` = SDK default.

    Returns:
        The decorated function if Langfuse is configured, otherwise the original function
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Only auto-instrument with @observe in legacy inline mode. In exporter
        # mode the LangfuseExporter produces every observation from the captured
        # trace stream, so a live @observe span here would double-emit.
        if not settings.langfuse_inline_enabled:
            return f
        # `observe` treats None as "use SDK default", so passing the optionals
        # straight through is equivalent to omitting them.
        return observe(
            name=name if name is not None else f.__name__,
            as_type=as_type,
            capture_input=capture_input,
            capture_output=capture_output,
        )(f)

    if func is not None:
        # Used as @conditional_observe (without parentheses)
        return decorator(func)
    else:
        # Used as @conditional_observe(name="...") (with parentheses and keyword args)
        return decorator


def flush_langfuse() -> None:
    """Flush buffered Langfuse spans on shutdown.

    The SDK's background timer/atexit hook don't fire reliably on SIGTERM, so
    the final batch is dropped without this. No-op when Langfuse is unconfigured.
    """
    if not settings.LANGFUSE_PUBLIC_KEY:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.debug("Failed to flush Langfuse on shutdown", exc_info=True)


# Bounded OrderedDict for accumulated metrics to prevent memory leaks.
# If an exception occurs between accumulate_metric() and log_performance_metrics(),
# the metrics would stay in memory forever. Using OrderedDict allows us to evict
# the oldest entries (FIFO) when exceeding MAX_ACCUMULATED_TASKS.
MAX_ACCUMULATED_TASKS = 1000

# OrderedDict[task_name, list[tuple[metric_name, metric_value, metric_unit]]]
accumulated_metrics: OrderedDict[str, list[tuple[str, str | int | float, str]]] = (
    OrderedDict()
)


def format_reasoning_inputs_as_markdown(
    representation: Representation,
    history: str,
    new_turn: str,
    message_created_at: datetime.datetime,
) -> str:
    """
    Format reasoning inputs as markdown for logging.
    Args:
        representation: Current/working representation
        history: Conversation history
        new_turn: New user message
        message_created_at: Message timestamp
    Returns:
        Formatted markdown string
    """
    parts: list[str] = []

    parts.append("## Reasoning Inputs\n")
    parts.append(
        f"**Current Time**: {message_created_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    parts.append("")

    # Add representation
    parts.append("### Current Representation\n")
    parts.append(representation.format_as_markdown())

    # Add history
    if history:
        parts.append("### Conversation History\n")
        parts.append(history.strip())
        parts.append("")

    # Add new turn
    if new_turn:
        parts.append("### New Turn\n")
        parts.append(new_turn.strip())
        parts.append("")

    return "\n".join(parts)


def accumulate_metric(
    task_name: str,
    label: str,
    value: str | int | float,
    unit: str,
) -> None:
    """
    Accumulate a metric value to be printed the next time log_performance_metrics is called.

    This function uses a bounded OrderedDict to prevent memory leaks. If the number of
    tracked tasks exceeds MAX_ACCUMULATED_TASKS, the oldest entries are evicted (FIFO).

    Args:
        task_name: Task identifier
        label: Metric label
        value: Metric value
        unit: Metric unit
    """
    # Evict oldest entries if we've exceeded the maximum to prevent memory leaks.
    # This handles the case where exceptions occur before log_performance_metrics is called.
    while (
        len(accumulated_metrics) >= MAX_ACCUMULATED_TASKS
        and task_name not in accumulated_metrics
    ):
        # popitem(last=False) removes the oldest (first inserted) entry
        accumulated_metrics.popitem(last=False)

    accumulated_metrics.setdefault(task_name, []).append((label, value, unit))


def log_token_usage_metrics(
    task_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
) -> None:
    """
    Log cache-aware token usage metrics.

    Args:
        task_name: The task name for metric accumulation
        input_tokens: Total input tokens (cached + uncached)
        output_tokens: Output tokens generated
        cache_read_input_tokens: Tokens read from cache (90% cheaper)
        cache_creation_input_tokens: Tokens written to cache (25% more expensive)

    Returns:
        None
    """
    accumulate_metric(task_name, "input_tokens", input_tokens, "tokens")
    accumulate_metric(
        task_name,
        "cache_read_input_tokens",
        cache_read_input_tokens,
        "tokens",
    )
    accumulate_metric(
        task_name,
        "cache_creation_input_tokens",
        cache_creation_input_tokens,
        "tokens",
    )
    # Total uncached tokens (what you're paying full price for)
    # = total - cache_read (those were cheap) + cache_creation (those cost 1.25x)
    uncached_input_tokens = (
        input_tokens - cache_read_input_tokens + cache_creation_input_tokens
    )
    accumulate_metric(
        task_name, "uncached_input_tokens", uncached_input_tokens, "tokens"
    )
    accumulate_metric(task_name, "output_tokens", output_tokens, "tokens")


def log_performance_metrics(
    task_slug: str,
    task_name: str,
    metrics: list[tuple[str, str | int | float, str]] | None = None,
    title: str = "PERFORMANCE",
) -> None:
    """
    Log performance metrics and optionally send them to the global collector.

    PERFORMANCE_LOG_FORMAT=compact emits numeric metrics on one INFO line and
    keeps large "blob" metrics at DEBUG. PERFORMANCE_LOG_FORMAT=rich prints the
    local Rich panel, including blob metrics, for interactive readability.

    Args:
        task_slug: Slug of the task that generated these metrics
        task_name: Name of the task that generated these metrics
        metrics: List of (metric_name, value, unit) tuples
        title: Prefix for the log line
    """
    task_name = f"{task_slug}_{task_name}"
    # No-op if metrics were evicted (due to MAX_ACCUMULATED_TASKS limit) and no
    # additional metrics are passed. This handles the case where an exception
    # caused old metrics to be evicted before log_performance_metrics was called.
    if not accumulated_metrics.get(task_name) and not metrics:
        return
    if metrics is None:
        metrics = []
    metrics = accumulated_metrics.pop(task_name, []) + metrics

    if COLLECT_METRICS_LOCAL:
        append_metrics_to_file(task_slug, task_name, metrics)

    if settings.PERFORMANCE_LOG_FORMAT == "rich":
        _log_performance_metrics_rich(task_name, metrics, title)
        return

    # Keep large text payloads out of the compact INFO summary.
    blob_metrics: list[tuple[str, str | int | float, str]] = []
    summary_parts: list[str] = []
    for metric, value, unit in metrics:
        if unit == "blob":
            blob_metrics.append((metric, value, unit))
            continue
        if unit == "ms" and isinstance(value, int | float):
            formatted_value = f"{value:.0f}ms"
        elif unit == "s" and isinstance(value, int | float):
            formatted_value = f"{value:.3f}s"
        elif unit in ("", "tokens", "count", "id"):
            formatted_value = str(value)
        else:
            formatted_value = f"{value}{unit}"
        summary_parts.append(f"{metric}={formatted_value}")

    if summary_parts:
        logger.info("%s %s | %s", title, task_name, " | ".join(summary_parts))
    else:
        logger.info("%s %s", title, task_name)

    for metric, value, _unit in blob_metrics:
        logger.debug("%s %s :: %s\n%s", title, task_name, metric, value)


def _log_performance_metrics_rich(
    task_name: str,
    metrics: list[tuple[str, str | int | float, str]],
    title: str,
) -> None:
    blob_metrics: list[tuple[str, str | int | float, str]] = []
    non_blob_metrics: list[tuple[str, str | int | float, str]] = []
    for metric in metrics:
        (blob_metrics if metric[2] == "blob" else non_blob_metrics).append(metric)

    table = Table(
        show_header=True,
        header_style="bold green",
        box=None,
        padding=(0, 1),
    )
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", justify="right", style="yellow", width=15)
    table.add_column("Unit", style="dim", width=8)

    for metric, value, unit in non_blob_metrics:
        if unit == "ms" and isinstance(value, int | float):
            formatted_value = f"{value:.0f}"
        elif unit == "s" and isinstance(value, int | float):
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = str(value)

        table.add_row(metric.replace("_", " ").title(), formatted_value, unit)

    content_items: list[RenderableType] = [table]

    for metric, value, _unit in blob_metrics:
        content_items.append(Text.assemble("    ", (f"\n{metric}:", "bold"), "     "))
        content_items.append(Text(str(value)))

    panel = Panel(
        Group(*content_items),
        title=f"[bold green]{title} - {task_name}[/]",
        box=box.ROUNDED,
        padding=(1, 2),
        width=80,
    )

    console.print(panel)
    console.print()


def normalize_template_path(path: str) -> str:
    if path != "/" and path.endswith("/"):
        return path.rstrip("/")
    return path


def get_route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return normalize_template_path(route.path)
    return "unknown"
