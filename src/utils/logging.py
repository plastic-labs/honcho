"""
Custom utility logging functions for Langfuse integration.
This module provides specialized formatters for all @observe decorated functions
and a conditional observe decorator that only applies when Langfuse is configured.
"""

import datetime
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

from fastapi import Request
from langfuse import observe  # pyright: ignore
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from src.config import settings
from src.utils.metrics_collector import append_metrics_to_file
from src.utils.representation import (
    Representation,
)

# Global console instance for consistent formatting
console = Console(markup=True)

COLLECT_METRICS_LOCAL = settings.COLLECT_METRICS_LOCAL

P = ParamSpec("P")
R = TypeVar("R")


@overload
def conditional_observe(
    func: Callable[P, R],
) -> Callable[P, R]: ...


@overload
def conditional_observe(
    *,
    name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def conditional_observe(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Conditionally apply the @observe decorator only when LANGFUSE_PUBLIC_KEY is present.

    Can be used in two ways:
    1. As a decorator: @conditional_observe
    2. As a decorator factory: @conditional_observe(name="...")

    Args:
        func: The function to potentially decorate (when used as @conditional_observe)
        name: Optional name for the observation (when used as @conditional_observe(name="..."))

    Returns:
        The decorated function if Langfuse is configured, otherwise the original function
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        if settings.LANGFUSE_PUBLIC_KEY:
            observe_name = name if name is not None else f.__name__
            return observe(name=observe_name)(f)
        else:
            return f

    if func is not None:
        # Used as @conditional_observe (without parentheses)
        return decorator(func)
    else:
        # Used as @conditional_observe(name="...") (with parentheses and keyword args)
        return decorator


# dict[task_name, list[tuple[metric_name, metric_value, metric_unit]]]
accumulated_metrics: dict[str, list[tuple[str, str | int | float, str]]] = {}


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


def log_representation(
    representation: Representation,
) -> None:
    """
    Log representation in a tree structure.
    Args:
        representation: Representation to log
    """
    tree = Tree("📊 REPRESENTATION")

    type_branch = tree.add(f"[bold cyan]EXPLICIT[/] ({len(representation.explicit)})")
    for i, obs in enumerate(representation.explicit, 1):
        type_branch.add(f"[dim]{i}.[/] {obs}")

    type_branch = tree.add(f"[bold cyan]DEDUCTIVE[/] ({len(representation.deductive)})")
    for i, obs in enumerate(representation.deductive, 1):
        type_branch.add(f"[dim]{i}.[/] {obs}")

    console.print(tree)
    console.print()


def accumulate_metric(
    task_name: str,
    label: str,
    value: str | int | float,
    unit: str,
) -> None:
    """
    Accumulate a metric value to be printed the next time log_performance_metrics is called.
    Args:
        label: Metric label
        value: Metric value
        unit: Metric unit
    """
    accumulated_metrics.setdefault(task_name, []).append((label, value, unit))


def log_performance_metrics(
    task_slug: str,
    task_name: str,
    metrics: list[tuple[str, str | int | float, str]] | None = None,
    title: str = "⚡ PERFORMANCE",
) -> None:
    """
    Log performance metrics in a clean table and optionally send to global collector.

    Args:
        task_slug: Slug of the task that generated these metrics
        task_name: Name of the task that generated these metrics
        metrics: Dictionary of metric names and (value, unit) tuples
        title: Table title
    """
    task_name = f"{task_slug}_{task_name}"
    if not accumulated_metrics.get(task_name) and not metrics:
        return
    if metrics is None:
        metrics = []
    metrics = accumulated_metrics.get(task_name, []) + metrics
    accumulated_metrics[task_name].clear()

    if COLLECT_METRICS_LOCAL:
        append_metrics_to_file(task_slug, task_name, metrics)

    # Remove metrics with "blob" unit type. They get printed separately below the table.
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
        if unit == "ms":
            formatted_value = f"{value:.0f}"
        elif unit == "s":
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = str(value)

        table.add_row(metric.replace("_", " ").title(), formatted_value, unit)

    # Build content for the panel
    content_items: list[RenderableType] = [table]

    if blob_metrics:
        content_items.append(Text(""))  # Empty line separator
        for metric, value, _unit in blob_metrics:
            content_items.append(Text(f"{metric}:", style="bold cyan"))
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


def save_reasoning_trace(
    provider: str,
    model: str,
    max_tokens: int,
    peer_id: str,
    peer_card: list[str] | None,
    message_created_at: datetime.datetime,
    working_representation: Representation,
    history: str,
    new_turns: list[str],
    explicit_prompt: str,
    explicit_response: dict[str, Any],
    deductive_prompt: str,
    deductive_response: dict[str, Any],
    atomic_propositions: list[str],
) -> None:
    """
    Save the reasoning trace (explicit + deductive calls) to trace.jsonl file.
    Uses JSONL format (one JSON object per line) for efficient appending.
    Only writes if COLLECT_METRICS_LOCAL is enabled.

    Args:
        provider: LLM provider name
        model: Model name
        max_tokens: Max tokens setting
        peer_id: ID of peer being analyzed
        peer_card: Peer card information
        message_created_at: Timestamp of message
        working_representation: Current representation context
        history: Conversation history
        new_turns: New conversation turns
        explicit_prompt: Prompt for explicit reasoning
        explicit_response: Response from explicit reasoning (as dict)
        deductive_prompt: Prompt for deductive reasoning
        deductive_response: Response from deductive reasoning (as dict)
        atomic_propositions: Atomic propositions passed to deductive reasoner
    """
    if not COLLECT_METRICS_LOCAL:
        return

    import fcntl
    import json
    from pathlib import Path

    trace_file = Path("trace.jsonl")

    trace_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "peer_id": peer_id,
        "peer_card": peer_card,
        "message_created_at": message_created_at.isoformat(),
        "working_representation": {
            "explicit": [obs.content for obs in working_representation.explicit],
            "implicit": [obs.content for obs in working_representation.implicit],
            "deductive": [
                {
                    "conclusion": obs.conclusion,
                    "premises": obs.premises,
                }
                for obs in working_representation.deductive
            ],
        },
        "history": history,
        "new_turns": new_turns,
        "explicit_call": {
            "prompt": explicit_prompt,
            "response": explicit_response,
        },
        "deductive_call": {
            "prompt": deductive_prompt,
            "response": deductive_response,
            "atomic_propositions": atomic_propositions,
        },
    }

    # Use file locking to handle concurrent writes from multiple processes
    with open(trace_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(trace_data) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
