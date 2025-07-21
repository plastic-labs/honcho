"""
Rich-powered logging utilities for beautiful console output.
"""

import datetime
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.utils.logging import console
from src.utils.shared_models import ObservationDict, ReasoningResponseWithThinking


class ObservationWithContent(Protocol):
    """Protocol for objects with content attribute."""

    content: str


class ObservationWithConclusion(Protocol):
    """Protocol for objects with conclusion and optional premises."""

    conclusion: str
    premises: Sequence[str] | None


# Union type for all possible observation types
ObservationType = (
    str
    | ObservationDict
    | ObservationWithContent
    | ObservationWithConclusion
    | dict[str, Any]
)


def format_reasoning_response_as_markdown(
    response: ReasoningResponseWithThinking | None,
) -> str:
    """
    Format a ReasoningResponse object as markdown.
    Args:
        response: ReasoningResponse object or similar structure
    Returns:
        Formatted markdown string
    """
    if not response:
        return "No reasoning response available"

    parts: list[str] = []

    # Add thinking section if available
    if hasattr(response, "thinking") and response.thinking:
        parts.append("## Thinking\n")
        parts.append(response.thinking.strip())
        parts.append("")

    # Add explicit observations
    if hasattr(response, "explicit") and response.explicit:
        parts.append("## Explicit Observations\n")
        for i, obs in enumerate(response.explicit, 1):
            parts.append(f"{i}. {obs}")
        parts.append("")

    # Add deductive observations
    if hasattr(response, "deductive") and response.deductive:
        parts.append("## Deductive Observations\n")
        for i, obs in enumerate(response.deductive, 1):
            if hasattr(obs, "conclusion"):
                parts.append(f"{i}. **Conclusion**: {obs.conclusion}")
                if hasattr(obs, "premises") and obs.premises:
                    parts.append("   **Premises**:")
                    for premise in obs.premises:
                        parts.append(f"   - {premise}")
                parts.append("")
            else:
                parts.append(f"{i}. {obs}")
        parts.append("")

    return "\n".join(parts)


def format_reasoning_inputs_as_markdown(
    context: ReasoningResponseWithThinking | None,
    history: str,
    new_turn: str,
    message_created_at: datetime.datetime,
) -> str:
    """
    Format reasoning inputs as markdown for logging.
    Args:
        context: Current context/observations
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

    # Add context if available
    if context:
        parts.append("### Current Context\n")
        if hasattr(context, "explicit") and context.explicit:
            parts.append("**Explicit Observations**:")
            for obs in context.explicit:
                parts.append(f"- {obs}")
            parts.append("")

        if hasattr(context, "deductive") and context.deductive:
            parts.append("**Deductive Observations**:")
            for obs in context.deductive:
                if hasattr(obs, "conclusion"):
                    parts.append(f"- {obs.conclusion}")
                else:
                    parts.append(f"- {obs}")
            parts.append("")

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


def log_thinking_panel(
    thinking: str | None,
) -> None:
    """
    Log thinking content in a beautiful panel.
    Args:
        thinking: Thinking content to display (can be None)
    """
    if not thinking:
        console.print("[dim]No thinking content available[/]")
        return

    panel = Panel(
        thinking.strip(),
        title="🧠 THINKING",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )

    # Use console.print for immediate output only
    console.print(panel)


def log_observations_tree(
    observations: dict[str, list[Any]],
) -> None:
    """
    Log observations in a tree structure.
    Args:
        observations: Dictionary of observation types and their lists
    """
    tree = Tree("📊 OBSERVATIONS")

    for obs_type, obs_list in observations.items():
        if obs_list:
            type_branch = tree.add(
                f"[bold cyan]{obs_type.title()}[/] ({len(obs_list)})"
            )

            for i, obs in enumerate(obs_list):  # Show all observations
                content = _extract_observation_text(obs)
                truncated = content[:120] + "..." if len(content) > 120 else content
                type_branch.add(f"[dim]{i + 1}.[/] {truncated}")

    console.print(tree)


def log_performance_metrics(
    metrics: Mapping[str, str | int | float],
    title: str = "⚡ PERFORMANCE",
) -> None:
    """
    Log performance metrics in a clean table.
    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    table.add_column("Unit", style="dim")

    for metric, value in metrics.items():
        if isinstance(value, float):
            if "duration" in metric.lower() or "time" in metric.lower():
                formatted_value = f"{value:.2f}"
                unit = "ms" if value < 1000 else "s"
            elif "score" in metric.lower() or "percentage" in metric.lower():
                formatted_value = f"{value:.1%}"
                unit = ""
            else:
                formatted_value = f"{value:.3f}"
                unit = ""
        else:
            formatted_value = str(value)
            unit = ""

        table.add_row(metric.replace("_", " ").title(), formatted_value, unit)

    console.print(table)


def _extract_observation_text(obs: ObservationType) -> str:
    """Extract text content from various observation types, including premises."""
    if isinstance(obs, str):
        return obs
    elif isinstance(obs, dict):
        # Handle dict-based structured observations first
        if "conclusion" in obs:
            conclusion: str = str(obs["conclusion"])
            premises: list[Any] = list(obs.get("premises", []))
            if premises:
                premises_text = "\n" + "\n".join(f"    - {str(p)}" for p in premises)
                return f"{conclusion}{premises_text}"
            return conclusion
        return str(obs.get("content", obs))
    else:
        # Handle object-based observations
        # Use Any type for this branch since we're doing dynamic attribute checking
        obj: Any = obs
        if hasattr(obj, "conclusion"):
            conclusion = str(obj.conclusion)
            if hasattr(obj, "premises") and obj.premises:
                premises_text = "\n" + "\n".join(
                    f"    - {str(p)}" for p in obj.premises
                )
                return f"{conclusion}{premises_text}"
            return conclusion
        elif hasattr(obj, "content"):
            return str(obj.content)
        else:
            return str(obj)
