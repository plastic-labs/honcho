"""
Rich-powered logging utilities for beautiful console output.
"""

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Global console instance for direct rich output
console = Console()


def setup_rich_logging(level: str = "INFO") -> None:
    """
    Configure rich logging with beautiful formatting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string level to logging constant
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    numeric_level = log_levels.get(level.upper(), logging.INFO)

    # Configure root logger with RichHandler
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_suppress=[
                    "fastapi",
                    "starlette",
                    "uvicorn",
                    "sqlalchemy",
                    "asyncio",
                ],
                markup=True,
            )
        ],
        force=True,  # Override any existing configuration
    )

    # Disable SQLAlchemy engine logging (it's too verbose)
    logging.getLogger("sqlalchemy.engine.Engine").disabled = True


def log_thinking_panel(
    thinking: str,
    depth: int = 0,
    title: str = "🧠 THINKING",
) -> None:
    """
    Log thinking content in a beautiful panel.

    Args:
        thinking: Thinking content to display
        depth: Recursion depth for context
        title: Panel title
    """
    panel_title = f"{title} (Depth {depth})" if depth > 0 else title

    panel = Panel(
        thinking.strip(),
        title=panel_title,
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )

    # Use console.print for immediate output only
    console.print(panel)


def log_observations_tree(
    observations: dict[str, list[Any]],
    title: str = "📊 OBSERVATIONS",
) -> None:
    """
    Log observations in a tree structure.

    Args:
        observations: Dictionary of observation types and their lists
        title: Tree root title
    """
    tree = Tree(title)

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


def log_changes_table(
    changes: dict[str, dict[str, list[str]]],
    significance_score: float,
    threshold: float,
) -> None:
    """
    Log detected changes in a formatted table.

    Args:
        changes: Dictionary of changes by type and action
        significance_score: Calculated significance score
        threshold: Threshold for significance
    """
    table = Table(
        title="🔄 DETECTED CHANGES", show_header=True, header_style="bold magenta"
    )
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Action", style="green", no_wrap=True)
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Examples", style="dim")

    has_changes = False
    for obs_type, actions in changes.items():
        for action, items in actions.items():
            if items:
                has_changes = True
                examples = ", ".join(items[:2])  # Show first 2 examples
                if len(items) > 2:
                    examples += f", ... (+{len(items) - 2} more)"

                table.add_row(
                    obs_type.title(),
                    action.title(),
                    str(len(items)),
                    examples[:60] + "..." if len(examples) > 60 else examples,
                )

    if has_changes:
        console.print(table)

        # Add significance indicator
        significance_text = Text()
        if significance_score >= threshold:
            significance_text.append(
                "✅ Significant changes detected ", style="bold green"
            )
        else:
            significance_text.append("❌ Changes below threshold ", style="bold red")
        significance_text.append(f"({significance_score:.1%} vs {threshold:.1%})")

        console.print(significance_text)
    else:
        console.print("[dim]No changes detected[/]")


def log_error_panel(
    logger: logging.Logger,
    error: Exception,
    context: str | None = None,
) -> None:
    """
    Log error in a prominent panel.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about the error
    """
    error_text = f"[bold red]{error.__class__.__name__}[/]: {str(error)}"
    if context:
        error_text = f"{context}\n\n{error_text}"

    panel = Panel(
        error_text,
        title="🚨 ERROR",
        title_align="left",
        border_style="red",
        padding=(1, 2),
    )

    console.print(panel)
    logger.error(f"{error.__class__.__name__}: {str(error)}", exc_info=error)


def log_performance_metrics(
    metrics: dict[str, Any],
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


def _extract_observation_text(obs: Any) -> str:
    """Extract text content from various observation types, including premises."""
    if isinstance(obs, str):
        return obs
    elif hasattr(obs, "conclusion"):
        # Handle structured observations with premises
        conclusion = obs.conclusion
        if hasattr(obs, "premises") and obs.premises:
            premises_text = "\n" + "\n".join(f"    - {p}" for p in obs.premises)
            return f"{conclusion}{premises_text}"
        return conclusion
    elif hasattr(obs, "content"):
        return obs.content
    elif isinstance(obs, dict):
        # Handle dict-based structured observations
        if "conclusion" in obs:
            conclusion = obs["conclusion"]
            premises = obs.get("premises", [])
            if premises:
                premises_text = "\n" + "\n".join(f"    - {p}" for p in premises)
                return f"{conclusion}{premises_text}"
            return conclusion
        return obs.get("content", str(obs))
    else:
        return str(obs)
