"""Terminal User Interface for Arceus solver visualization."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .metrics import SolverMetrics


class ArceusTUI:
    """Terminal User Interface for visualizing the ARC solver."""

    # ARC color palette (0-9)
    ARC_COLORS = {
        0: "black",
        1: "blue",
        2: "red",
        3: "green",
        4: "yellow",
        5: "grey70",
        6: "magenta",
        7: "orange1",
        8: "cyan",
        9: "brown",
    }

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.current_task_id = ""
        self.current_input_grid: List[List[int]] = []
        self.current_output_grid: Optional[List[List[int]]] = None
        self.expected_output_grid: Optional[List[List[int]]] = None
        self.current_attempt_grid: Optional[List[List[int]]] = None
        self.current_transformation: str = ""
        self.iteration_number: int = 0
        self.agent_logs: List[Dict[str, Any]] = []
        self.memory_operations: List[Dict[str, Any]] = []
        self.metrics: Optional[SolverMetrics] = None
        self.max_logs = 20
        self.max_memory_ops = 15

        # Training examples for context
        self.training_examples: List[Dict] = []  # Store training examples
        self.show_correct_solution: bool = False  # Show correct answer when wrong

        self._setup_layout()

    def _setup_layout(self):
        """Setup the TUI layout with 4 panels."""
        self.layout.split_column(
            Layout(name="header", size=7),  # Reduced header size
            Layout(name="main", ratio=1),
            Layout(name="footer", size=8),
        )

        # Give more space to puzzle visualization (left panel)
        self.layout["main"].split_row(
            Layout(name="left", ratio=3), Layout(name="right", ratio=2)
        )

        self.layout["right"].split_column(
            Layout(name="agent_logs", ratio=3), Layout(name="memory_viz", ratio=2)
        )

    def _make_header(self) -> Panel:
        """Create the header with ASCII art title."""
        title = Text()

        # Main ARCEUS logo with smooth gradient - original readable font
        # Line 1 - Smooth gradient: cyan → blue → magenta → red
        title.append("            ,6\"Yb.  ", style="bold cyan")
        title.append("`7Mb,od8 ", style="bold bright_cyan")
        title.append(",p6\"bo   ", style="bold blue")
        title.append(".gP\"Ya ", style="bold bright_magenta")
        title.append("`7MM  ", style="bold magenta")
        title.append("`7MM  ", style="bold red")
        title.append(",pP\"Ybd\n", style="bold bright_red")

        # Line 2 - Continuing smooth gradient flow
        title.append("           8)   MM", style="bold cyan")
        title.append("    MM' \"'", style="bold bright_cyan")
        title.append("6M'  OO", style="bold blue")
        title.append("  ,M'   Yb", style="bold bright_magenta")
        title.append("  MM", style="bold magenta")
        title.append("    MM", style="bold red")
        title.append("  8I   `\"\n", style="bold bright_red")

        # Line 3 - Gradient wave continues
        title.append("            ,pm9MM", style="bold cyan")
        title.append("    MM", style="bold bright_cyan")
        title.append("    8M", style="bold blue")
        title.append("       8M\"\"\"\"\"\"", style="bold bright_magenta")
        title.append("  MM", style="bold magenta")
        title.append("    MM", style="bold red")
        title.append("  `YMMMa.\n", style="bold bright_red")

        # Line 4 - Gradient flowing
        title.append("           8M   MM", style="bold cyan")
        title.append("    MM", style="bold bright_cyan")
        title.append("    YM.", style="bold blue")
        title.append("    , YM.", style="bold bright_magenta")
        title.append("    ,  MM", style="bold magenta")
        title.append("    MM", style="bold red")
        title.append("  L.   I8\n", style="bold bright_red")

        # Line 5 - Final gradient sweep
        title.append("           `Moo9^Yo.", style="bold cyan")
        title.append(".JMML.", style="bold bright_cyan")
        title.append("   YMbmd'", style="bold blue")
        title.append("   `Mbmmd'", style="bold bright_magenta")
        title.append("  `Mbod\"YML.", style="bold magenta")
        title.append("M9mmmP'", style="bold bright_red")

        return Panel(
            Align.center(title),
            box=DOUBLE,
            style="bold magenta",
            border_style="bold bright_magenta"
        )

    def _render_grid(self, grid: List[List[int]], title: str = "Grid") -> Panel:
        """Render a grid with colors, adapting size based on grid dimensions."""
        if not grid:
            return Panel("No grid data", title=title, expand=False)

        # Determine cell size based on grid dimensions
        rows = len(grid)
        cols = len(grid[0]) if grid else 0

        # Use smaller cells for larger grids
        if rows > 15 or cols > 15:
            cell_char = "▪"
            cell_width = 1
        elif rows > 10 or cols > 10:
            cell_char = "■"
            cell_width = 1
        else:
            cell_char = "█"
            cell_width = 2

        grid_table = Table.grid(padding=0)
        for _ in range(cols):
            grid_table.add_column(justify="center", width=cell_width)

        for row in grid:
            styled_cells = []
            for cell in row:
                # Handle case where cell might be a list (malformed grid structure)
                if isinstance(cell, list):
                    # Use a placeholder for malformed cells
                    styled_cells.append(Text("?", style="bold red"))
                    continue
                color = self.ARC_COLORS.get(cell, "white")
                styled_cells.append(Text(cell_char, style=f"bold {color}"))
            grid_table.add_row(*styled_cells)

        return Panel(
            Align.center(grid_table),
            title=f"[bold]{title}[/bold]",
            border_style="green",
            box=ROUNDED,
            expand=False,
            padding=(0, 1)
        )

    def _render_mini_grid(self, grid: List[List[int]], title: str = "Grid") -> RenderableType:
        """Render a smaller grid for training examples (compact visualization without panel)."""
        if not grid:
            return Text(f"{title}: No data", style="dim")

        # Create compact grid without panel overhead
        grid_lines = []
        grid_lines.append(Text(f"{title}:", style="dim cyan"))

        for row in grid:
            row_text = Text()
            for cell in row:
                if isinstance(cell, list):
                    row_text.append("?", style="bold red")
                    continue
                color = self.ARC_COLORS.get(cell, "white")
                # Use smaller block character for compact display
                row_text.append("▪", style=f"bold {color}")
            grid_lines.append(row_text)

        return Group(*grid_lines)

    def _make_puzzle_panel(self) -> RenderableType:
        """Create the puzzle visualization panel."""
        if not self.current_input_grid:
            return Panel(
                Align.center("Waiting for task..."),
                title="[bold]Puzzle Visualization[/bold]",
                border_style="green",
            )

        components = []

        # Show training examples if available (with VERY clear distinction)
        if self.training_examples:
            # HEADER: Make it super clear these are EXAMPLES to learn from
            components.append(Text(""))
            components.append(Panel(
                Text("📚 TRAINING EXAMPLES - Learn the Pattern from These",
                     style="bold bright_cyan", justify="center"),
                border_style="bright_cyan",
                box=DOUBLE,
                padding=(0, 2)
            ))
            components.append(Text(""))

            # Create a single compact table with all examples side-by-side
            examples_table = Table.grid(padding=(0, 1))

            # Build columns: Ex1_In, →, Ex1_Out, |, Ex2_In, →, Ex2_Out
            for _ in range(len(self.training_examples) * 3 - 1):  # 3 cols per example, minus 1 for last separator
                examples_table.add_column(justify="center")

            try:
                row_items = []
                for idx, example in enumerate(self.training_examples, 1):
                    # Ensure example is a dictionary
                    if not isinstance(example, dict):
                        continue
                    example_input = example.get("input", [])
                    example_output = example.get("output", [])

                    # Create mini grids for training examples (very compact)
                    input_viz = self._render_mini_grid(example_input, f"In")
                    output_viz = self._render_mini_grid(example_output, f"Out")

                    row_items.append(input_viz)
                    row_items.append(Text("→", style="bold bright_magenta"))
                    row_items.append(output_viz)

                    # Add separator between examples (not after last one)
                    if idx < len(self.training_examples):
                        row_items.append(Text(" │ ", style="dim cyan"))

                if row_items:
                    examples_table.add_row(*row_items)
                    components.append(examples_table)
            except Exception as e:
                # Skip malformed examples
                pass

            components.append(Text(""))  # Just a spacer

            # HEADER: Make it super clear THIS is what needs to be solved
            components.append(Panel(
                Text("🎯 TEST PUZZLE - Apply Pattern Here to Solve",
                     style="bold bright_yellow", justify="center"),
                border_style="bright_yellow",
                box=DOUBLE,
                padding=(0, 2)
            ))
            components.append(Text(""))

        # Create side-by-side layout for test input and iteration result
        side_by_side_table = Table.grid(padding=(0, 2))
        side_by_side_table.add_column(justify="left", ratio=1)  # Test Input column
        side_by_side_table.add_column(justify="center", width=5)  # Arrow column
        side_by_side_table.add_column(justify="left", ratio=1)  # Output/Attempt column

        # Left side: Test Input
        input_panel = self._render_grid(self.current_input_grid, "📥 Test Input (Given)")

        # Middle: Arrow and iteration info
        middle_content = Text()
        if self.current_transformation:
            middle_content.append(f"Iter {self.iteration_number}\n", style="bold cyan")
            middle_content.append("→\n", style="bold bright_magenta")
            middle_content.append(f"{self.current_transformation[:20]}", style="dim yellow")
        else:
            middle_content.append("→", style="bold bright_magenta")

        # Right side: Attempt result or status
        right_content = []

        if self.current_attempt_grid:
            attempt_panel = self._render_grid(self.current_attempt_grid, "📤 Agent's Output")
            right_content.append(attempt_panel)

            # Show comparison status below the grid
            if self.expected_output_grid:
                match = self._grids_match(self.current_attempt_grid, self.expected_output_grid)
                status_text = Text()
                if match:
                    status_text.append("✓ MATCH! ", style="bold green")
                    status_text.append("Solved correctly!", style="bold bright_green")
                else:
                    status_text.append("✗ INCORRECT", style="bold red")
                right_content.append(Text(""))
                right_content.append(status_text)
        elif self.current_output_grid:
            output_panel = self._render_grid(self.current_output_grid, "✅ Final Solution")
            right_content.append(output_panel)
            right_content.append(Text(""))
            right_content.append(Text("✓ SOLVED", style="bold green"))
        else:
            right_content.append(Text("⏳ Analyzing pattern...", style="bold yellow"))

        # Assemble the side-by-side layout
        right_group = Group(*right_content) if right_content else Text("...")
        side_by_side_table.add_row(input_panel, middle_content, right_group)

        components.append(side_by_side_table)

        # Show expected output below if incorrect and enabled
        if self.current_attempt_grid and self.expected_output_grid and self.show_correct_solution:
            if not self._grids_match(self.current_attempt_grid, self.expected_output_grid):
                components.append(Text(""))
                components.append(Panel(
                    Text("✓ CORRECT SOLUTION (What it should be)",
                         style="bold bright_green", justify="center"),
                    border_style="bright_green",
                    box=ROUNDED,
                    padding=(0, 1)
                ))
                correct_panel = self._render_grid(self.expected_output_grid, "📋 Expected Output")
                components.append(correct_panel)

        grid_group = Group(*components)

        title_text = f"[bold]Puzzle Visualization[/bold]"
        if self.iteration_number > 0:
            title_text = f"[bold]Puzzle Visualization[/bold] [dim]- Iter {self.iteration_number}[/dim]"

        return Panel(
            grid_group,
            title=title_text,
            border_style="green",
            expand=True,  # Use all available space
            padding=(0, 1)  # Reduce padding
        )

    def _grids_match(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids match."""
        if not grid1 or not grid2:
            return False
        if len(grid1) != len(grid2):
            return False
        if len(grid1[0]) != len(grid2[0]):
            return False
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True

    def _validate_grid(self, grid: Any, grid_name: str = "grid") -> bool:
        """Validate that a grid has the proper structure (list of lists of integers)."""
        # Check if grid is a list
        if not isinstance(grid, list):
            self.add_agent_log("warning", f"Invalid {grid_name}: not a list (type: {type(grid).__name__})")
            return False

        # Check if grid is not empty
        if len(grid) == 0:
            self.add_agent_log("warning", f"Invalid {grid_name}: empty grid")
            return False

        # Check if all rows are lists
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                self.add_agent_log("warning", f"Invalid {grid_name}: row {i} is not a list (type: {type(row).__name__})")
                return False

            # Check if all cells are integers
            for j, cell in enumerate(row):
                if isinstance(cell, list):
                    self.add_agent_log("warning", f"Invalid {grid_name}: cell [{i}][{j}] is a list instead of integer")
                    return False
                if not isinstance(cell, (int, float)):
                    self.add_agent_log("warning", f"Invalid {grid_name}: cell [{i}][{j}] is not numeric (type: {type(cell).__name__})")
                    return False

        return True

    def _make_agent_logs_panel(self) -> Panel:
        """Create the agent reasoning/logs panel."""
        if not self.agent_logs:
            return Panel(
                "Waiting for agent activity...",
                title="[bold]Agent Reasoning & Logs[/bold]",
                border_style="blue",
            )

        log_table = Table(show_header=False, box=None, padding=(0, 1))
        log_table.add_column("Time", style="dim", width=8)
        log_table.add_column("Event", style="cyan", width=12)
        log_table.add_column("Content", style="white", overflow="fold")

        for log in self.agent_logs[-self.max_logs :]:
            timestamp = log.get("timestamp", "")
            if timestamp:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
            else:
                time_str = "--:--:--"

            event_type = log.get("event_type", "unknown")
            content = log.get("content", "")

            # Style based on event type
            event_style = "cyan"
            if event_type == "error":
                event_style = "red"
            elif event_type == "success":
                event_style = "green"
            elif event_type == "hypothesis":
                event_style = "yellow"

            # Truncate event type if too long
            event_display = event_type[:10] if len(event_type) > 10 else event_type
            log_table.add_row(time_str, Text(event_display, style=event_style), content[:100])

        return Panel(
            log_table,
            title="[bold]Agent Logs[/bold]",
            border_style="blue",
            expand=True,
            padding=(0, 1)
        )

    def _make_memory_viz_panel(self) -> Panel:
        """Create the memory visualization panel with statistics."""
        # Add memory statistics header with per-peer breakdown
        components = []

        if self.metrics:
            stats_text = Text()
            stats_text.append(f"📊 Sessions: {self.metrics.num_sessions_created} | ", style="cyan")
            stats_text.append(f"Messages: {self.metrics.num_messages_ingested}", style="green")

            # Per-peer fact counts
            if self.metrics.facts_per_peer:
                stats_text.append("\n💾 Facts per peer: ", style="yellow")
                peer_items = list(self.metrics.facts_per_peer.items())
                for i, (peer_name, count) in enumerate(peer_items):
                    # Shorten peer names for display
                    short_name = peer_name.replace("_peer", "").replace("_", " ").title()
                    stats_text.append(f"{short_name[:12]}: {count}", style="magenta")
                    if i < len(peer_items) - 1:
                        stats_text.append("  ", style="")

            components.append(stats_text)
            components.append(Text(""))  # Spacer

        if not self.memory_operations:
            if not components:
                return Panel(
                    "No memory operations yet...",
                    title="[bold]Memory[/bold]",
                    border_style="magenta",
                )
        else:
            mem_table = Table(show_header=True, box=None, padding=(0, 1))
            mem_table.add_column("Operation", style="magenta", width=18)
            mem_table.add_column("Details", style="white", overflow="fold")
            mem_table.add_column("Results", style="green", width=8)

            for op in self.memory_operations[-self.max_memory_ops :]:
                op_type = op.get("operation", "unknown")[:16]  # Truncate operation name
                details = op.get("details", "")
                results = str(op.get("num_results", 0))

                mem_table.add_row(op_type, details[:60], results)

            components.append(mem_table)

        content = Group(*components) if components else "No memory data available"

        return Panel(
            content,
            title="[bold]Memory[/bold]",
            border_style="magenta",
            expand=True,
            padding=(0, 1)
        )

    def _make_metrics_panel(self) -> Panel:
        """Create the comprehensive metrics display panel."""
        if not self.metrics:
            return Panel(
                "No metrics available",
                title="[bold]Key Metrics[/bold]",
                border_style="yellow",
            )

        # Create two-column layout for more comprehensive metrics
        metrics_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        metrics_table.add_column("Metric", style="yellow bold", width=16)
        metrics_table.add_column("Value", style="white", justify="right", width=14)
        metrics_table.add_column("Metric2", style="yellow bold", width=16)
        metrics_table.add_column("Value2", style="white", justify="right")

        # Row 1: Time & LLM Calls
        metrics_table.add_row(
            "Time", f"{self.metrics.get_elapsed_time():.1f}s",
            "LLM Calls", str(self.metrics.num_llm_calls)
        )

        # Row 2: Iterations & Tokens
        metrics_table.add_row(
            "Iterations", str(self.metrics.num_iterations),
            "Tokens", f"{self.metrics.total_tokens:,}"
        )

        # Row 3: API Cost & Cost per Token
        cost_per_token = f"${self.metrics.get_cost_per_token():.6f}" if self.metrics.total_tokens > 0 else "$0"
        metrics_table.add_row(
            "API Cost", f"${self.metrics.api_cost:.4f}",
            "Cost/Token", cost_per_token
        )

        # Row 4: Memory Queries & Messages Stored
        metrics_table.add_row(
            "Memory Queries", str(self.metrics.num_memory_queries),
            "Messages Stored", str(self.metrics.num_messages_ingested)
        )

        # Row 5: Hypotheses & Sessions
        metrics_table.add_row(
            "Hypotheses", str(self.metrics.num_hypotheses_generated),
            "Sessions", str(self.metrics.num_sessions_created)
        )

        # Row 6: Verifications & Facts
        metrics_table.add_row(
            "Verifications", f"{self.metrics.num_verifications} ({self.metrics.get_verification_success_rate():.0f}%)",
            "Facts Stored", str(self.metrics.num_facts_stored)
        )

        # Add status indicator
        if self.metrics.solved:
            status = Text("✓ SOLVED", style="bold green")
        elif self.metrics.end_time:
            status = Text("✗ UNSOLVED", style="bold red")
        else:
            status = Text("⏳ SOLVING", style="bold yellow")

        content = Group(metrics_table, Text(""), Align.center(status))

        return Panel(
            content,
            title="[bold]Metrics[/bold]",
            border_style="yellow",
            height=8,
            padding=(0, 1)
        )

    def render(self) -> Layout:
        """Render the complete TUI layout."""
        self.layout["header"].update(self._make_header())
        self.layout["left"].update(self._make_puzzle_panel())
        self.layout["agent_logs"].update(self._make_agent_logs_panel())
        self.layout["memory_viz"].update(self._make_memory_viz_panel())
        self.layout["footer"].update(self._make_metrics_panel())

        return self.layout

    def update_task(self, task_id: str, input_grid: List[List[int]], expected_output: Optional[List[List[int]]] = None, training_examples: Optional[List[Dict]] = None):
        """Update the current task being solved with optional training examples."""
        self.current_task_id = task_id

        # Validate input grid structure
        if self._validate_grid(input_grid, "input"):
            self.current_input_grid = input_grid
        else:
            self.current_input_grid = []

        # Validate expected output grid structure
        if expected_output is not None and self._validate_grid(expected_output, "expected_output"):
            self.expected_output_grid = expected_output
        else:
            self.expected_output_grid = expected_output if expected_output is None else []

        # Store training examples (first 2 for display)
        if training_examples:
            self.training_examples = training_examples[:2]
        else:
            self.training_examples = []

        self.current_output_grid = None
        self.current_attempt_grid = None
        self.current_transformation = ""
        self.iteration_number = 0
        self.show_correct_solution = False

    def update_transformation_attempt(
        self,
        iteration: int,
        transformation: str,
        result_grid: Optional[List[List[int]]] = None
    ):
        """Update the current transformation being attempted."""
        self.iteration_number = iteration
        self.current_transformation = transformation

        # Validate grid structure before setting
        if result_grid is not None:
            if self._validate_grid(result_grid, f"transformation result ({transformation})"):
                self.current_attempt_grid = result_grid
            else:
                self.current_attempt_grid = None
        else:
            self.current_attempt_grid = None

    def clear_attempt(self):
        """Clear the current attempt (for next iteration)."""
        self.current_attempt_grid = None
        self.current_transformation = ""

    def update_output(self, output_grid: List[List[int]]):
        """Update the output grid (final solution)."""
        if self._validate_grid(output_grid, "output"):
            self.current_output_grid = output_grid
        else:
            self.current_output_grid = []
        self.current_attempt_grid = None
        self.current_transformation = ""

    def mark_failed_and_show_solution(self):
        """Mark that the agent failed to solve the puzzle and show the correct solution."""
        self.show_correct_solution = True

    def add_agent_log(
        self, event_type: str, content: str, timestamp: Optional[str] = None
    ):
        """Add a log entry from the agent."""
        self.agent_logs.append(
            {
                "timestamp": timestamp or datetime.now().isoformat(),
                "event_type": event_type,
                "content": content,
            }
        )

    def add_memory_operation(self, operation: str, details: str, num_results: int = 0):
        """Add a memory operation entry."""
        self.memory_operations.append(
            {"operation": operation, "details": details, "num_results": num_results}
        )

    def update_metrics(self, metrics: SolverMetrics):
        """Update the metrics display."""
        self.metrics = metrics


async def run_tui_with_solver(
    tui: ArceusTUI, solver_task: asyncio.Task, refresh_rate: float = 0.1
):
    """Run the TUI alongside the solver task."""
    with Live(tui.render(), console=tui.console, refresh_per_second=1 / refresh_rate) as live:
        while not solver_task.done():
            live.update(tui.render())
            await asyncio.sleep(refresh_rate)

        # Final update
        live.update(tui.render())
