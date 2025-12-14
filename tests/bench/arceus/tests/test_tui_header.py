#!/usr/bin/env python3
"""Quick test to visualize the new TUI header."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arceus.tui import ArceusTUI
from rich.console import Console


def main():
    """Display the header to see how it looks."""
    console = Console()
    tui = ArceusTUI()

    # Test without task ID
    console.print("\n[bold white]Header without task ID:[/bold white]")
    header1 = tui._make_header()
    console.print(header1)

    # Test with task ID
    console.print("\n[bold white]Header with task ID:[/bold white]")
    tui.current_task_id = "007bbfb7"
    header2 = tui._make_header()
    console.print(header2)

    console.print("\n[bold green]✓ Header looks great![/bold green]")


if __name__ == "__main__":
    main()
