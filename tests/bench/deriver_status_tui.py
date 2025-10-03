"""
Honcho Deriver Status TUI

A Textual-based terminal user interface that displays real-time status of the
Honcho deriver work queue across all workspaces. Polls get_deriver_status()
once per second for each workspace and aggregates the results.
"""

from datetime import datetime
from typing import ClassVar

from honcho_core import AsyncHoncho as AsyncHonchoCore
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, ProgressBar, Static


class StatusDisplay(Static):
    """Widget to display deriver queue status."""

    def update_status(
        self,
        pending: int,
        in_progress: int,
        completed: int,
        workspace_count: int,
        timestamp: datetime | None = None,
    ) -> None:
        """Update the status display with new values."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        status = f"""
[bold cyan]Honcho Queue Status[/bold cyan]
[yellow]Last Updated:[/yellow] {time_str}
[yellow]Workspaces:[/yellow]              {workspace_count:>6}

[yellow]Pending:[/yellow]      {pending:>6}
[yellow]In Progress:[/yellow]  {in_progress:>6}
[green]Completed:[/green]    {completed:>6}
"""
        self.update(status)


class DeriverStatusApp(App[None]):
    """A Textual app to display Honcho deriver status."""

    CSS: ClassVar[str] = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        content-align: center middle;
        padding: 2;
    }

    StatusDisplay {
        margin-bottom: 2;
    }

    ProgressBar {
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [  # pyright: ignore
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    def __init__(self, honcho_url: str = "http://localhost:8000"):
        super().__init__()
        self.honcho_url: str = honcho_url
        self.honcho_client: AsyncHonchoCore = AsyncHonchoCore(base_url=honcho_url)
        self.status_display: StatusDisplay | None = None
        self.progress_bar: ProgressBar | None = None
        self.completed_count: int = 0

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(
            Vertical(
                StatusDisplay("Loading..."),
                ProgressBar(total=1, show_eta=False),
                id="main-container",
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start the status polling when the app is mounted."""
        self.status_display = self.query_one(StatusDisplay)
        self.progress_bar = self.query_one(ProgressBar)
        self.set_interval(1.0, self.poll_status)

    async def poll_status(self) -> None:
        """Poll the deriver status for all workspaces and aggregate results."""
        try:
            # List all workspaces
            workspaces_page = await self.honcho_client.workspaces.list()
            workspaces = list(workspaces_page.items)

            # Aggregate stats across all workspaces
            total_pending = 0
            total_in_progress = 0

            for workspace in workspaces:
                try:
                    status = await self.honcho_client.workspaces.deriver_status(
                        workspace_id=workspace.id
                    )
                    total_pending += status.pending_work_units
                    total_in_progress += status.in_progress_work_units
                except Exception:
                    # Skip workspace if we can't get its status
                    continue

            active = total_pending + total_in_progress

            # Track completed work units (increment when total active decreases)
            if self.progress_bar and self.progress_bar.total is not None:
                prev_active = int(self.progress_bar.total - self.progress_bar.progress)
                if prev_active > active and prev_active > 0:
                    self.completed_count += prev_active - active

            total = active + self.completed_count

            if self.status_display:
                self.status_display.update_status(
                    pending=total_pending,
                    in_progress=total_in_progress,
                    completed=self.completed_count,
                    workspace_count=len(workspaces),
                    timestamp=datetime.now(),
                )

            if self.progress_bar:
                # Progress bar shows: completed out of total all-time work units
                # With 0 completed + 5 active = 0% (0/5)
                # With 3 completed + 2 active = 60% (3/5)
                # With 5 completed + 0 active = 100% (5/5)
                if total > 0:
                    self.progress_bar.update(total=total, progress=self.completed_count)
                else:
                    # No work units - show empty progress bar
                    self.progress_bar.update(total=1, progress=0)

        except Exception as e:
            if self.status_display:
                self.status_display.update(
                    f"[bold red]Error fetching status:[/bold red]\n{str(e)}"
                )

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = (  # pyright: ignore
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


def main() -> None:
    """Run the TUI application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Display Honcho deriver work queue status across all workspaces in a TUI"
    )
    parser.add_argument(
        "--honcho-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the Honcho instance (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    app = DeriverStatusApp(honcho_url=args.honcho_url)
    app.run()


if __name__ == "__main__":
    main()
