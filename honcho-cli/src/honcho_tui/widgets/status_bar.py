"""Bottom status bar: workspace, peer, queue summary, uptime."""

from __future__ import annotations

from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static
from textual.reactive import reactive

from honcho_tui.theme import ACCENT, FG_DIM, FG_MUTED, GREEN, WARN, DOT


class StatusBar(Widget):
    """Fixed bottom bar showing workspace, peer, queue, and uptime."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        layout: horizontal;
        background: #141820;
        color: #3d6480;
        padding: 0 1;
    }
    """

    workspace_id: reactive[str] = reactive("")
    peer_id: reactive[str] = reactive("")
    session_id: reactive[str] = reactive("")
    pending: reactive[int] = reactive(0)
    running: reactive[int] = reactive(0)
    uptime: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("", id="status-content")

    def on_mount(self) -> None:
        self._render_status()

    def watch_workspace_id(self, value: str) -> None:
        self._render_status()

    def watch_peer_id(self, value: str) -> None:
        self._render_status()

    def watch_session_id(self, value: str) -> None:
        self._render_status()

    def watch_pending(self, value: int) -> None:
        self._render_status()

    def watch_running(self, value: int) -> None:
        self._render_status()

    def watch_uptime(self, value: int) -> None:
        self._render_status()

    def _render_status(self) -> None:
        try:
            content = self.query_one("#status-content", Static)
        except Exception:
            return

        parts: list[str] = []

        if self.workspace_id:
            ws = self.workspace_id[:16] + "…" if len(self.workspace_id) > 16 else self.workspace_id
            parts.append(f"[{ACCENT}]ws[/{ACCENT}] {ws}")
        else:
            parts.append(f"[{FG_MUTED}]no workspace[/{FG_MUTED}]")

        if self.peer_id:
            p = self.peer_id[:14] + "…" if len(self.peer_id) > 14 else self.peer_id
            parts.append(f"[{FG_DIM}]peer[/{FG_DIM}] {p}")

        if self.session_id:
            s = self.session_id[:12] + "…" if len(self.session_id) > 12 else self.session_id
            parts.append(f"[{FG_DIM}]sess[/{FG_DIM}] {s}")

        # Queue
        if self.running > 0:
            parts.append(f"[{GREEN}]{DOT}[/{GREEN}] [{FG_DIM}]{self.running} running[/{FG_DIM}]")
        if self.pending > 0:
            parts.append(f"[{WARN}]{DOT}[/{WARN}] [{FG_DIM}]{self.pending} pending[/{FG_DIM}]")

        # Uptime (right-aligned via spacer)
        h = self.uptime // 3600
        m = (self.uptime % 3600) // 60
        s = self.uptime % 60
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

        sep = f"  [{FG_MUTED}]·[/{FG_MUTED}]  "
        left = sep.join(parts)
        content.update(f"{left}  [{FG_MUTED}]{uptime_str}[/{FG_MUTED}]")
