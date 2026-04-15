"""Bottom status bar: workspace, peer, queue summary, uptime, pulsing dot."""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from honcho_tui.theme import ACCENT, DOT, DOT_EMPTY, FG_DIM, FG_MUTED, GREEN, WARN


class StatusBar(Static):
    """Fixed bottom bar — rendered as a single Rich Text line."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: #242424;
        color: #555555;
        padding: 0 2;
    }
    """

    workspace_id: reactive[str] = reactive("")
    peer_id: reactive[str] = reactive("")
    session_id: reactive[str] = reactive("")
    pending: reactive[int] = reactive(0)
    running: reactive[int] = reactive(0)
    uptime: reactive[int] = reactive(0)

    _pulse_on: bool = True

    def on_mount(self) -> None:
        self.set_interval(0.7, self._pulse)
        self._update_content()

    def _pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self._update_content()

    def watch_workspace_id(self, _: str) -> None:
        self._update_content()

    def watch_peer_id(self, _: str) -> None:
        self._update_content()

    def watch_session_id(self, _: str) -> None:
        self._update_content()

    def watch_pending(self, _: int) -> None:
        self._update_content()

    def watch_running(self, _: int) -> None:
        self._update_content()

    def watch_uptime(self, _: int) -> None:
        self._update_content()

    def _update_content(self) -> None:
        t = Text(no_wrap=True, overflow="ellipsis")

        # Pulsing connection dot
        conn_style = ACCENT if self._pulse_on else FG_MUTED
        t.append(f"{DOT} ", style=conn_style)

        if self.workspace_id:
            ws = self.workspace_id[:14] + "…" if len(self.workspace_id) > 14 else self.workspace_id
            t.append("ws ", style=ACCENT)
            t.append(f"{ws}  ", style=FG_DIM)
        else:
            t.append("no workspace  ", style=FG_MUTED)

        if self.peer_id:
            p = self.peer_id[:12] + "…" if len(self.peer_id) > 12 else self.peer_id
            t.append("peer ", style=FG_DIM)
            t.append(f"{p}  ", style=FG_DIM)

        if self.session_id:
            s = self.session_id[:12] + "…" if len(self.session_id) > 12 else self.session_id
            t.append("sess ", style=FG_DIM)
            t.append(f"{s}  ", style=FG_DIM)

        if self.running > 0:
            t.append(f"{DOT} ", style=GREEN)
            t.append(f"{self.running} running  ", style=FG_DIM)
        if self.pending > 0:
            t.append(f"{DOT} ", style=WARN)
            t.append(f"{self.pending} pending  ", style=FG_DIM)

        total = int(self.uptime)
        h = total // 3600
        m = (total % 3600) // 60
        s_val = total % 60
        t.append(f"{h:02d}:{m:02d}:{s_val:02d}", style=FG_MUTED)

        self.update(t)
