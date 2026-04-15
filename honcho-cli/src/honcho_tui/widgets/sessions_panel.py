"""Left panel: session list with keyboard navigation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import ListItem, ListView, Static
from textual.widget import Widget

from honcho_tui.theme import ACCENT, ACCENT_DIM, FG, FG_DIM, FRAMES_DNA, SEP


class SessionSelected(Message):
    """Posted when the user selects a session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__()


class _SessionItem(ListItem):
    """Single row in the session list."""

    def __init__(self, session_id: str, is_active: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session_id = session_id
        self.is_active = is_active

    def compose(self) -> ComposeResult:
        dot = f"[{ACCENT}]●[/{ACCENT}]" if self.is_active else f"[{FG_DIM}]○[/{FG_DIM}]"
        short_id = self.session_id[:18] + "…" if len(self.session_id) > 18 else self.session_id
        yield Static(f" {dot} [{FG}]{short_id}[/{FG}]")


class SessionsPanel(Widget):
    """Left-side panel showing all sessions in the workspace."""

    DEFAULT_CSS = """
    SessionsPanel {
        layout: vertical;
        width: 100%;
        height: 100%;
    }
    """

    _frame_idx: int = 0
    _loading: bool = False

    def compose(self) -> ComposeResult:
        yield Static(f" [{ACCENT}]SESSIONS[/{ACCENT}]", classes="panel-title")
        yield ListView(id="sessions-list")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

    def _tick_spinner(self) -> None:
        if self._loading:
            self._frame_idx = (self._frame_idx + 1) % len(FRAMES_DNA)

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        header = self.query_one(".panel-title", Static)
        if loading:
            frame = FRAMES_DNA[self._frame_idx]
            header.update(f" [{ACCENT}]SESSIONS[/{ACCENT}]  [{ACCENT_DIM}]{frame}[/{ACCENT_DIM}]")
        else:
            header.update(f" [{ACCENT}]SESSIONS[/{ACCENT}]")

    def load_sessions(self, sessions: list[dict]) -> None:
        """Populate the list with session data (called from main thread)."""
        self.set_loading(False)
        lv = self.query_one("#sessions-list", ListView)
        lv.clear()
        if not sessions:
            lv.append(ListItem(Static(f"  [{FG_DIM}]no sessions[/{FG_DIM}]")))
            return
        for s in sessions:
            lv.append(_SessionItem(s["id"], is_active=s.get("is_active", True)))

    def show_error(self, msg: str) -> None:
        self.set_loading(False)
        lv = self.query_one("#sessions-list", ListView)
        lv.clear()
        short = msg[:22] + "…" if len(msg) > 22 else msg
        lv.append(ListItem(Static(f"  [red]{short}[/red]")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _SessionItem):
            self.post_message(SessionSelected(item.session_id))
