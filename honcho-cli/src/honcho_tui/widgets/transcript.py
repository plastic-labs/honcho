"""Center panel: message transcript + dialectic query bar."""

from __future__ import annotations

import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static

from honcho_tui.theme import (
    ACCENT,
    ACCENT_DIM,
    BLUE,
    CURSOR,
    FG,
    FG_DIM,
    FG_MUTED,
    FRAMES_HELIX,
    GREEN,
    ORANGE,
)


class QuerySubmitted(Message):
    """Posted when the user submits a dialectic query."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class TranscriptPanel(Widget):
    """Center pane: message log above, query input below."""

    DEFAULT_CSS = """
    TranscriptPanel {
        layout: vertical;
        width: 100%;
        height: 100%;
    }
    """

    _frame_idx: int = 0
    _streaming: bool = False
    _cursor_on: bool = True

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=False, markup=True, wrap=True, id="log")
        with Widget(id="query-bar"):
            yield Static(f"[{ACCENT}]>[/{ACCENT}]", id="query-prefix")
            yield Input(
                placeholder="query dialectic…",
                id="query-input",
            )

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.08, self._tick)
        self._cursor_timer = self.set_interval(0.42, self._blink_cursor)
        self.show_welcome()

    # ── Timers ────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._streaming:
            self._frame_idx = (self._frame_idx + 1) % len(FRAMES_HELIX)

    def _blink_cursor(self) -> None:
        self._cursor_on = not self._cursor_on
        if self._streaming:
            self._update_stream_cursor()

    def _update_stream_cursor(self) -> None:
        log = self.query_one("#log", RichLog)
        cursor_char = CURSOR if self._cursor_on else " "
        # Cursor line is updated via the streaming append path
        _ = cursor_char  # referenced during actual stream writes

    # ── Public API ────────────────────────────────────────────────────────────

    def show_welcome(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        t = Text()
        t.append("  HONCHO\n", style=f"bold {ACCENT}")
        t.append("  memory that reasons\n\n", style=FG_MUTED)
        t.append(f"  [{FG_MUTED}]select a session or query the dialectic[/{FG_MUTED}]\n")
        log.write(t)

    def show_no_workspace(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        t = Text()
        t.append("\n  no workspace configured\n\n", style=f"bold {ORANGE}")
        t.append("  set HONCHO_WORKSPACE_ID or pass --workspace\n", style=FG_DIM)
        t.append("  run 'honcho init' to configure\n", style=FG_DIM)
        log.write(t)

    def show_loading(self, label: str = "loading…") -> None:
        self._streaming = True
        log = self.query_one("#log", RichLog)
        log.clear()
        frame = FRAMES_HELIX[self._frame_idx]
        t = Text()
        t.append(f"\n  {frame} ", style=ACCENT)
        t.append(label, style=FG_DIM)
        log.write(t)

    def clear_loading(self) -> None:
        self._streaming = False

    def load_messages(self, session_id: str, messages: list[dict]) -> None:
        """Render a session's message history."""
        self._streaming = False
        log = self.query_one("#log", RichLog)
        log.clear()

        # Header
        short_sid = session_id[:28] + "…" if len(session_id) > 28 else session_id
        header = Text()
        header.append(f"  session  ", style=FG_MUTED)
        header.append(short_sid, style=f"bold {FG}")
        header.append(f"\n  {'─' * 50}\n", style=FG_MUTED)
        log.write(header)

        if not messages:
            log.write(Text(f"  [{FG_DIM}]no messages[/{FG_DIM}]\n"))
            return

        for msg in messages:
            self._write_message(log, msg)

    def _write_message(self, log: RichLog, msg: dict) -> None:
        peer_id = msg.get("peer_id", "unknown")
        content = msg.get("content", "")
        created_at = msg.get("created_at")

        # Timestamp
        ts = ""
        if created_at:
            try:
                dt = datetime.datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                ts = dt.strftime("%H:%M")
            except Exception:
                ts = str(created_at)[:5]

        # Header line
        header = Text()
        header.append(f"  {ts} ", style=FG_MUTED)
        header.append(peer_id, style=f"bold {BLUE}")
        header.append("\n")
        log.write(header)

        # Content
        body = Text()
        for line in content.splitlines():
            body.append(f"    {line}\n", style=FG)
        if not content.strip():
            body.append("    (empty)\n", style=FG_MUTED)
        log.write(body)
        log.write(Text("\n"))

    def append_response(self, label: str, content: str, is_query: bool = False) -> None:
        """Append a query + response pair to the log."""
        log = self.query_one("#log", RichLog)
        self._streaming = False

        if is_query:
            q_text = Text()
            q_text.append(f"  > ", style=f"bold {ACCENT}")
            q_text.append(f"{label}\n\n", style=f"bold {FG}")
            log.write(q_text)

        resp = Text()
        resp.append("  dialectic\n", style=f"bold {GREEN}")
        for line in content.splitlines():
            resp.append(f"    {line}\n", style=FG)
        resp.append("\n")
        log.write(resp)

    def show_error(self, msg: str) -> None:
        self._streaming = False
        log = self.query_one("#log", RichLog)
        t = Text()
        t.append(f"\n  error  ", style="bold red")
        t.append(f"{msg}\n", style=FG_DIM)
        log.write(t)

    def focus_input(self) -> None:
        self.query_one("#query-input", Input).focus()

    # ── Events ────────────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self.post_message(QuerySubmitted(query))
