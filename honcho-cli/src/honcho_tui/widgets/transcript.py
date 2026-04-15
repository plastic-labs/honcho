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
    BLUE,
    FG,
    FG_DIM,
    FG_MUTED,
    FRAMES_HELIX,
    GREEN,
    RUST,
)
from honcho_tui.widgets.animations import WaveformWidget


class QuerySubmitted(Message):
    """Posted when the user submits a dialectic query."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__()


class TranscriptPanel(Widget):
    """Center pane: waveform header + message log + query input."""

    DEFAULT_CSS = """
    TranscriptPanel {
        layout: vertical;
        width: 100%;
        height: 100%;
    }
    """

    _frame_idx: int = 0
    _streaming: bool = False

    def compose(self) -> ComposeResult:
        yield WaveformWidget(id="transcript-wave")
        yield RichLog(highlight=False, markup=False, wrap=True, id="log")
        with Widget(id="query-bar"):
            yield Static(f"[{ACCENT}]>[/{ACCENT}]", id="query-prefix", markup=True)
            yield Input(placeholder="query dialectic…", id="query-input")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.08, self._tick)
        self.show_welcome()

    def _tick(self) -> None:
        if self._streaming:
            self._frame_idx = (self._frame_idx + 1) % len(FRAMES_HELIX)

    # ── Public API ────────────────────────────────────────────────────────────

    def show_welcome(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        t = Text()
        t.append("\n  HONCHO\n", style=f"bold {ACCENT}")
        t.append("  memory that reasons\n\n", style=FG_MUTED)
        t.append("  select a session from the left panel\n", style=FG_DIM)
        t.append("  or query the dialectic below\n\n", style=FG_DIM)
        t.append("  bindings\n", style=f"bold {FG_DIM}")
        for key, desc in [
            ("ctrl+d", "focus dialectic input"),
            ("tab",    "focus session list"),
            ("r",      "refresh"),
            ("q",      "quit"),
        ]:
            t.append(f"    {key}", style=f"bold {ACCENT}")
            t.append(f"  {desc}\n", style=FG_MUTED)
        log.write(t)

    def show_no_workspace(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        t = Text()
        t.append("\n  no workspace configured\n\n", style=f"bold {RUST}")
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

        short_sid = session_id[:32] + "…" if len(session_id) > 32 else session_id
        header = Text()
        header.append(f"\n  {short_sid}\n", style=f"bold {FG}")
        header.append(f"  {'─' * 40}\n\n", style=FG_MUTED)
        log.write(header)

        if not messages:
            log.write(Text("  no messages\n", style=FG_DIM))
            return

        for msg in messages:
            self._write_message(log, msg)

    def _write_message(self, log: RichLog, msg: dict) -> None:
        peer_id = msg.get("peer_id", "unknown")
        content = msg.get("content", "")
        created_at = msg.get("created_at")

        ts = ""
        if created_at:
            try:
                dt = datetime.datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                ts = dt.strftime("%H:%M")
            except Exception:
                ts = str(created_at)[:5]

        header = Text()
        header.append(f"  {ts}  ", style=FG_MUTED)
        header.append(peer_id, style=f"bold {BLUE}")
        header.append("\n")
        log.write(header)

        body = Text()
        for line in content.splitlines():
            body.append("    ", style=FG)
            body.append(line, style=FG)
            body.append("\n")
        if not content.strip():
            body.append("    (empty)\n", style=FG_MUTED)
        log.write(body)
        log.write(Text("\n"))

    def append_response(self, label: str, content: str, is_query: bool = False) -> None:
        log = self.query_one("#log", RichLog)
        self._streaming = False

        if is_query:
            q = Text()
            q.append(f"  > ", style=f"bold {ACCENT}")
            q.append(f"{label}\n\n", style=f"bold {FG}")
            log.write(q)

        resp = Text()
        resp.append("  dialectic\n", style=f"bold {GREEN}")
        for line in content.splitlines():
            resp.append("    ")
            resp.append(line, style=FG)
            resp.append("\n")
        resp.append("\n")
        log.write(resp)

    def show_error(self, msg: str) -> None:
        self._streaming = False
        log = self.query_one("#log", RichLog)
        t = Text()
        t.append(f"\n  error\n", style="bold red")
        t.append(f"  {msg}\n", style=FG_DIM)
        log.write(t)

    def focus_input(self) -> None:
        self.query_one("#query-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.clear()
        self.post_message(QuerySubmitted(query))
