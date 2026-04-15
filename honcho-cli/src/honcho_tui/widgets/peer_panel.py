"""Right panel: peer card, conclusions, queue status."""

from __future__ import annotations

from rich.markup import escape as markup_escape
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.containers import VerticalScroll

from honcho_tui.theme import (
    ACCENT,
    ACCENT_DIM,
    DOT,
    DOT_EMPTY,
    FG,
    FG_DIM,
    FG_MUTED,
    FRAMES_ORBIT,
    GREEN,
    TREE_LAST,
    TREE_MID,
    WARN,
)
from honcho_tui.widgets.animations import FreqBarsWidget


def _card_to_str(card) -> str:
    """Normalize whatever get_card() returns to a plain string."""
    if card is None:
        return ""
    if isinstance(card, str):
        return card
    if isinstance(card, list):
        return "\n".join(str(item) for item in card)
    return str(card)


class PeerPanel(Widget):
    """Right-side panel: freq bars + peer card + conclusions + queue."""

    DEFAULT_CSS = """
    PeerPanel {
        layout: vertical;
        width: 100%;
        height: 100%;
    }
    """

    _frame_idx: int = 0
    _loading: bool = False

    def compose(self) -> ComposeResult:
        yield Static(f" [{ACCENT}]PEER[/{ACCENT}]", classes="panel-title", markup=True)
        yield FreqBarsWidget(id="freq-bars")
        yield VerticalScroll(id="peer-scroll")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.12, self._tick)
        self._show_empty()

    def _tick(self) -> None:
        if self._loading:
            self._frame_idx = (self._frame_idx + 1) % len(FRAMES_ORBIT)
            header = self.query_one(".panel-title", Static)
            frame = FRAMES_ORBIT[self._frame_idx]
            header.update(f" [{ACCENT}]PEER[/{ACCENT}]  [{ACCENT_DIM}]{frame}[/{ACCENT_DIM}]")

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
        if not loading:
            header = self.query_one(".panel-title", Static)
            header.update(f" [{ACCENT}]PEER[/{ACCENT}]")

    def _show_empty(self) -> None:
        scroll = self.query_one("#peer-scroll", VerticalScroll)
        scroll.remove_children()
        scroll.mount(Static(f"\n  [{FG_DIM}]select a session[/{FG_DIM}]", markup=True))

    def load_peer(
        self,
        peer_id: str,
        card,
        conclusions: list[dict],
        queue: dict | None,
    ) -> None:
        """Populate panel with peer data (called from main thread)."""
        self.set_loading(False)

        short_pid = peer_id[:16] + "…" if len(peer_id) > 16 else peer_id
        header = self.query_one(".panel-title", Static)
        header.update(f" [{ACCENT}]PEER[/{ACCENT}]  [{FG_DIM}]{short_pid}[/{FG_DIM}]")

        scroll = self.query_one("#peer-scroll", VerticalScroll)
        scroll.remove_children()

        # ── Card ─────────────────────────────────────────────────────────────
        scroll.mount(Static(f"[{ACCENT}]CARD[/{ACCENT}]", classes="section-label", markup=True))

        card_str = _card_to_str(card).strip()
        if card_str:
            preview = markup_escape(card_str[:380])
            if len(card_str) > 380:
                preview += f"\n[{FG_MUTED}]… {len(card_str) - 380} more[/{FG_MUTED}]"
            scroll.mount(Static(preview, classes="card-body", markup=True))
        else:
            scroll.mount(Static(f"[{FG_DIM}]no card yet[/{FG_DIM}]", classes="card-body", markup=True))

        # ── Conclusions ───────────────────────────────────────────────────────
        count = len(conclusions)
        scroll.mount(Static(
            f"\n[{ACCENT}]CONCLUSIONS[/{ACCENT}] [{FG_MUTED}]({count})[/{FG_MUTED}]",
            classes="section-label", markup=True,
        ))

        if not conclusions:
            scroll.mount(Static(f"[{FG_DIM}]no conclusions yet[/{FG_DIM}]", classes="card-body", markup=True))
        else:
            for i, c in enumerate(conclusions[:12]):
                content = c.get("content", "")
                safe = markup_escape(content[:54] + "…" if len(content) > 54 else content)
                prefix = TREE_LAST if i == min(len(conclusions), 12) - 1 else TREE_MID
                scroll.mount(Static(
                    f"[{FG_DIM}]{prefix}[/{FG_DIM}][{FG}]{safe}[/{FG}]",
                    classes="conclusion-row", markup=True,
                ))
            if count > 12:
                scroll.mount(Static(f"[{FG_MUTED}]   … {count - 12} more[/{FG_MUTED}]", markup=True))

        # ── Queue ─────────────────────────────────────────────────────────────
        scroll.mount(Static(f"\n[{ACCENT}]QUEUE[/{ACCENT}]", classes="section-label", markup=True))

        if queue is None:
            scroll.mount(Static(f"[{FG_DIM}]unavailable[/{FG_DIM}]", markup=True))
        else:
            pending   = queue.get("pending", 0) or 0
            running   = queue.get("running", 0) or 0
            completed = queue.get("completed", 0) or 0

            p_dot = f"[{WARN}]{DOT}[/{WARN}]" if pending > 0 else f"[{FG_MUTED}]{DOT_EMPTY}[/{FG_MUTED}]"
            r_dot = f"[{GREEN}]{DOT}[/{GREEN}]" if running > 0 else f"[{FG_MUTED}]{DOT_EMPTY}[/{FG_MUTED}]"

            scroll.mount(Static(
                f"{p_dot} [{FG_DIM}]{pending} pending[/{FG_DIM}]  {r_dot} [{FG_DIM}]{running} running[/{FG_DIM}]",
                classes="queue-row", markup=True,
            ))
            scroll.mount(Static(
                f"[{FG_MUTED}]{DOT} {completed} done[/{FG_MUTED}]",
                classes="queue-row", markup=True,
            ))

    def show_error(self, msg: str) -> None:
        self.set_loading(False)
        scroll = self.query_one("#peer-scroll", VerticalScroll)
        scroll.remove_children()
        safe = markup_escape(msg[:80])
        scroll.mount(Static(f"\n[red]{safe}[/red]", markup=True))
