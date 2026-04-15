"""Honcho TUI — main Textual application."""

from __future__ import annotations

import argparse
import os

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import var

from honcho_tui import __version__
from honcho_tui.client import build_client, collect_page
from honcho_tui.theme import TCSS
from honcho_tui.widgets import PeerPanel, SessionsPanel, StatusBar, TranscriptPanel
from honcho_tui.widgets.sessions_panel import SessionSelected
from honcho_tui.widgets.transcript import QuerySubmitted


class HonchoTUI(App[None]):
    """Honcho TUI — memory that reasons."""

    TITLE = "HONCHO"
    CSS = TCSS

    BINDINGS = [
        Binding("q", "quit", "quit", priority=True),
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("r", "refresh_all", "refresh"),
        Binding("ctrl+d", "focus_query", "dialectic"),
        Binding("escape", "blur_query", "back"),
        Binding("tab", "focus_sessions", "sessions"),
    ]

    # ── Reactive state ────────────────────────────────────────────────────────
    workspace_id: var[str] = var("")
    peer_id: var[str] = var("")
    active_session: var[str] = var("")
    uptime: var[int] = var(0)

    def compose(self) -> ComposeResult:
        with Horizontal(id="layout"):
            yield SessionsPanel(id="sessions-panel")
            with Vertical(id="center"):
                yield TranscriptPanel(id="transcript")
            yield PeerPanel(id="peer-panel")
        yield StatusBar(id="status-bar")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        try:
            _, config = build_client()
            self.workspace_id = config.workspace_id or ""
            self.peer_id = config.peer_id or ""
        except Exception:
            self.workspace_id = ""
            self.peer_id = ""

        # Sync status bar
        bar = self.query_one("#status-bar", StatusBar)
        bar.workspace_id = self.workspace_id
        bar.peer_id = self.peer_id

        self.set_interval(1.0, self._tick_uptime)

        transcript = self.query_one("#transcript", TranscriptPanel)
        if not self.workspace_id:
            transcript.show_no_workspace()
        else:
            self._load_sessions()

    def _tick_uptime(self) -> None:
        self.uptime += 1
        bar = self.query_one("#status-bar", StatusBar)
        bar.uptime = self.uptime

    # ── Workers ───────────────────────────────────────────────────────────────

    def _load_sessions(self) -> None:
        panel = self.query_one("#sessions-panel", SessionsPanel)
        panel.set_loading(True)
        self.run_worker(self._fetch_sessions, thread=True)

    def _fetch_sessions(self) -> None:
        try:
            client, config = build_client(
                workspace_id=self.workspace_id or None,
                peer_id=self.peer_id or None,
            )
            raw = collect_page(client.sessions())
            sessions = [
                {
                    "id": s.id,
                    "is_active": getattr(s, "is_active", True),
                }
                for s in raw
            ]
            self.call_from_thread(
                self.query_one("#sessions-panel", SessionsPanel).load_sessions,
                sessions,
            )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#sessions-panel", SessionsPanel).show_error,
                str(e),
            )

    def _load_session(self, session_id: str) -> None:
        transcript = self.query_one("#transcript", TranscriptPanel)
        peer_panel = self.query_one("#peer-panel", PeerPanel)

        transcript.show_loading(f"loading session {session_id[:20]}…")
        peer_panel.set_loading(True)

        self.run_worker(
            lambda: self._fetch_session(session_id),
            thread=True,
        )

    def _fetch_session(self, session_id: str) -> None:
        try:
            client, config = build_client(
                workspace_id=self.workspace_id or None,
                peer_id=self.peer_id or None,
            )
            sess = client.session(session_id)

            # Messages (first page, last 50)
            try:
                msgs_page = sess.messages()
                raw_msgs = list(msgs_page.items)[-50:]
                messages = [
                    {
                        "id": m.id,
                        "peer_id": m.peer_id,
                        "content": m.content,
                        "created_at": m.created_at,
                    }
                    for m in raw_msgs
                ]
            except Exception:
                messages = []

            # Figure out peer to show
            peer_id = self.peer_id
            if not peer_id:
                try:
                    session_peers = collect_page(sess.peers())
                    if session_peers:
                        peer_id = session_peers[0].id
                except Exception:
                    peer_id = ""

            # Peer card + conclusions
            card = None
            conclusions: list[dict] = []
            queue = None

            if peer_id:
                p = client.peer(peer_id)
                try:
                    card = p.get_card()
                except Exception:
                    card = None

                try:
                    conc_page = p.conclusions.list(size=20)
                    conclusions = [
                        {"id": c.id, "content": c.content, "created_at": c.created_at}
                        for c in conc_page.items
                    ]
                except Exception:
                    conclusions = []

                try:
                    q = client.queue_status()
                    queue = {
                        "pending": getattr(q, "pending", 0),
                        "running": getattr(q, "running", 0) or getattr(q, "processing", 0),
                        "completed": getattr(q, "completed", 0) or getattr(q, "done", 0),
                    }
                except Exception:
                    queue = None

            # Update UI on main thread
            self.call_from_thread(self._on_session_loaded, session_id, messages, peer_id, card, conclusions, queue)

        except Exception as e:
            self.call_from_thread(self._on_session_error, str(e))

    def _on_session_loaded(
        self,
        session_id: str,
        messages: list[dict],
        peer_id: str,
        card: str | None,
        conclusions: list[dict],
        queue: dict | None,
    ) -> None:
        self.active_session = session_id

        # Update status bar
        bar = self.query_one("#status-bar", StatusBar)
        bar.session_id = session_id
        if peer_id:
            bar.peer_id = peer_id
        if queue:
            bar.pending = queue.get("pending", 0)
            bar.running = queue.get("running", 0)

        self.query_one("#transcript", TranscriptPanel).load_messages(session_id, messages)

        if peer_id:
            self.query_one("#peer-panel", PeerPanel).load_peer(peer_id, card, conclusions, queue)
        else:
            self.query_one("#peer-panel", PeerPanel).set_loading(False)

    def _on_session_error(self, msg: str) -> None:
        self.query_one("#transcript", TranscriptPanel).show_error(msg)
        self.query_one("#peer-panel", PeerPanel).show_error(msg)

    def _run_query(self, query: str) -> None:
        transcript = self.query_one("#transcript", TranscriptPanel)
        transcript.show_loading(f"querying dialectic…")
        self.run_worker(
            lambda: self._fetch_query(query),
            thread=True,
        )

    def _fetch_query(self, query: str) -> None:
        try:
            client, config = build_client(
                workspace_id=self.workspace_id or None,
                peer_id=self.peer_id or None,
            )

            peer_id = self.peer_id
            if not peer_id:
                self.call_from_thread(
                    self.query_one("#transcript", TranscriptPanel).show_error,
                    "no peer configured — set HONCHO_PEER_ID",
                )
                return

            p = client.peer(peer_id)
            response = p.chat(
                query,
                session=self.active_session or None,
            )

            self.call_from_thread(
                self.query_one("#transcript", TranscriptPanel).append_response,
                query,
                str(response) if response else "(no response)",
                True,
            )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#transcript", TranscriptPanel).show_error,
                str(e),
            )

    # ── Message handlers ──────────────────────────────────────────────────────

    def on_session_selected(self, event: SessionSelected) -> None:
        self._load_session(event.session_id)

    def on_query_submitted(self, event: QuerySubmitted) -> None:
        if not event.query.strip():
            return
        self._run_query(event.query)

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_refresh_all(self) -> None:
        if self.workspace_id:
            self._load_sessions()
            if self.active_session:
                self._load_session(self.active_session)

    def action_focus_query(self) -> None:
        self.query_one("#transcript", TranscriptPanel).focus_input()

    def action_blur_query(self) -> None:
        self.query_one("#sessions-panel").focus()

    def action_focus_sessions(self) -> None:
        self.query_one("#sessions-panel").focus()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="honcho-tui",
        description=f"Honcho TUI v{__version__} — memory that reasons",
    )
    parser.add_argument("--workspace", "-w", metavar="ID", help="Workspace ID (overrides HONCHO_WORKSPACE_ID)")
    parser.add_argument("--peer", "-p", metavar="ID", help="Peer ID (overrides HONCHO_PEER_ID)")
    parser.add_argument("--version", "-V", action="version", version=f"honcho-tui {__version__}")
    args = parser.parse_args()

    if args.workspace:
        os.environ["HONCHO_WORKSPACE_ID"] = args.workspace
    if args.peer:
        os.environ["HONCHO_PEER_ID"] = args.peer

    app = HonchoTUI()
    app.run()


if __name__ == "__main__":
    main()
