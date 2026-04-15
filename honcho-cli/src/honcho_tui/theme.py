"""Color palette, spinner frames, and TCSS for Honcho TUI."""

from __future__ import annotations

# ── Color tokens — kitty gray (warm charcoal, no navy) ────────────────────────
BG          = "#1a1a1a"    # warm dark charcoal
BG_ELEVATED = "#242424"    # slightly raised surface
BG_PANEL    = "#1e1e1e"    # side panels
BG_INPUT    = "#141414"    # query input
FG          = "#e2e2e2"    # near-white foreground
FG_DIM      = "#9a9a9a"    # mid-gray
FG_MUTED    = "#555555"    # inactive / decorative
BORDER      = "#333333"    # subtle border
ACCENT      = "#56d4dd"    # bright cyan — matches CLI branding
ACCENT_DIM  = "#1e6e78"    # dim cyan
BLUE        = "#8EA8FF"    # indigo accent (workstation shell)
CYAN        = "#56d4dd"
SAGE        = "#63D0A6"    # green / good
RUST        = "#F7A072"    # warn / orange
CLAY        = "#D4A373"    # secondary warm accent
PURPLE      = "#bc8cff"
GREEN       = "#63D0A6"    # good / success
WARN        = "#F7A072"    # warn
CRITICAL    = "#FF6B6B"    # critical
ORANGE      = "#D4A373"

# ── Spinner frame sets ─────────────────────────────────────────────────────────
FRAMES_HELIX: list[str] = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
FRAMES_ORBIT: list[str] = ["◐", "◓", "◑", "◒"]
FRAMES_DNA:   list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
FRAMES_PULSE: list[str] = ["·", "·", "•", "•", "●", "•", "•", "·"]

# ── Symbols ────────────────────────────────────────────────────────────────────
DOT       = "●"
DOT_EMPTY = "○"
DOT_HALF  = "◐"
CURSOR    = "▍"
TREE_LAST = "└ "
TREE_MID  = "├ "
SEP       = "─"

# ── TCSS ──────────────────────────────────────────────────────────────────────
TCSS = """
Screen {
    background: #1a1a1a;
    color: #e2e2e2;
    layers: base overlay;
}

/* ── MAIN LAYOUT ── */
#layout {
    height: 1fr;
    layout: horizontal;
}

/* ── LEFT: SESSIONS ── */
#sessions-panel {
    width: 26;
    background: #1e1e1e;
    border-right: solid #333333;
    layout: vertical;
}

.panel-title {
    height: 1;
    background: #242424;
    color: #56d4dd;
    text-style: bold;
    content-align: left middle;
    padding: 0 1;
}

ListView {
    background: #1e1e1e;
    height: 1fr;
    padding: 0;
}

ListView:focus {
    border: none;
}

ListItem {
    background: #1e1e1e;
    height: 1;
    padding: 0 1;
    color: #555555;
}

ListItem:hover {
    background: #242424;
    color: #e2e2e2;
}

ListItem.--highlight {
    background: #2a2a2a;
    color: #56d4dd;
}

/* ── CENTER ── */
#center {
    width: 1fr;
    layout: vertical;
    background: #1a1a1a;
}

/* ── WAVEFORM ── */
WaveformWidget {
    height: 1;
    background: #1a1a1a;
}

SineWaveWidget {
    height: 1;
    background: #1a1a1a;
}

/* ── TRANSCRIPT LOG ── */
RichLog {
    height: 1fr;
    background: #1a1a1a;
    padding: 0 2;
    scrollbar-color: #333333;
    scrollbar-background: #1a1a1a;
    scrollbar-corner-color: #1a1a1a;
    border: none;
}

/* ── QUERY BAR ── */
#query-bar {
    height: 3;
    background: #141414;
    border-top: solid #333333;
    layout: horizontal;
    align: left middle;
    padding: 0 2;
}

#query-prefix {
    width: 2;
    color: #56d4dd;
    text-style: bold;
    content-align: left middle;
}

#query-input {
    width: 1fr;
    background: #141414;
    color: #e2e2e2;
    border: none;
    padding: 0 0;
}

#query-input:focus {
    border: none;
}

/* ── RIGHT: PEER PANEL ── */
#peer-panel {
    width: 32;
    background: #1e1e1e;
    border-left: solid #333333;
    layout: vertical;
}

FreqBarsWidget {
    height: 4;
    background: #1e1e1e;
    padding: 0 1;
    border-bottom: solid #333333;
}

#peer-scroll {
    height: 1fr;
    background: #1e1e1e;
    padding: 0 1;
    scrollbar-color: #333333;
    scrollbar-background: #1e1e1e;
}

.section-label {
    color: #56d4dd;
    text-style: bold;
    height: 1;
    margin-top: 1;
}

.card-body {
    color: #9a9a9a;
}

.conclusion-row {
    color: #9a9a9a;
    height: auto;
}

.queue-row {
    height: 1;
}

/* ── STATUS BAR ── */
StatusBar {
    dock: bottom;
    height: 1;
    background: #242424;
    color: #555555;
    padding: 0 2;
}

/* ── COLLAPSIBLE ── */
Collapsible {
    background: #1e1e1e;
    border: none;
    padding: 0;
    margin: 0;
}

CollapsibleTitle {
    color: #555555;
    background: #1e1e1e;
    padding: 0;
}

CollapsibleTitle:hover {
    color: #e2e2e2;
    background: #242424;
}
"""
