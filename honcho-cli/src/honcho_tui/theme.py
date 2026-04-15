"""Color palette, spinner frames, and TCSS for Honcho TUI."""

from __future__ import annotations

# ── Color tokens — cyan/navy to match the honcho CLI aesthetic ────────────────
BG          = "#0d1117"    # near-black with blue cast
BG_ELEVATED = "#161b22"    # panel header / status bar
BG_PANEL    = "#0d1117"    # side panels
BG_INPUT    = "#0a0e14"    # query input background
FG          = "#B6DAFD"    # periwinkle — matches BRAND in branding.py
FG_DIM      = "#7aadce"    # mid-brightness cyan-blue
FG_MUTED    = "#3d6480"    # dim inactive text
BORDER      = "#2a4a5a"    # panel borders
ACCENT      = "#56d4dd"    # bright cyan — primary accent
ACCENT_DIM  = "#1e4a55"    # dim cyan for inactive states
BLUE        = "#4169e1"    # royal blue
CYAN        = "#56d4dd"
PURPLE      = "#bc8cff"
GREEN       = "#7ee6a8"    # good / success
WARN        = "#e6c855"    # warn
CRITICAL    = "#FF6B6B"    # critical
ORANGE      = "#e6a855"

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
CHEVRON_OPEN   = "▾"
CHEVRON_CLOSED = "▸"

# ── TCSS ──────────────────────────────────────────────────────────────────────
TCSS = """
Screen {
    background: #0d1117;
    color: #B6DAFD;
    layers: base overlay;
}

/* ── MAIN LAYOUT ── */
#layout {
    height: 1fr;
    layout: horizontal;
}

/* ── LEFT: SESSIONS ── */
#sessions-panel {
    width: 28;
    background: #0d1117;
    border-right: solid #2a4a5a;
    layout: vertical;
}

.panel-title {
    height: 1;
    background: #161b22;
    color: #56d4dd;
    text-style: bold;
    content-align: left middle;
    padding: 0 1;
}

ListView {
    background: #0d1117;
    height: 1fr;
}

ListView:focus {
    border: none;
}

ListItem {
    background: #0d1117;
    height: 2;
    padding: 0 1;
    color: #3d6480;
}

ListItem:hover {
    background: #161b22;
    color: #B6DAFD;
}

ListItem.--highlight {
    background: #0f1e2e;
    color: #B6DAFD;
}

/* ── CENTER ── */
#center {
    width: 1fr;
    layout: vertical;
}

RichLog {
    height: 1fr;
    background: #0d1117;
    padding: 0 1;
    scrollbar-color: #2a4a5a;
    scrollbar-background: #0d1117;
    scrollbar-corner-color: #0d1117;
    border: none;
}

/* ── QUERY BAR ── */
#query-bar {
    height: 3;
    background: #0a0e14;
    border-top: solid #2a4a5a;
    layout: horizontal;
    align: left middle;
    padding: 0 1;
}

#query-prefix {
    width: 2;
    color: #56d4dd;
    text-style: bold;
    content-align: left middle;
}

#query-input {
    width: 1fr;
    background: #0a0e14;
    color: #B6DAFD;
    border: none;
    padding: 0 0;
}

#query-input:focus {
    border: none;
}

/* ── RIGHT: PEER PANEL ── */
#peer-panel {
    width: 34;
    background: #0d1117;
    border-left: solid #2a4a5a;
    layout: vertical;
}

#peer-scroll {
    height: 1fr;
    background: #0d1117;
    padding: 0 1;
    scrollbar-color: #2a4a5a;
    scrollbar-background: #0d1117;
}

.section-label {
    color: #56d4dd;
    text-style: bold;
    height: 1;
    margin-top: 1;
}

.card-body {
    color: #7aadce;
}

.conclusion-row {
    color: #7aadce;
    height: auto;
}

.queue-row {
    height: 1;
}

/* ── STATUS BAR ── */
StatusBar {
    dock: bottom;
    height: 1;
    background: #161b22;
    color: #3d6480;
    layout: horizontal;
    padding: 0 1;
}

/* ── SPINNER ── */
.spinner {
    color: #56d4dd;
    width: 1;
}

/* ── COLLAPSIBLE ── */
Collapsible {
    background: #0d1117;
    border: none;
    padding: 0;
    margin: 0;
}

CollapsibleTitle {
    color: #3d6480;
    background: #0d1117;
    padding: 0;
}

CollapsibleTitle:hover {
    color: #B6DAFD;
    background: #161b22;
}
"""
