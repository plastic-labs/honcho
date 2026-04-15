"""Color palette, spinner frames, and TCSS for Honcho TUI."""

from __future__ import annotations

# ── Color tokens ──────────────────────────────────────────────────────────────
BG          = "#0b0e14"
BG_ELEVATED = "#141820"
BG_PANEL    = "#0f1318"
BG_INPUT    = "#0c1017"
FG          = "#c9d1d9"
FG_MUTED    = "#5c6370"
FG_DIM      = "#8b949e"
BORDER      = "#2a2d35"
ACCENT      = "#FFBF00"    # amber — primary
ACCENT_DIM  = "#7a5c00"    # dim amber border
BLUE        = "#4169e1"    # royal blue
CYAN        = "#56d4dd"
PURPLE      = "#bc8cff"
GREEN       = "#8FBC8F"    # good
WARN        = "#FFD700"    # warn
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
    background: #0b0e14;
    color: #c9d1d9;
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
    background: #0f1318;
    border-right: solid #2a2d35;
    layout: vertical;
}

.panel-title {
    height: 1;
    background: #141820;
    color: #FFBF00;
    text-style: bold;
    content-align: left middle;
    padding: 0 1;
}

ListView {
    background: #0f1318;
    height: 1fr;
}

ListView:focus {
    border: none;
}

ListItem {
    background: #0f1318;
    height: 2;
    padding: 0 1;
    color: #8b949e;
}

ListItem:hover {
    background: #141820;
    color: #c9d1d9;
}

ListItem.--highlight {
    background: #1a1f2e;
    color: #c9d1d9;
}

/* ── CENTER ── */
#center {
    width: 1fr;
    layout: vertical;
}

RichLog {
    height: 1fr;
    background: #0b0e14;
    padding: 0 1;
    scrollbar-color: #2a2d35;
    scrollbar-background: #0b0e14;
    scrollbar-corner-color: #0b0e14;
    border: none;
}

/* ── QUERY BAR ── */
#query-bar {
    height: 3;
    background: #0c1017;
    border-top: solid #2a2d35;
    layout: horizontal;
    align: left middle;
    padding: 0 1;
}

#query-prefix {
    width: 2;
    color: #FFBF00;
    text-style: bold;
    content-align: left middle;
}

#query-input {
    width: 1fr;
    background: #0c1017;
    color: #c9d1d9;
    border: none;
    padding: 0 0;
}

#query-input:focus {
    border: none;
}

/* ── RIGHT: PEER PANEL ── */
#peer-panel {
    width: 34;
    background: #0f1318;
    border-left: solid #2a2d35;
    layout: vertical;
}

#peer-scroll {
    height: 1fr;
    background: #0f1318;
    padding: 0 1;
    scrollbar-color: #2a2d35;
    scrollbar-background: #0f1318;
}

.section-label {
    color: #FFBF00;
    text-style: bold;
    height: 1;
    margin-top: 1;
}

.section-sep {
    color: #2a2d35;
    height: 1;
}

.card-body {
    color: #8b949e;
}

.conclusion-row {
    color: #8b949e;
    height: auto;
}

.queue-row {
    height: 1;
}

/* ── STATUS BAR ── */
StatusBar {
    dock: bottom;
    height: 1;
    background: #141820;
    color: #5c6370;
    layout: horizontal;
    padding: 0 1;
}

/* ── SPINNER ── */
.spinner {
    color: #FFBF00;
    width: 1;
}

/* ── COLLAPSIBLE ── */
Collapsible {
    background: #0f1318;
    border: none;
    padding: 0;
    margin: 0;
}

CollapsibleTitle {
    color: #8b949e;
    background: #0f1318;
    padding: 0;
}

CollapsibleTitle:hover {
    color: #c9d1d9;
    background: #141820;
}

/* ── STARTUP OVERLAY ── */
#startup-overlay {
    layer: overlay;
    width: 60;
    height: 12;
    background: #141820;
    border: solid #2a2d35;
    align: center middle;
    padding: 1 2;
    layout: vertical;
}

#startup-title {
    color: #FFBF00;
    text-style: bold;
    text-align: center;
}

#startup-msg {
    color: #8b949e;
    text-align: center;
    margin-top: 1;
}

#workspace-input {
    margin-top: 1;
    background: #0c1017;
    color: #c9d1d9;
    border: solid #2a2d35;
}
"""
