"""Animated display widgets: waveform, frequency bars, matrix cascade.

IMPORTANT: All Static subclasses here must:
  1. Call super().__init__("", ...) to set initial content before first render
  2. Never define a method named _render() — that's a Textual internal
"""

from __future__ import annotations

import math
import random

from rich.text import Text
from textual.widgets import Static

from honcho_tui.theme import ACCENT, CYAN, FG_DIM, FG_MUTED

BARS = " ▁▂▃▄▅▆▇█"
BLOCK = "█"
HALF_BOT = "▄"
_MC = "アイウエオカキクケコサシスセソタチツテト0123456789ABCDEF◆◇○●"


class WaveformWidget(Static):
    """Scrolling dual-sine waveform using Unicode block characters."""

    DEFAULT_CSS = """
    WaveformWidget {
        height: 1;
        background: #1a1a1a;
    }
    """

    _phase: float = 0.0
    _noise: list[float] = []

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self._noise = [random.random() for _ in range(256)]
        self.set_interval(0.07, self._advance)

    def _advance(self) -> None:
        self._phase += 0.20
        width = max(self.size.width, 40) if self.size.width > 0 else 60
        t = Text()
        for i in range(width):
            s1 = math.sin(self._phase + i * 0.28)
            s2 = math.sin(self._phase * 0.6 + i * 0.45 + 1.7)
            n = self._noise[(i + int(self._phase * 5)) % len(self._noise)]
            val = (s1 * 0.5 + s2 * 0.3 + n * 0.2 + 1.0) / 2.0
            idx = int(val * (len(BARS) - 1))
            bar = BARS[max(0, min(idx, len(BARS) - 1))]
            if val > 0.85:
                style = ACCENT
            elif val > 0.65:
                style = CYAN
            elif val > 0.35:
                style = FG_DIM
            else:
                style = FG_MUTED
            t.append(bar, style=style)
        self.update(t)


class FreqBarsWidget(Static):
    """Animated vertical frequency bars — sci-fi instrument readout."""

    DEFAULT_CSS = """
    FreqBarsWidget {
        height: 4;
        background: #1e1e1e;
        padding: 0 1;
        border-bottom: solid #333333;
    }
    """

    NUM_BARS = 14
    MAX_H = 4.0

    _heights: list[float] = []
    _targets: list[float] = []
    _phase: float = 0.0

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self._heights = [random.random() * self.MAX_H for _ in range(self.NUM_BARS)]
        self._targets = list(self._heights)
        self.set_interval(0.09, self._advance)

    def _advance(self) -> None:
        self._phase += 0.10
        for i in range(self.NUM_BARS):
            self._heights[i] += (self._targets[i] - self._heights[i]) * 0.30
            if random.random() < 0.13:
                sine = (math.sin(self._phase + i * 0.55) + 1) / 2
                self._targets[i] = sine * self.MAX_H * 0.65 + random.random() * self.MAX_H * 0.35
        self._draw()

    def _draw(self) -> None:
        rows: list[Text] = []
        max_r = int(self.MAX_H)
        for row in range(max_r, 0, -1):
            t = Text()
            for h in self._heights:
                frac = h - (row - 1)
                if frac >= 1.0:
                    style = ACCENT if row == max_r else (CYAN if row >= max_r - 1 else FG_DIM)
                    t.append(BLOCK, style=style)
                elif frac >= 0.5:
                    t.append(HALF_BOT, style=FG_DIM)
                else:
                    t.append(" ")
                t.append(" ")
            rows.append(t)

        combined = Text()
        for line in rows:
            combined.append_text(line)
            combined.append("\n")
        self.update(combined)


class MatrixCascadeWidget(Static):
    """Falling character cascade for loading states."""

    DEFAULT_CSS = """
    MatrixCascadeWidget {
        height: 5;
        background: #1a1a1a;
        padding: 0 1;
    }
    """

    COLS = 10
    ROWS = 5
    _grid: list[list[str]] = []

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self._grid = [
            [random.choice(_MC) for _ in range(self.COLS)]
            for _ in range(self.ROWS)
        ]
        self.set_interval(0.13, self._advance)

    def _advance(self) -> None:
        self._grid.pop()
        self._grid.insert(0, [random.choice(_MC) for _ in range(self.COLS)])
        t = Text()
        for r_idx, row in enumerate(self._grid):
            fade = r_idx / max(1, self.ROWS - 1)
            style = ACCENT if fade < 0.2 else (CYAN if fade < 0.45 else (FG_DIM if fade < 0.75 else FG_MUTED))
            for ch in row:
                t.append(ch, style=style)
                t.append(" ")
            t.append("\n")
        self.update(t)


class SineWaveWidget(Static):
    """Horizontal sine wave using dot characters."""

    DEFAULT_CSS = """
    SineWaveWidget {
        height: 1;
        background: #1a1a1a;
    }
    """

    _CHARS = ["·", "∘", "○", "⊙", "●", "⊙", "○", "∘"]
    _phase: float = 0.0

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)

    def on_mount(self) -> None:
        self.set_interval(0.09, self._advance)

    def _advance(self) -> None:
        self._phase += 0.22
        width = max(self.size.width, 40) if self.size.width > 0 else 60
        t = Text()
        for i in range(width):
            val = (math.sin(self._phase + i * 0.35) + 1) / 2
            idx = int(val * (len(self._CHARS) - 1))
            ch = self._CHARS[idx]
            style = ACCENT if val > 0.8 else (CYAN if val > 0.55 else FG_MUTED)
            t.append(ch, style=style)
        self.update(t)
