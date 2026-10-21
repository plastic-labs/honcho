"""Terminal checkbox picker for selecting agents interactively.

Provides a raw-mode arrow-key + space checkbox picker when connected to a TTY,
with graceful fallback to a numbered prompt when raw mode is unavailable.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.text import Text

if TYPE_CHECKING:
    from honcho_cli.agents import Agent

_console = Console()


def _read_key() -> str:
    """Read a single keypress from standard input in raw mode."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            ch2 = msvcrt.getch()
            if ch2 == b"H":
                return "up"
            if ch2 == b"P":
                return "down"
            return "unknown"
        if ch == b"\r":
            return "enter"
        if ch == b" ":
            return "space"
        if ch == b"\x03":
            return "ctrl_c"
        return ch.decode("utf-8", errors="ignore").lower()

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, when=termios.TCSANOW)
        ch = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            # Check for escape sequence
            import select

            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch2 = os.read(fd, 1).decode("utf-8", errors="ignore")
                if ch2 == "[":
                    ch3 = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch in ("", "\x03", "\x04"):
            return "ctrl_c"
        return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_checkbox_picker(agents: list[Agent]) -> list[Agent]:
    """Present a checkbox picker for the given agents.

    Returns the list of chosen agents.
    """
    if not agents:
        return []

    # If non-interactive or cannot read stdin, use prompt fallback
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return _fallback_picker(agents)

    try:
        return _raw_checkbox_picker(agents)
    except (OSError, ValueError):
        return _fallback_picker(agents)


def _raw_checkbox_picker(agents: list[Agent]) -> list[Agent]:
    cursor = 0
    selected: set[int] = set()

    installed = [agent.has_honcho() for agent in agents]
    with Live(console=_console, auto_refresh=False, transient=True) as live:
        while True:
            # Prepare render lines
            lines: list[str] = [
                "Select agents to add Honcho:",
                "[dim]Use ↑/↓ or j/k to navigate · Space to toggle · 'a' to toggle all · Enter to confirm · 'q' to cancel[/dim]",
                "",
            ]

            for i, agent in enumerate(agents):
                checked = escape("[x]" if i in selected else "[ ]")
                pointer = "❯" if i == cursor else " "
                kind_str = f"({agent.kind})"
                already = (
                    " [dim](Honcho already installed)[/dim]" if installed[i] else ""
                )

                if i == cursor:
                    lines.append(
                        f"[bold cyan]{pointer} {checked} {agent.name}[/bold cyan] [dim]{kind_str}[/dim]{already}"
                    )
                else:
                    lines.append(
                        f"  {checked} {agent.name} [dim]{kind_str}[/dim]{already}"
                    )

            lines.append("")

            live.update(Text.from_markup("\n".join(lines)), refresh=True)

            key = _read_key()

            if key in ("up", "k"):
                cursor = (cursor - 1) % len(agents)
            elif key in ("down", "j"):
                cursor = (cursor + 1) % len(agents)
            elif key == "space":
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.add(cursor)
            elif key in ("a", "A"):
                if len(selected) == len(agents):
                    selected.clear()
                else:
                    selected = set(range(len(agents)))
            elif key == "enter":
                break
            elif key in ("escape", "ctrl_c", "q"):
                return []

        return [agents[i] for i in sorted(selected)]


def _fallback_picker(agents: list[Agent]) -> list[Agent]:
    """Numbered text prompt fallback when raw terminal interaction is unavailable."""
    _console.print("\nDetected agents:")
    for i, agent in enumerate(agents, 1):
        already = " [dim](Honcho already installed)[/dim]" if agent.has_honcho() else ""
        _console.print(f"  {i}. {agent.name} [dim]({agent.kind})[/dim]{already}")
    _console.print(
        "\nEnter numbers to install (comma-separated, e.g. '1, 2', 'all', or press Enter to cancel):"
    )

    try:
        line = sys.stdin.readline().strip()
    except OSError:
        return []

    if not line:
        return []

    if line.lower() == "all":
        return [a for a in agents if not a.has_honcho()]

    chosen: list[Agent] = []
    for part in line.split(","):
        p = part.strip()
        if p.isdigit():
            idx = int(p) - 1
            if 0 <= idx < len(agents):
                chosen.append(agents[idx])
    return chosen
