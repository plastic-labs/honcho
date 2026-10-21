"""Recent-exchange ring file for blocking dialectic mode.

A resumed Hermes process starts each turn fresh, so the exchanges a blocking dialectic query
quotes are kept on disk under HERMES_HOME rather than in memory. Each exchange is the user's
message, a one-line-per-call summary of the tools the agent ran, and the agent's final reply.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("plugins.memory.honcho")

RING_FILENAME = "honcho_recent_turns.json"
KEEP_EXCHANGES = 3
_KEEP_SESSIONS = 16
_MAX_TOOL_LINES = 25
_ARGS_CHARS, _RESULT_CHARS, _TEXT_CHARS = 120, 200, 4000
_WS_RE = re.compile(r"\s+")
_ring_lock = threading.Lock()


def ring_path(hermes_home: Optional[Path]) -> Path:
    """The ring file under ``hermes_home``, else under the active HERMES_HOME, else ~/.hermes."""
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        except Exception:
            hermes_home = Path.home() / ".hermes"
    return Path(hermes_home) / RING_FILENAME


def _clip(text: str, limit: int) -> str:
    """``text`` on one line, cut to ``limit`` chars with an ellipsis."""
    flat = _WS_RE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _content_text(content: Any) -> str:
    """Text of a message's content, whether a string or a list of multimodal parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def _is_real_user(message: Any) -> bool:
    """A user message that a person wrote, not user-role runtime scaffolding."""
    try:
        from agent.conversation_compression import _is_real_user_message
        return _is_real_user_message(message)
    except Exception:
        return isinstance(message, dict) and message.get("role") == "user"


def summarize_tool_calls(messages: list) -> list[str]:
    """One ``name(args) → result`` line per assistant tool call, results matched by ``tool_call_id``."""
    results = {m.get("tool_call_id"): _content_text(m.get("content"))
               for m in messages if isinstance(m, dict) and m.get("role") == "tool"}
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            fn = (call.get("function") if isinstance(call, dict) else None) or {}
            args = fn.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
            result = results.get(call.get("id"))
            outcome = "(no result)" if result is None else _clip(result, _RESULT_CHARS) or "(empty)"
            lines.append(f"{fn.get('name') or '?'}({_clip(args or '', _ARGS_CHARS)}) → {outcome}")
    if len(lines) > _MAX_TOOL_LINES:
        lines = lines[:_MAX_TOOL_LINES] + [f"… {len(lines) - _MAX_TOOL_LINES} more"]
    return lines


def extract_exchange(user_content: str, assistant_content: str, messages: Optional[list]) -> dict[str, Any]:
    """The latest exchange: the turn's user message, its tool calls, and the final reply.

    ``messages`` is the transcript Hermes hands ``sync_turn``; the turn is everything after the
    last real user message. Without it the exchange has no tool summary."""
    user, assistant, tools = user_content or "", assistant_content or "", []
    if messages:
        start = next((i for i in range(len(messages) - 1, -1, -1) if _is_real_user(messages[i])), None)
        if start is not None:
            turn = messages[start:]
            tools = summarize_tool_calls(turn[1:])
            user = user or _content_text(turn[0].get("content"))
            if not assistant:
                replies = [_content_text(m.get("content")) for m in turn[1:]
                           if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")]
                assistant = replies[-1] if replies else ""
    return {"user": user.strip()[:_TEXT_CHARS], "tools": tools, "assistant": assistant.strip()[:_TEXT_CHARS],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _read_ring(path: Path) -> dict[str, list]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.debug("Honcho recent-turns file unreadable, starting fresh: %s", e)
        return {}
    sessions = data.get("sessions") if isinstance(data, dict) else None
    return {k: v for k, v in sessions.items() if isinstance(v, list)} if isinstance(sessions, dict) else {}


def append_exchange(path: Path, session_key: str, exchange: dict[str, Any]) -> None:
    """Append ``exchange`` to the session's ring, keeping the last few; atomic replace, never raises."""
    try:
        from utils import atomic_json_write
        with _ring_lock:
            sessions = _read_ring(path)
            turns = (sessions.pop(session_key, None) or []) + [exchange]
            # Re-inserted last, so the oldest-touched sessions are the ones dropped.
            sessions[session_key] = turns[-KEEP_EXCHANGES:]
            for stale in list(sessions)[:-_KEEP_SESSIONS]:
                sessions.pop(stale, None)
            path.parent.mkdir(parents=True, exist_ok=True)
            # User text and tool output are private, so the file is owner-only.
            atomic_json_write(path, {"version": 1, "sessions": sessions}, mode=0o600)
    except Exception as e:
        logger.warning("Honcho recent-turns write failed: %s", e)


def load_exchanges(path: Path, session_key: str) -> list[dict[str, Any]]:
    """The session's stored exchanges, oldest first; [] when there are none or the file is unreadable."""
    with _ring_lock:
        return [t for t in _read_ring(path).get(session_key, []) if isinstance(t, dict)]


def render_exchanges(exchanges: list[dict[str, Any]]) -> str:
    """The ``{recent_turns}`` block: each exchange's user message, tool calls, and reply, oldest first."""
    if not exchanges:
        return "(no earlier exchanges)"
    blocks = []
    for n, turn in enumerate(exchanges, 1):
        lines = [f"### Exchange {n} of {len(exchanges)}", f"User: {turn.get('user') or '(empty)'}"]
        if tools := turn.get("tools"):
            lines += ["Tool calls:", *(f"- {line}" for line in tools)]
        lines.append(f"Assistant: {turn.get('assistant') or '(no reply)'}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
