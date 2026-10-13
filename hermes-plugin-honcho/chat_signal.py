"""Read Honcho's tool-loop diagnostics (``iterations`` / ``capped_out``) off a dialectic answer.

The server's chat route answers ``POST /v3/workspaces/{ws}/peers/{peer}/chat`` with
``{"content": str, "iterations": int | None, "capped_out": bool | None}`` when the deployment
reports them, and puts the same two keys on the SSE terminator. ``capped_out`` is true when
Honcho's tool loop burned its whole per-level iteration budget and fell back to a tool-less
synthesis: the answer reads like a normal one but is not built from the retrieved facts.

The installed SDK drops both keys before this plugin can see them: ``honcho/peer.py``
``Peer.chat()`` posts the route, then returns ``data.get("content")`` and nothing else. So the
diagnostics are reachable on the wire but not through ``chat()``'s return type. Rather than fork a
vendored package, this module posts the same route through the SDK's own HTTP client — auth, base
URL, timeout and retry handling stay the SDK's — and falls back to ``Peer.chat()`` whenever those
internals are missing, so an SDK reshuffle degrades to the pre-change behaviour (signal unknown)
instead of breaking recall.

Both keys are optional and either may be ``None``, which means *unknown* (an older server, or a
response shape we did not get) — never "not capped". A server that does not report them keeps the
old behaviour exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("plugins.memory.honcho.session")

# Honcho's reasoning levels, weakest first — the order an escalation walks up.
REASONING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "max")


@dataclass(frozen=True)
class ChatAnswer:
    """One dialectic answer plus whatever the tool loop reported about it.

    ``iterations`` and ``capped_out`` are optional on the wire and both may be ``None``: that
    means *unknown* (older Honcho, or a response shape we did not get), never "not capped".
    """

    content: str | None = None
    iterations: int | None = None
    capped_out: bool | None = None

    @property
    def truncated(self) -> bool:
        """True only on an affirmative cap-out report; ``None`` is unknown, not complete."""
        return self.capped_out is True


def next_reasoning_level(level: str | None) -> str | None:
    """The next level up from ``level``, or None at the ceiling or for an unrecognised level."""
    if level not in REASONING_LEVELS:
        return None
    idx = REASONING_LEVELS.index(level)
    return REASONING_LEVELS[idx + 1] if idx + 1 < len(REASONING_LEVELS) else None


def _as_int(value: Any) -> int | None:
    """Coerce a JSON number to int; anything else (bool, str, mocks) is unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _as_bool(value: Any) -> bool | None:
    """Coerce a JSON boolean (or its 0/1 and "true"/"false" spellings) to bool; else unknown."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _legacy_chat(peer: Any, query: str, *, target: Any, session: Any, reasoning_level: str | None) -> Any:
    """``Peer.chat()`` with exactly the call shape this plugin used before the change."""
    kwargs: dict[str, Any] = {}
    if target is not None:
        kwargs["target"] = target
    if session is not None:
        kwargs["session"] = session
    if reasoning_level:
        kwargs["reasoning_level"] = reasoning_level
    return peer.chat(query, **kwargs)


def _raw_body(peer: Any, query: str, *, target: Any, session: Any, reasoning_level: str | None) -> dict | None:
    """The chat route's raw JSON body, or None when the SDK's HTTP layer isn't reachable.

    Mirrors ``Peer.chat()``'s request construction (ensure_workspace, resolve_id, the four body
    keys) so the answer is the same answer, just with the keys the SDK's return type discards.
    """
    try:
        from honcho.http import routes
        from honcho.utils import resolve_id
    except Exception:  # honcho absent or reshuffled
        return None
    try:
        client = getattr(peer, "_honcho", None)
        http = getattr(client, "_http", None)
        workspace_id = getattr(peer, "workspace_id", None) or getattr(client, "workspace_id", None)
        peer_id = getattr(peer, "id", None)
        if http is None or not workspace_id or not isinstance(peer_id, str) or not peer_id:
            return None
        ensure_workspace = getattr(client, "_ensure_workspace", None)
        if callable(ensure_workspace):
            ensure_workspace()
        body: dict[str, Any] = {"query": query, "stream": False}
        if target is not None:
            body["target"] = resolve_id(target)
        if session is not None:
            body["session_id"] = resolve_id(session)
        if reasoning_level:
            body["reasoning_level"] = reasoning_level
        data = http.post(routes.peer_chat(workspace_id, peer_id), body=body)
        # A mock or a reshuffled SDK yields a non-dict here; fall back rather than guess.
        return data if isinstance(data, dict) else None
    except Exception as e:
        # Never break recall over the diagnostics: the caller retries through chat() below.
        logger.debug("Honcho chat diagnostics unavailable (%s: %s) — falling back to peer.chat()", type(e).__name__, e)
        return None


def chat_with_signal(
    peer: Any,
    query: str,
    *,
    target: Any = None,
    session: Any = None,
    reasoning_level: str | None = None,
) -> ChatAnswer:
    """One dialectic ``chat`` call, carrying the tool-loop diagnostics when they are reachable.

    ``content`` is None for a falsy body content, matching the SDK's own "no answer" convention.
    """
    data = _raw_body(peer, query, target=target, session=session, reasoning_level=reasoning_level)
    if data is not None:
        content = data.get("content")
        return ChatAnswer(
            content=content if isinstance(content, str) and content else None,
            iterations=_as_int(data.get("iterations")),
            capped_out=_as_bool(data.get("capped_out")),
        )
    fallback = _legacy_chat(peer, query, target=target, session=session, reasoning_level=reasoning_level)
    return ChatAnswer(content=fallback or None)
