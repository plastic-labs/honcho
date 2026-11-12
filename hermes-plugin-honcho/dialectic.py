"""The per-turn ``.chat()`` injection for HonchoMemoryProvider: cadence/backoff, liveness, reasoning level."""

from __future__ import annotations

import logging
import threading
import time

from .client import spawn_context_thread

logger = logging.getLogger("plugins.memory.honcho")


class DialecticMixin:
    """Dialectic prefetch state machine. State attributes are initialised in
    HonchoMemoryProvider.__init__; this mixin only reads/writes them."""

    _LEVEL_ORDER = ("minimal", "low", "medium", "high", "max")
    # The injection's reasoning level: low, raised by query length (+1 at 120 chars, +2 at 400), at most high.
    _BASE_LEVEL, _LEVEL_CAP = "low", "high"
    _HEURISTIC_LENGTH_MEDIUM, _HEURISTIC_LENGTH_HIGH = 120, 400
    # A dialectic runs every turn; consecutive empty answers widen that, up to _BACKOFF_MAX turns.
    _DIALECTIC_CADENCE, _BACKOFF_MAX = 1, 8
    # A thread older than timeout × multiplier is treated as dead so a hung Honcho call
    # can't block future fires indefinitely.
    _STALE_THREAD_MULTIPLIER = 2.0
    # A pending result fired more than cadence × multiplier turns ago is discarded on read,
    # so a stale conversational pivot isn't injected after trivial-prompt turns.
    _STALE_RESULT_MULTIPLIER = 2
    # Turn-1 budgets: session init gets the first; the first dialectic gets the second.
    _FIRST_TURN_BASE_TIMEOUT, _FIRST_TURN_DIALECTIC_CAP = 3.0, 2.0
    # Identifies the newest dialectic thread; only it may publish a result.
    _dialectic_token: object | None = None

    def _thread_is_live(self) -> bool:
        """Thread-alive guard that treats threads past the stale threshold as dead."""
        if not self._prefetch_thread or not self._prefetch_thread.is_alive():
            return False
        timeout = self._config.timeout if self._config and self._config.timeout else 8.0
        stale_after = timeout * self._STALE_THREAD_MULTIPLIER
        age = time.monotonic() - self._prefetch_thread_started_at
        if age > stale_after:
            logger.debug("Honcho prefetch thread age %.1fs exceeds stale threshold %.1fs — treating as dead", age, stale_after)
            return False
        return True

    def _effective_cadence(self) -> int:
        """Cadence plus empty-streak backoff, capped at _BACKOFF_MAX × base."""
        cadence, streak = self._DIALECTIC_CADENCE, self._dialectic_empty_streak
        return min(cadence + streak, cadence * self._BACKOFF_MAX) if streak > 0 else cadence

    def _note_dialectic_failure(self, exc: BaseException) -> None:
        """Widen the empty-streak backoff after a failed cycle; auth failures are exempt
        because waiting cannot fix a dead token."""
        from .session import HonchoAuthError
        if isinstance(exc, HonchoAuthError):
            logger.warning("Honcho dialectic auth failure (not counted toward cadence backoff): %s", exc)
            return
        self._dialectic_empty_streak += 1

    def _spawn_dialectic(self, query: str, *, thread_name: str, fired_at: int, log_label: str) -> threading.Thread:
        """Start a background dialectic run that publishes into the pending-result slot, unless a
        newer run has started since. Only a non-empty result advances ``_last_dialectic_turn`` (so
        empty returns retry next turn) and resets the empty streak; failures widen the backoff."""
        token = object()
        with self._prefetch_lock:
            self._dialectic_token = token

        def _superseded() -> bool:
            if token is self._dialectic_token:
                return False
            logger.debug("Honcho %s superseded by a newer dialectic; result dropped", log_label)
            return True

        def _run() -> None:
            try:
                r = self._run_dialectic(query)
            except Exception as exc:
                logger.debug("Honcho %s failed: %s", log_label, exc)
                with self._prefetch_lock:
                    if _superseded():
                        return
                self._note_dialectic_failure(exc)
                return
            with self._prefetch_lock:
                if _superseded():
                    return
                if r and r.strip():
                    self._prefetch_result = r
                    self._prefetch_result_fired_at = fired_at
                    self._last_dialectic_turn = fired_at
                    self._dialectic_empty_streak = 0
                else:
                    self._dialectic_empty_streak += 1

        self._prefetch_thread_started_at = time.monotonic()
        thread = spawn_context_thread(_run, name=thread_name, owner=self)
        thread.start()
        self._prefetch_thread = thread
        return thread

    def _consume_pending_dialectic(self) -> str:
        """Pop the pending dialectic result, or "" when none is ready or it is stale."""
        with self._prefetch_lock:
            dialectic_result, fired_at = self._prefetch_result, self._prefetch_result_fired_at
            self._prefetch_result, self._prefetch_result_fired_at = "", -999
        stale_limit = self._DIALECTIC_CADENCE * self._STALE_RESULT_MULTIPLIER
        if dialectic_result and fired_at >= 0 and (self._turn_count - fired_at) > stale_limit:
            logger.debug("Honcho pending dialectic discarded as stale: fired_at=%d, turn=%d, limit=%d",
                         fired_at, self._turn_count, stale_limit)
            return ""
        return dialectic_result if (dialectic_result and dialectic_result.strip()) else ""

    def _reasoning_level_for(self, query: str) -> str:
        """The injection's level: low, one step up for a query of 120+ chars and two for 400+, at most high."""
        if not query:
            return self._BASE_LEVEL
        n = len(query)
        bump = 0 if n < self._HEURISTIC_LENGTH_MEDIUM else 1 if n < self._HEURISTIC_LENGTH_HIGH else 2
        cap_idx = self._LEVEL_ORDER.index(self._LEVEL_CAP)
        return self._LEVEL_ORDER[min(self._LEVEL_ORDER.index(self._BASE_LEVEL) + bump, cap_idx)]

    @staticmethod
    def _dialectic_prompt(is_cold: bool) -> str:
        """What the injection asks Honcho: who the user is until a first answer lands, then what matters now."""
        if is_cold:
            return ("Who is this person? What are their preferences, goals, and working style? "
                    "Focus on facts that would help an AI assistant be immediately useful.")
        return ("Given what's been discussed in this session so far, what context about this user is most "
                "relevant to the current conversation? Prioritize active context over biographical facts.")

    def _run_dialectic(self, query: str) -> str:
        """One ``.chat()`` about the user, from their own view, sized by ``query``'s length."""
        if not self._manager or not self._session_key:
            return ""
        is_cold = self._last_dialectic_turn == -999
        level = self._reasoning_level_for(query)
        logger.debug("Honcho dialectic: level=%s, cold=%s", level, is_cold)
        return self._manager.dialectic_query(self._session_key, self._dialectic_prompt(is_cold),
                                             reasoning_level=level, peer="user") or ""
