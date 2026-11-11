"""Recall/read-path cluster for HonchoSessionManager: cards, search, conclusions, chat. Every read is a peer's
own view, and none calls ``context()``."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .session_auth import HonchoAuthError

logger = logging.getLogger("plugins.memory.honcho.session")

_FAILED = object()  # sentinel: a guarded call raised (distinct from a legitimately empty/None result)

# Automatic injection: Honcho's chat answer is clipped to this, and a query to this.
_INJECTION_MAX_CHARS, _DIALECTIC_MAX_INPUT_CHARS = 600, 10000
# Conclusion reads return what Honcho derived, never the explicit facts already in the messages.
_DERIVED_ONLY = {"level": {"in": ["deductive", "inductive", "contradiction"]}}


class SessionContextMixin:
    """Peer card / representation / search / conclusion / chat operations against Honcho."""

    def _guarded(self, fn: Callable[[], Any], default: Any, level: int, msg: str, *args: Any) -> Any:
        """Run ``fn``; re-raise HonchoAuthError, log (exc appended to args) and return default otherwise."""
        try:
            return fn()
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.log(level, msg, *args, e)
            return default

    def _guarded_authed(self, label: str, fn: Callable[[], Any], default: Any, level: int, msg: str, *args: Any) -> Any:
        """``_guarded`` around ``_authed_call(label, fn)`` (401 -> forced refresh + one retry)."""
        return self._guarded(lambda: self._authed_call(label, fn), default, level, msg, *args)

    def _guarded_session(
        self, session_key: str, fn: Callable[[Any], Any], default: Any, level: int, msg: str, *args: Any,
    ) -> Any:
        """``_guarded`` over ``fn(session)`` for the cached session; ``default`` when no session is cached."""
        session = self._cached_session(session_key)
        return self._guarded(lambda: fn(session), default, level, msg, *args) if session else default

    @staticmethod
    def _normalize_card(card: Any) -> list[str]:
        """Normalize Honcho card payloads into a plain list of strings."""
        return ([str(item) for item in card if item] if isinstance(card, list) else [str(card)]) if card else []

    @staticmethod
    def _target_kwargs(target: str | None) -> dict[str, str]:
        """SDK peer getters take ``target=`` only when observing another peer."""
        return {} if target is None else {"target": target}

    def _fetch_peer_card(self, peer_id: str, *, target: str | None = None) -> list[str]:
        """Fetch a peer card from the peer object (session.context() can return an empty card)."""
        def _get_card() -> Any:
            peer = self._get_or_create_peer(peer_id)
            getters = (getattr(peer, n, None) for n in ("get_card", "card"))  # "card" is the legacy SDK getter
            getter = next((g for g in getters if callable(g)), None)
            return getter(**self._target_kwargs(target)) if getter else None
        return self._normalize_card(self._authed_call("peer card fetch", _get_card))

    def _fetch_peer_context(self, peer_id: str) -> dict[str, Any]:
        """A peer's own representation and card, from their dedicated getters. Raises HonchoAuthError when
        auth is dead or a 401 survives the forced refresh."""
        failed = "Direct %s failed for '%%s': %%s"
        representation = self._guarded_authed(
            "peer representation fetch", lambda: self._get_or_create_peer(peer_id).representation(),
            "", logging.DEBUG, failed % "peer.representation()", peer_id,
        ) or ""
        card = self._guarded(lambda: self._fetch_peer_card(peer_id), [], logging.DEBUG, failed % "peer card fetch", peer_id)
        return {"representation": representation, "card": card}

    def _peer_context_strings(self, peer_id: str):
        """``_fetch_peer_context`` flattened to ``(representation, newline-joined card)``."""
        ctx = self._fetch_peer_context(peer_id)
        return ctx["representation"], "\n".join(ctx["card"])

    def get_peer_card(self, session_key: str, peer: str = "user") -> list[str]:
        """Fetch a peer card (curated facts, no LLM). [] if unavailable; raises HonchoAuthError."""
        def _fetch(session: Any) -> list[str]:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            return self._fetch_peer_card(observer_peer_id, target=target_peer_id)
        return self._guarded_session(session_key, _fetch, [], logging.DEBUG, "Failed to fetch peer card from Honcho: %s")

    def search_context(self, session_key: str, query: str, max_tokens: int = 800, peer: str = "user") -> str:
        """Hybrid search over raw messages visible from ``peer``'s perspective, all sessions. Snippets
        accumulate until ``max_tokens`` (~4 chars/token) is exhausted. Returns "" when nothing matches;
        raises HonchoAuthError on rejected credentials."""
        session = self._cached_session(session_key)
        q = (query or "").strip()[:4000]  # Honcho caps query length for the embedding model.
        if not session or not q:
            return ""
        peer_id = self._resolve_peer_id(session, peer)
        char_budget = max(200, int(max_tokens) * 4)
        limit = max(3, min(20, char_budget // 300))
        messages = self._guarded_authed(
            "message search", lambda: self.honcho.search(q, filters={"peer_perspective": peer_id}, limit=limit),
            _FAILED, logging.DEBUG, "Honcho message search failed (peer_perspective=%s): %s", peer_id,
        )
        if messages is _FAILED or not messages:
            # Older Honcho versions lack the perspective filter (raise), and servers before v3.1.0
            # moved joined_at on every peer re-add, so the filter returns [] for history the peer was
            # present for; upgrading doesn't restore those windows. A peer's own messages are always
            # in its perspective, so peer-authored search is a safe fallback for both.
            messages = self._guarded_authed(
                "peer search", lambda: self._peer_ref(peer_id).search(q, limit=limit),
                None, logging.DEBUG, "Honcho peer search fallback also failed: %s",
            )
        if not messages:
            return ""
        # Author labels distinguish user-stated facts from assistant-derived ones.
        lines: list[str] = []
        for m in messages:
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            author = getattr(m, "peer_id", "") or "unknown"
            who = "assistant" if author == session.assistant_peer_id else author
            sess = getattr(m, "session_id", "") or ""
            entry = f"[{who}{f' · {sess}' if sess else ''}] {content[:1200]}"
            # Budget left after the joined snippets so far plus the separator this entry would need.
            remaining = char_budget - len("\n\n".join(lines)) - (2 if lines else 0)
            if remaining <= 0:
                break
            truncated = len(entry) > remaining
            entry = entry[:remaining].rstrip()
            if not entry:
                break
            lines.append(entry)
            if truncated:
                break
        return "\n\n".join(lines)

    def _conclusions_scope(self, session: Any, target_peer_id: str) -> Any:
        """The peer's conclusions about itself (its own view); shared by create/delete/list."""
        return self._get_or_create_peer(target_peer_id).conclusions_of(target_peer_id)

    def create_conclusion(self, session_key: str, content: str, peer: str = "user") -> bool:
        """Write a conclusion (durable fact) about ``peer`` back to Honcho."""
        if not content or not content.strip():
            return False
        session = self._cached_session(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping conclusion", session_key)
            return False

        def _create() -> bool:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                logger.warning("Could not resolve conclusion peer '%s' for session '%s'", peer, session_key)
                return False
            payload = [{"content": content.strip(), "session_id": session.honcho_session_id}]
            self._authed_call("conclusion create", lambda: self._conclusions_scope(session, target_peer_id).create(payload))
            logger.info("Created conclusion about %s for %s: %s", target_peer_id, session_key, content[:80])
            return True
        return self._guarded(_create, False, logging.ERROR, "Failed to create conclusion: %s")

    def delete_conclusion(self, session_key: str, conclusion_id: str, peer: str = "user") -> bool:
        """Delete a conclusion by ID. Use only for PII removal."""
        def _delete(session: Any) -> bool:
            target_peer_id = self._resolve_peer_id(session, peer)
            self._authed_call(
                "conclusion delete", lambda: self._conclusions_scope(session, target_peer_id).delete(conclusion_id),
            )
            logger.info("Deleted conclusion %s for %s", conclusion_id, session_key)
            return True
        return self._guarded_session(
            session_key, _delete, False, logging.ERROR, "Failed to delete conclusion %s: %s", conclusion_id,
        )

    def list_conclusions(self, session_key: str, query: str | None = None, peer: str = "user", limit: int = 20):
        """List (or semantically search with ``query``) the conclusions Honcho derived, as {"id", "content"}
        dicts. Explicit ones only restate the messages, so they are left out."""
        def _list(session: Any) -> list[dict]:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                return []

            def _fetch() -> Any:
                scope = self._conclusions_scope(session, target_peer_id)
                if query:
                    return scope.query(query, top_k=limit, filters=_DERIVED_ONLY)
                return scope.list(size=limit, filters=_DERIVED_ONLY).items
            return [{"id": c.id, "content": c.content} for c in self._authed_call("conclusion list", _fetch)]
        return self._guarded_session(session_key, _list, [], logging.DEBUG, "Honcho list_conclusions failed: %s")

    def set_peer_card(self, session_key: str, card: list[str], peer: str = "user") -> list[str] | None:
        """Replace a peer's card. Returns the updated card, or None on failure."""
        def _update(session: Any) -> list[str] | None:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            if observer_peer_id is None:
                logger.warning("Could not resolve peer '%s' for set_peer_card in session '%s'", peer, session_key)
                return None
            result = self._authed_call(
                "peer card update",
                lambda: self._get_or_create_peer(observer_peer_id).set_card(card, **self._target_kwargs(target_peer_id)),
            )
            logger.info("Updated peer card observer=%s target=%s (%d facts)",
                        observer_peer_id, target_peer_id or observer_peer_id, len(card))
            return result
        return self._guarded_session(session_key, _update, None, logging.ERROR, "Failed to set peer card: %s")

    def seed_ai_identity(self, session_key: str, content: str, source: str = "manual") -> bool:
        """Seed the AI peer's representation from text (SOUL.md, exported chats, ...), sent as an
        assistant-peer message so Honcho's reasoning model incorporates it. Unlike the other
        operations, auth failures are logged and swallowed here too."""
        if not content or not content.strip():
            return False
        session = self._cached_session(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping AI seed", session_key)
            return False
        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping AI seed", session_key)
            return False
        wrapped = f"<ai_identity_seed>\n<source>{source}</source>\n\n{content.strip()}\n</ai_identity_seed>"

        def _seed() -> bool:
            assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
            self._sdk_session(session.honcho_session_id).add_messages([assistant_peer.message(wrapped)])
            logger.info("Seeded AI identity from '%s' into %s", source, session_key)
            return True
        try:
            return self._authed_call("identity seed", _seed)
        except Exception as e:
            logger.error("Failed to seed AI identity: %s", e)
            return False

    def get_ai_representation(self, session_key: str) -> dict[str, str]:
        """Fetch the AI peer's representation + card ("" values if unavailable)."""
        def _fetch(session: Any) -> dict[str, str]:
            rep, card = self._peer_context_strings(session.assistant_peer_id)
            return {"representation": rep, "card": card}
        return self._guarded_session(
            session_key, _fetch, {"representation": "", "card": ""}, logging.DEBUG,
            "Failed to fetch AI representation: %s",
        )

    def dialectic_query(
        self, session_key: str, query: str, reasoning_level: str | None = None, peer: str = "user",
        apply_injection_cap: bool = True, raise_errors: bool = False, scope: str | list[str] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Ask Honcho's chat endpoint about a peer, from that peer's own view (an LLM call on the backend; run
        off-thread). ``reasoning_level`` defaults to low. ``apply_injection_cap`` clips the answer for automatic
        injection; explicit tool calls pass False. ``raise_errors`` re-raises backend failures instead of
        returning "" so a tool call can tell a timeout from an empty answer (#36098). ``scope`` limits the
        answer to sessions in those scopes; ``timeout`` overrides the client's for this call. Raises
        HonchoAuthError when credentials are rejected after a forced refresh and one retry."""
        session = self._cached_session(session_key)
        target_peer_id = self._resolve_peer_id(session, peer) if session else None
        if target_peer_id is None:
            return ""
        if len(query) > _DIALECTIC_MAX_INPUT_CHARS:
            query = query[:_DIALECTIC_MAX_INPUT_CHARS].rsplit(" ", 1)[0]
        kwargs: dict[str, Any] = {"reasoning_level": reasoning_level or "low"}
        if scope:
            kwargs["scope"] = scope
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            result = self._authed_call(
                "dialectic query", lambda: self._get_or_create_peer(target_peer_id).chat(query, **kwargs) or "")
            cap = _INJECTION_MAX_CHARS if apply_injection_cap else 0
            return result[:cap].rsplit(" ", 1)[0] + " …" if result and cap and len(result) > cap else result
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.warning("Honcho dialectic query failed: %s", e)
            if raise_errors:
                raise
            return ""

    def workspace_chat(self, query: str, reasoning_level: str, scope: str | list[str] | None = None) -> str:
        """Ask Honcho about everyone in the workspace (optionally limited to ``scope``). Raises on failure."""
        kwargs: dict[str, Any] = {"reasoning_level": reasoning_level}
        if scope:
            kwargs["scope"] = scope
        return self._authed_call("workspace chat", lambda: self.honcho.chat(query, **kwargs)) or ""
