"""Honcho-based session management for conversation history."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.honcho.client import get_honcho_client

if TYPE_CHECKING:
    from honcho import Honcho
    from honcho.session import SessionPeerConfig


@dataclass
class HonchoSession:
    """
    A conversation session backed by Honcho.

    Provides the same interface as the original Session class
    but stores messages in Honcho for AI-native memory.
    """

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """
        Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return.

        Returns:
            List of messages in LLM format.
        """
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class HonchoSessionManager:
    """
    Manages conversation sessions using Honcho.

    Replaces the file-based SessionManager with Honcho's
    AI-native memory system for user modeling.
    """

    def __init__(self, honcho: Honcho | None = None, context_tokens: int | None = None):
        """
        Initialize the session manager.

        Args:
            honcho: Optional Honcho client. If not provided, uses the singleton.
            context_tokens: Max tokens for context() calls (None = Honcho default).
        """
        self._honcho = honcho
        self._context_tokens = context_tokens
        self._cache: dict[str, HonchoSession] = {}
        self._peers_cache: dict[str, Any] = {}
        self._sessions_cache: dict[str, Any] = {}

    @property
    def honcho(self) -> Honcho:
        """Get the Honcho client, initializing if needed."""
        if self._honcho is None:
            self._honcho = get_honcho_client()
        return self._honcho

    def _get_or_create_peer(self, peer_id: str) -> Any:
        """
        Get or create a Honcho peer.

        Peers are lazy in SDK v2 -- no API call until first use.
        Observation settings are controlled per-session via SessionPeerConfig.

        Args:
            peer_id: The peer identifier.

        Returns:
            The Honcho peer object.
        """
        if peer_id in self._peers_cache:
            return self._peers_cache[peer_id]

        peer = self.honcho.peer(peer_id)
        self._peers_cache[peer_id] = peer
        return peer

    def _get_or_create_honcho_session(
        self, session_id: str, user_peer: Any, assistant_peer: Any
    ) -> Any:
        """
        Get or create a Honcho session with peers configured.

        Args:
            session_id: The session identifier.
            user_peer: The user peer object.
            assistant_peer: The assistant peer object.

        Returns:
            The Honcho session object.
        """
        if session_id in self._sessions_cache:
            logger.debug(f"Honcho session '{session_id}' retrieved from cache")
            return self._sessions_cache[session_id], []

        session = self.honcho.session(session_id)

        # Configure peer observation settings
        from honcho.session import SessionPeerConfig
        user_config = SessionPeerConfig(observe_me=True, observe_others=True)
        ai_config = SessionPeerConfig(observe_me=False, observe_others=True)

        session.add_peers([(user_peer, user_config), (assistant_peer, ai_config)])

        # Load existing messages via context() - single call for messages + metadata
        existing_messages = []
        try:
            ctx = session.context(summary=True, tokens=self._context_tokens)
            existing_messages = ctx.messages or []

            # Verify chronological ordering
            if existing_messages and len(existing_messages) > 1:
                timestamps = [m.created_at for m in existing_messages if m.created_at]
                if timestamps and timestamps != sorted(timestamps):
                    logger.warning(
                        f"Honcho messages not chronologically ordered for session '{session_id}', sorting"
                    )
                    existing_messages = sorted(
                        existing_messages,
                        key=lambda m: m.created_at or datetime.min,
                    )

            if existing_messages:
                logger.info(f"Honcho session '{session_id}' retrieved ({len(existing_messages)} existing messages)")
            else:
                logger.info(f"Honcho session '{session_id}' created (new)")
        except Exception as e:
            logger.warning(f"Honcho session '{session_id}' loaded (failed to fetch context: {e})")

        self._sessions_cache[session_id] = session
        return session, existing_messages

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize an ID to match Honcho's pattern: ^[a-zA-Z0-9_-]+"""
        return re.sub(r'[^a-zA-Z0-9_-]', '-', id_str)

    def get_or_create(self, key: str) -> HonchoSession:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            logger.debug(f"Local session cache hit: {key}")
            return self._cache[key]

        # Parse key to extract user identifier
        # Format: channel:chat_id (e.g., "telegram:123456789")
        parts = key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else key

        # Create peer IDs (sanitized for Honcho's ID pattern)
        user_peer_id = self._sanitize_id(f"user-{channel}-{chat_id}")
        assistant_peer_id = "nanobot-assistant"

        # Sanitize session ID for Honcho
        honcho_session_id = self._sanitize_id(key)

        # Get or create peers
        user_peer = self._get_or_create_peer(user_peer_id)
        assistant_peer = self._get_or_create_peer(assistant_peer_id)

        # Get or create Honcho session
        honcho_session, existing_messages = self._get_or_create_honcho_session(
            honcho_session_id, user_peer, assistant_peer
        )

        # Convert Honcho messages to local format
        local_messages = []
        for msg in existing_messages:
            role = "assistant" if msg.peer_id == assistant_peer_id else "user"
            local_messages.append({
                "role": role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                "_synced": True,  # Already in Honcho
            })

        # Create local session wrapper with existing messages
        session = HonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=honcho_session_id,
            messages=local_messages,
        )

        self._cache[key] = session
        return session

    def save(self, session: HonchoSession) -> None:
        """
        Save messages to Honcho.

        This syncs the local message cache to Honcho's storage.

        Args:
            session: The session to save.
        """
        if not session.messages:
            return

        # Get the Honcho session and peers
        user_peer = self._get_or_create_peer(session.user_peer_id)
        assistant_peer = self._get_or_create_peer(session.assistant_peer_id)
        honcho_session = self._sessions_cache.get(session.honcho_session_id)

        if not honcho_session:
            honcho_session, _ = self._get_or_create_honcho_session(
                session.honcho_session_id, user_peer, assistant_peer
            )

        # Convert messages to Honcho format and send
        # Only send new messages (those without a 'synced' flag)
        new_messages = [m for m in session.messages if not m.get("_synced")]

        if not new_messages:
            return

        honcho_messages = []
        for msg in new_messages:
            peer = user_peer if msg["role"] == "user" else assistant_peer
            honcho_messages.append(peer.message(msg["content"]))
            msg["_synced"] = True

        try:
            honcho_session.add_messages(honcho_messages)
            logger.debug(f"Synced {len(honcho_messages)} messages to Honcho for {session.key}")
        except Exception as e:
            # Mark messages as not synced on failure
            for msg in new_messages:
                msg["_synced"] = False
            logger.error(f"Failed to sync messages to Honcho: {e}")

        # Update cache
        self._cache[session.key] = session

    def delete(self, key: str) -> bool:
        """
        Delete a session from local cache.

        Args:
            key: Session key.

        Returns:
            True if deleted from cache, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def new_session(self, key: str) -> HonchoSession:
        """
        Create a new session, preserving the old one for user modeling.

        Args:
            key: Original session key (e.g., "discord:123456").

        Returns:
            A fresh HonchoSession with no message history.
        """
        import time

        # Remove old session from caches (but don't delete from Honcho)
        old_session = self._cache.pop(key, None)
        if old_session:
            self._sessions_cache.pop(old_session.honcho_session_id, None)

        # Create new session with timestamp suffix
        timestamp = int(time.time())
        new_key = f"{key}:{timestamp}"

        session = self.get_or_create(new_key)
        self._cache[key] = session

        logger.info(f"Created new session for {key} (honcho: {session.honcho_session_id})")
        return session

    def get_user_context(self, session_key: str, query: str) -> str:
        """
        Query Honcho's dialectic chat for user context.

        Args:
            session_key: The session key to get context for.
            query: Natural language question about the user.

        Returns:
            Honcho's response about the user.
        """
        session = self._cache.get(session_key)
        if not session:
            return "No session found for this context."

        user_peer = self._get_or_create_peer(session.user_peer_id)

        try:
            return user_peer.chat(query)
        except Exception as e:
            logger.error(f"Failed to get user context from Honcho: {e}")
            return f"Unable to retrieve user context: {e}"

    def get_prefetch_context(self, session_key: str, user_message: str | None = None) -> dict[str, str]:
        """
        Pre-fetch user context using Honcho's context() method.

        Args:
            session_key: The session key to get context for.
            user_message: The user's message for semantic search.

        Returns:
            Dictionary with 'representation' and 'card' keys.
        """
        session = self._cache.get(session_key)
        if not session:
            return {}

        honcho_session = self._sessions_cache.get(session.honcho_session_id)
        if not honcho_session:
            return {}

        try:
            ctx = honcho_session.context(
                summary=False,
                tokens=self._context_tokens,
                peer_target=session.user_peer_id,
                search_query=user_message,
            )
            card = ctx.peer_card or []
            card_str = "\n".join(card) if isinstance(card, list) else str(card)
            return {
                "representation": ctx.peer_representation or "",
                "card": card_str,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch context from Honcho: {e}")
            return {}

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all cached sessions."""
        return [
            {
                "key": s.key,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": len(s.messages),
            }
            for s in self._cache.values()
        ]
