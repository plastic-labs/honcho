"""
Honcho storage adapters for CrewAI memory.

`HonchoMemoryStorage` implements CrewAI's current unified memory
`StorageBackend` protocol. `HonchoStorage` is kept as a compatibility adapter
for older CrewAI `ExternalMemory` usage.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from honcho import Honcho

from honcho_crewai.exceptions import HonchoDependencyError

try:  # CrewAI <= 1.9 external memory interface.
    from crewai.memory.storage.interface import Storage as LegacyStorage
except ModuleNotFoundError:  # CrewAI >= 1.10 unified memory only.

    class LegacyStorage:  # type: ignore[no-redef]
        pass


try:  # CrewAI >= 1.10 unified memory types.
    from crewai.memory.types import MemoryRecord, ScopeInfo
except ModuleNotFoundError:
    MemoryRecord = None  # type: ignore[assignment]
    ScopeInfo = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_MEMORY_KIND = "crewai_memory_record"
_KIND_KEY = "honcho_crewai_kind"
_DELETED_KEY = "honcho_crewai_deleted"
_RECORD_ID_KEY = "crewai_record_id"
_SCOPE_KEY = "crewai_scope"
_CATEGORIES_KEY = "crewai_categories"
_MEMORY_METADATA_KEY = "crewai_metadata"
_IMPORTANCE_KEY = "crewai_importance"
_CREATED_AT_KEY = "crewai_created_at"
_LAST_ACCESSED_KEY = "crewai_last_accessed"
_EMBEDDING_KEY = "crewai_embedding"
_SOURCE_KEY = "crewai_source"
_PRIVATE_KEY = "crewai_private"


def _require_unified_memory() -> None:
    if MemoryRecord is None or ScopeInfo is None:
        raise HonchoDependencyError("CrewAI unified memory", "uv add crewai>=1.14.3")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Could not parse datetime %r", value)
    return fallback or datetime.now(UTC)


def _scope_matches(scope: str, scope_prefix: str | None) -> bool:
    if scope_prefix in (None, "", "/"):
        return True

    normalized = scope_prefix.rstrip("/")
    return scope == normalized or scope.startswith(f"{normalized}/")


def _category_matches(
    record_categories: list[str], categories: list[str] | None
) -> bool:
    if not categories:
        return True
    return bool(set(record_categories).intersection(categories))


def _metadata_matches(
    metadata: dict[str, Any], metadata_filter: dict[str, Any] | None
) -> bool:
    if not metadata_filter:
        return True
    return all(metadata.get(key) == value for key, value in metadata_filter.items())


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class HonchoMemoryStorage:
    """
    CrewAI unified memory storage backend backed by Honcho messages.

    CrewAI's current memory system embeds records before passing them to custom
    storage. This adapter stores those embeddings in Honcho message metadata and
    performs vector search locally over the session's active memory records.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        peer_id: str = "crewai-memory",
        honcho_client: Honcho | None = None,
    ) -> None:
        _require_unified_memory()
        self.honcho = honcho_client or Honcho()
        self.session_id = session_id or str(uuid.uuid4())
        self.peer_id = peer_id
        self._session: Any | None = None
        self._peer: Any | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            self._session = self.honcho.session(self.session_id)
        return self._session

    @property
    def peer(self) -> Any:
        if self._peer is None:
            self._peer = self.honcho.peer(self.peer_id)
        return self._peer

    def save(self, records: list[Any]) -> None:
        """Save CrewAI memory records to Honcho."""
        if not records:
            return

        messages = [
            self.peer.message(
                record.content,
                metadata=self._record_metadata(record),
                created_at=record.created_at,
            )
            for record in records
        ]
        self.session.add_messages(messages)

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Search records by cosine similarity over CrewAI-provided embeddings."""
        matches: list[tuple[Any, float]] = []
        for _, record in self._active_record_messages():
            if not self._record_matches(
                record, scope_prefix, categories, metadata_filter
            ):
                continue

            score = _cosine_similarity(query_embedding, record.embedding)
            if score >= min_score:
                matches.append((record, score))

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Tombstone records that match the delete criteria."""
        deleted = 0
        record_id_set = set(record_ids or [])

        for message, record in self._active_record_messages():
            if record_id_set and record.id not in record_id_set:
                continue
            if not self._record_matches(
                record, scope_prefix, categories, metadata_filter
            ):
                continue
            if older_than is not None and record.created_at >= older_than:
                continue

            metadata = dict(message.metadata)
            metadata[_DELETED_KEY] = True
            self.session.update_message(message, metadata=metadata)
            deleted += 1

        return deleted

    def update(self, record: Any) -> None:
        """Replace an existing record by tombstoning old copies and saving the new one."""
        self.delete(record_ids=[record.id])
        self.save([record])

    def get_record(self, record_id: str) -> Any | None:
        """Return the newest active record with the given ID."""
        records = [
            record
            for _, record in self._active_record_messages()
            if record.id == record_id
        ]
        if not records:
            return None
        return max(records, key=lambda record: record.created_at)

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        """List active records, newest first."""
        records = [
            record
            for _, record in self._active_record_messages()
            if _scope_matches(record.scope, scope_prefix)
        ]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> Any:
        """Build CrewAI scope metadata from active Honcho-backed records."""
        _require_unified_memory()
        records = self.list_records(scope_prefix=scope, limit=10_000)
        categories = sorted(
            {category for record in records for category in record.categories}
        )
        created_at_values = [record.created_at for record in records]

        return ScopeInfo(  # type: ignore[operator]
            path=scope,
            record_count=len(records),
            categories=categories,
            oldest_record=min(created_at_values) if created_at_values else None,
            newest_record=max(created_at_values) if created_at_values else None,
            child_scopes=self.list_scopes(scope),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes below `parent`."""
        children: set[str] = set()
        parent = parent.rstrip("/") or "/"

        for record in self.list_records(scope_prefix=parent, limit=10_000):
            scope = record.scope.rstrip("/") or "/"
            if scope == parent:
                continue

            if parent == "/":
                parts = [part for part in scope.split("/") if part]
                if parts:
                    children.add(f"/{parts[0]}")
            else:
                remainder = scope.removeprefix(parent).strip("/")
                if remainder:
                    children.add(f"{parent}/{remainder.split('/')[0]}")

        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """Count categories in active records."""
        counts: dict[str, int] = {}
        for record in self.list_records(scope_prefix=scope_prefix, limit=10_000):
            for category in record.categories:
                counts[category] = counts.get(category, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Count active records in a scope."""
        return len(self.list_records(scope_prefix=scope_prefix, limit=10_000))

    def reset(self, scope_prefix: str | None = None) -> None:
        """Tombstone all records in a scope, or all records when no scope is given."""
        self.delete(scope_prefix=scope_prefix)

    async def asave(self, records: list[Any]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete,
            scope_prefix,
            categories,
            record_ids,
            older_than,
            metadata_filter,
        )

    def _record_metadata(self, record: Any) -> dict[str, Any]:
        return {
            _KIND_KEY: _MEMORY_KIND,
            _DELETED_KEY: False,
            _RECORD_ID_KEY: record.id,
            _SCOPE_KEY: record.scope,
            _CATEGORIES_KEY: list(record.categories),
            _MEMORY_METADATA_KEY: dict(record.metadata),
            _IMPORTANCE_KEY: record.importance,
            _CREATED_AT_KEY: _iso(record.created_at),
            _LAST_ACCESSED_KEY: _iso(record.last_accessed),
            _EMBEDDING_KEY: record.embedding,
            _SOURCE_KEY: record.source,
            _PRIVATE_KEY: record.private,
        }

    def _active_record_messages(self) -> Iterable[tuple[Any, Any]]:
        for message in self._record_messages():
            metadata = message.metadata or {}
            if metadata.get(_DELETED_KEY):
                continue
            yield message, self._message_to_record(message)

    def _record_messages(self) -> Iterable[Any]:
        filters = {"metadata": {_KIND_KEY: _MEMORY_KIND}}
        for message in self.session.messages(filters=filters, size=100, reverse=True):
            if (message.metadata or {}).get(_KIND_KEY) == _MEMORY_KIND:
                yield message

    def _message_to_record(self, message: Any) -> Any:
        _require_unified_memory()
        metadata = message.metadata or {}
        return MemoryRecord(  # type: ignore[operator]
            id=metadata[_RECORD_ID_KEY],
            content=message.content,
            scope=metadata.get(_SCOPE_KEY, "/"),
            categories=list(metadata.get(_CATEGORIES_KEY) or []),
            metadata=dict(metadata.get(_MEMORY_METADATA_KEY) or {}),
            importance=metadata.get(_IMPORTANCE_KEY, 0.5),
            created_at=_parse_datetime(
                metadata.get(_CREATED_AT_KEY), message.created_at
            ),
            last_accessed=_parse_datetime(
                metadata.get(_LAST_ACCESSED_KEY), message.created_at
            ),
            embedding=metadata.get(_EMBEDDING_KEY),
            source=metadata.get(_SOURCE_KEY),
            private=bool(metadata.get(_PRIVATE_KEY, False)),
        )

    def _record_matches(
        self,
        record: Any,
        scope_prefix: str | None,
        categories: list[str] | None,
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        return (
            _scope_matches(record.scope, scope_prefix)
            and _category_matches(record.categories, categories)
            and _metadata_matches(record.metadata, metadata_filter)
        )


class HonchoStorage(LegacyStorage):
    """
    Backwards-compatible Honcho storage for CrewAI `ExternalMemory`.

    New CrewAI projects should prefer `HonchoMemoryStorage` with
    `crewai.Memory(storage=...)`.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str | None = None,
        honcho_client: Honcho | None = None,
        assistant_id: str = "assistant",
    ) -> None:
        self.honcho = honcho_client or Honcho()
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.session_id = session_id or str(uuid.uuid4())
        self._user: Any | None = None
        self._assistant: Any | None = None
        self._session: Any | None = None

    @property
    def user(self) -> Any:
        if self._user is None:
            self._user = self.honcho.peer(self.user_id)
        return self._user

    @property
    def assistant(self) -> Any:
        if self._assistant is None:
            self._assistant = self.honcho.peer(self.assistant_id)
        return self._assistant

    @property
    def session(self) -> Any:
        if self._session is None:
            self._session = self.honcho.session(self.session_id)
        return self._session

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """Save a CrewAI external-memory message to a Honcho session."""
        try:
            role = str(metadata.get("role", metadata.get("agent", "assistant"))).lower()
            peer = (
                self.user if role in {"user", "human", self.user_id} else self.assistant
            )
            content = str(value)

            self.session.add_messages([peer.message(content, metadata=metadata)])
            logger.debug("Saved %s message to Honcho session %s", role, self.session_id)

        except Exception:
            logger.exception("Error saving to Honcho")
            raise

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search session messages and return CrewAI external-memory records."""
        try:
            _ = score_threshold
            messages = self.session.search(query=query, filters=filters, limit=limit)
            results = []

            for message in messages:
                metadata = {
                    "peer_id": message.peer_id,
                    "created_at": str(message.created_at)
                    if hasattr(message, "created_at")
                    else None,
                }
                if getattr(message, "metadata", None):
                    metadata.update(message.metadata)

                results.append(
                    {
                        "content": message.content,
                        "memory": message.content,
                        "context": message.content,
                        "metadata": metadata,
                    }
                )

            logger.debug("Search for %r returned %d results", query, len(results))
            return results

        except Exception:
            logger.exception("Error searching Honcho")
            raise

    def reset(self) -> None:
        """Start writing to a fresh Honcho session."""
        self.session_id = str(uuid.uuid4())
        self._session = None
        logger.debug("Reset HonchoStorage to session %s", self.session_id)
