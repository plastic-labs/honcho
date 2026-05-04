"""
Tests for Honcho CrewAI storage adapters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from crewai.memory.types import MemoryRecord
from honcho_crewai import HonchoMemoryStorage, HonchoStorage


class FakeMessageCreate:
    def __init__(self, peer_id, content, metadata=None, created_at=None):
        self.peer_id = peer_id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at


class FakeMessage:
    def __init__(self, id, peer_id, content, metadata=None, created_at=None):
        self.id = id
        self.peer_id = peer_id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(UTC)


class FakePeer:
    def __init__(self, id):
        self.id = id

    def message(self, content, *, metadata=None, created_at=None):
        return FakeMessageCreate(self.id, content, metadata, created_at)

    def chat(self, query, **kwargs):
        return f"answer: {query} {kwargs}"


class FakeSession:
    def __init__(self, id):
        self.id = id
        self._messages = []

    def add_messages(self, messages):
        saved = []
        for message in messages:
            saved_message = FakeMessage(
                id=f"msg-{len(self._messages) + 1}",
                peer_id=message.peer_id,
                content=message.content,
                metadata=dict(message.metadata),
                created_at=message.created_at,
            )
            self._messages.append(saved_message)
            saved.append(saved_message)
        return saved

    def search(self, query, filters=None, limit=10):
        return self._messages[:limit]

    def messages(self, filters=None, size=100, reverse=False):
        messages = list(self._messages)
        if reverse:
            messages.reverse()
        return messages

    def update_message(self, message, metadata):
        message.metadata = metadata
        return message

    def context(self, **kwargs):
        return type(
            "FakeContext",
            (),
            {
                "summary": None,
                "peer_representation": None,
                "peer_card": None,
                "messages": self._messages,
            },
        )()


class FakeHoncho:
    def __init__(self):
        self.peer_calls = []
        self.session_calls = []
        self._peers = {}
        self._sessions = {}

    def peer(self, id):
        self.peer_calls.append(id)
        self._peers.setdefault(id, FakePeer(id))
        return self._peers[id]

    def session(self, id):
        self.session_calls.append(id)
        self._sessions.setdefault(id, FakeSession(id))
        return self._sessions[id]


class TestHonchoMemoryStorage:
    def test_initialization_is_lazy(self):
        honcho = FakeHoncho()

        storage = HonchoMemoryStorage(
            peer_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )

        assert storage.session_id == "session-1"
        assert storage.peer_id == "user-1"
        assert honcho.peer_calls == []
        assert honcho.session_calls == []

    def test_save_and_search_memory_records(self):
        honcho = FakeHoncho()
        storage = HonchoMemoryStorage(
            peer_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )
        record = MemoryRecord(
            id="record-1",
            content="User likes ramen",
            scope="/users/user-1",
            categories=["preferences"],
            metadata={"topic": "food"},
            embedding=[1.0, 0.0],
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
        )

        storage.save([record])
        matches = storage.search(
            [1.0, 0.0],
            scope_prefix="/users",
            categories=["preferences"],
            metadata_filter={"topic": "food"},
        )

        assert len(matches) == 1
        assert matches[0][0].id == "record-1"
        assert matches[0][1] == 1.0
        assert honcho.peer_calls == ["user-1"]
        assert honcho.session_calls == ["session-1"]

    def test_delete_update_and_discovery_methods(self):
        honcho = FakeHoncho()
        storage = HonchoMemoryStorage(
            peer_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )
        old_record = MemoryRecord(
            id="record-1",
            content="Old preference",
            scope="/users/user-1/preferences",
            categories=["preferences"],
            metadata={"topic": "food"},
            embedding=[1.0, 0.0],
            created_at=datetime.now(UTC) - timedelta(days=1),
            last_accessed=datetime.now(UTC) - timedelta(days=1),
        )
        new_record = MemoryRecord(
            id="record-1",
            content="New preference",
            scope="/users/user-1/preferences",
            categories=["preferences"],
            metadata={"topic": "food"},
            embedding=[0.0, 1.0],
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
        )

        storage.save([old_record])
        storage.update(new_record)

        assert storage.get_record("record-1").content == "New preference"
        assert storage.count("/users") == 1
        assert storage.list_categories("/users") == {"preferences": 1}
        assert storage.list_scopes("/") == ["/users"]
        assert storage.get_scope_info("/users").record_count == 1

        assert storage.delete(record_ids=["record-1"]) == 1
        assert storage.get_record("record-1") is None


class TestHonchoStorage:
    def test_legacy_initialization_is_lazy(self):
        honcho = FakeHoncho()

        storage = HonchoStorage(
            user_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )

        assert storage.session_id == "session-1"
        assert honcho.peer_calls == []
        assert honcho.session_calls == []

    def test_legacy_save_maps_roles_to_peers(self):
        honcho = FakeHoncho()
        storage = HonchoStorage(
            user_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )

        storage.save("User message", metadata={"role": "user"})
        storage.save("Assistant message", metadata={"role": "assistant"})

        messages = honcho._sessions["session-1"]._messages
        assert [message.peer_id for message in messages] == ["user-1", "assistant"]

    def test_legacy_search_returns_crewai_external_memory_format(self):
        honcho = FakeHoncho()
        storage = HonchoStorage(
            user_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )

        storage.save("User likes ramen", metadata={"role": "user", "topic": "food"})
        results = storage.search("ramen")

        assert results == [
            {
                "content": "User likes ramen",
                "memory": "User likes ramen",
                "context": "User likes ramen",
                "metadata": {
                    "peer_id": "user-1",
                    "created_at": str(results[0]["metadata"]["created_at"]),
                    "role": "user",
                    "topic": "food",
                },
            }
        ]

    def test_legacy_reset_is_lazy(self):
        honcho = FakeHoncho()
        storage = HonchoStorage(
            user_id="user-1",
            session_id="session-1",
            honcho_client=honcho,
        )

        storage.reset()

        assert storage.session_id != "session-1"
        assert honcho.session_calls == []
