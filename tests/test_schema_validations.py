from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import (
    ConclusionCreate,
    DocumentCreate,
    DocumentMetadata,
    MemoryExpiry,
    MemoryTaxonomy,
    MessageCreate,
    ObservationInput,
    PeerCreate,
    ResolvedConfiguration,
    SessionCreate,
    WorkspaceCreate,
)


class TestWorkspaceValidations:
    def test_valid_workspace_create(self):
        workspace = WorkspaceCreate(name="test", metadata={})
        assert workspace.name == "test"
        assert workspace.metadata == {}

    def test_app_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_app_name_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(name="a" * 101, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_app_invalid_metadata_type(self):
        with pytest.raises(ValidationError) as exc_info:
            WorkspaceCreate(
                name="test",
                metadata="not a dict",  # pyright: ignore
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "dict_type"


class TestPeerValidations:
    def test_valid_peer_create(self):
        peer = PeerCreate(name="test", metadata={})
        assert peer.name == "test"
        assert peer.metadata == {}

    def test_valid_peer_create_configuration(self):
        peer = PeerCreate(name="test", metadata={}, configuration={"test": True})
        assert peer.name == "test"
        assert peer.metadata == {}
        assert peer.configuration == {"test": True}

    def test_peer_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            PeerCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_peer_name_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            PeerCreate(name="a" * 101, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestSessionValidations:
    def test_valid_session_create(self):
        session = SessionCreate(name="test", metadata={})
        assert session.name == "test"
        assert session.metadata == {}

    def test_session_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            SessionCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"


class TestMessageValidations:
    def test_valid_message_create(self):
        msg = MessageCreate(content="test", peer_id="12345", metadata={})
        assert msg.content == "test"
        assert msg.peer_name == "12345"
        assert msg.metadata == {}

    def test_message_content_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            MessageCreate(content="a" * 50001, peer_id="12345", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestDocumentValidations:
    def test_valid_document_create(self):
        metadata = DocumentMetadata(
            message_ids=[1],
            premises=[],
            message_created_at="2021-01-01T00:00:00Z",
            memory=MemoryTaxonomy(
                domain="project:decision",
                horizon="medium",
                thesis_kind="decision",
                expiry=MemoryExpiry(type="none"),
            ),
        )
        doc = DocumentCreate(
            content="test content",
            session_name="test",
            level="explicit",
            metadata=metadata,
            embedding=[0.1, 0.2, 0.3],
        )
        assert doc.content == "test content"
        assert doc.level == "explicit"
        assert doc.metadata == metadata

    def test_document_content_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(
                content="",
                session_name="test",
                level="explicit",
                metadata=DocumentMetadata(
                    message_ids=[1],
                    premises=[],
                    message_created_at="2021-01-01T00:00:00Z",
                ),
                embedding=[0.1, 0.2, 0.3],
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_document_content_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(
                content="a" * 100001,
                session_name="test",
                level="explicit",
                metadata=DocumentMetadata(
                    message_ids=[1],
                    premises=[],
                    message_created_at="2021-01-01T00:00:00Z",
                ),
                embedding=[0.1, 0.2, 0.3],
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestMemoryTaxonomyValidations:
    def test_observation_input_accepts_memory_taxonomy(self):
        observation = ObservationInput.model_validate(
            {
                "content": "User prefers concise responses.",
                "level": "explicit",
                "memory": {
                    "domain": "user:preferences",
                    "horizon": "long",
                    "thesis_kind": "preference",
                    "expiry": {"type": "none"},
                },
            }
        )

        assert observation.memory is not None
        assert observation.memory.domain == "user:preferences"
        assert observation.memory.horizon == "long"
        assert observation.memory.thesis_kind == "preference"

    def test_memory_expiry_requires_review_at_for_review_type(self):
        with pytest.raises(ValidationError):
            MemoryExpiry(type="review")

    def test_conclusion_create_accepts_memory_taxonomy(self):
        conclusion = ConclusionCreate.model_validate(
            {
                "content": "Use the canonical LLM Wiki path.",
                "observer_id": "observer-a",
                "observed_id": "observed-b",
                "session_id": "session-1",
                "memory": {
                    "domain": "workspace:rule",
                    "horizon": "long",
                    "thesis_kind": "rule",
                    "expiry": {"type": "none"},
                },
            }
        )

        assert conclusion.memory is not None
        memory = MemoryTaxonomy.model_validate(conclusion.memory)
        assert memory.domain == "workspace:rule"
        assert memory.thesis_kind == "rule"

    def test_conclusion_create_accepts_lifecycle_native_fields(self):
        conclusion = ConclusionCreate.model_validate(
            {
                "content": "The workspace now uses the Windows-visible wiki path.",
                "observer_id": "observer-a",
                "observed_id": "observed-b",
                "memory": {
                    "domain": "workspace:rule",
                    "horizon": "long",
                    "thesis_kind": "rule",
                    "expiry": {"type": "event", "event_key": "workspace.reconfigured"},
                    "lifecycle": {
                        "review_due_at": "2026-05-01T00:00:00Z",
                        "pending_event_key": "workspace.reconfigured",
                        "supersedes": ["old-rule-1"],
                        "superseded_by": "new-rule-2",
                        "demote_after": "2026-06-01T00:00:00Z",
                    },
                },
            }
        )

        memory = MemoryTaxonomy.model_validate(conclusion.memory)
        assert memory.lifecycle is not None
        assert memory.lifecycle.review_due_at == "2026-05-01T00:00:00Z"
        assert memory.lifecycle.pending_event_key == "workspace.reconfigured"
        assert memory.lifecycle.supersedes == ["old-rule-1"]
        assert memory.lifecycle.superseded_by == "new-rule-2"

    def test_conclusion_create_rejects_invalid_lifecycle_supersedes(self):
        with pytest.raises(ValidationError):
            MemoryTaxonomy.model_validate(
                {
                    "domain": "workspace:rule",
                    "horizon": "long",
                    "thesis_kind": "rule",
                    "expiry": {"type": "none"},
                    "lifecycle": {
                        "supersedes": [],
                    },
                }
            )


class TestResolvedConfigurationMigration:
    """Test backward compatibility for queue items created before v3.0.0.

    In v3.0.0, the 'deriver' field was renamed to 'reasoning'. Old queue items
    may still have the 'deriver' field and need to be migrated at validation time.
    """

    def _make_config(self, **overrides: dict[str, Any]):
        """Helper to create a valid config dict with overrides."""
        base = {
            "reasoning": {"enabled": True},
            "peer_card": {"use": True, "create": True},
            "summary": {
                "enabled": True,
                "messages_per_short_summary": 20,
                "messages_per_long_summary": 60,
            },
            "dream": {"enabled": False},
        }
        base.update(overrides)
        return base

    def test_old_queue_item_with_deriver_field(self):
        """Old queue items with 'deriver' should be migrated to 'reasoning'."""
        old_payload = self._make_config()
        del old_payload["reasoning"]
        old_payload["deriver"] = {"enabled": True}

        config = ResolvedConfiguration.model_validate(old_payload)

        assert config.reasoning.enabled is True

    def test_new_queue_item_with_reasoning_field(self):
        """New queue items with 'reasoning' should work normally."""
        new_payload = self._make_config(reasoning={"enabled": False})

        config = ResolvedConfiguration.model_validate(new_payload)

        assert config.reasoning.enabled is False

    def test_migration_does_not_override_reasoning(self):
        """If both 'deriver' and 'reasoning' exist, 'reasoning' takes precedence."""
        payload = self._make_config(reasoning={"enabled": False})
        payload["deriver"] = {"enabled": True}

        config = ResolvedConfiguration.model_validate(payload)

        assert config.reasoning.enabled is False

    def test_missing_reasoning_and_deriver_fails(self):
        """Payload missing both 'reasoning' and 'deriver' should fail validation."""
        payload = self._make_config()
        del payload["reasoning"]

        with pytest.raises(ValidationError) as exc_info:
            ResolvedConfiguration.model_validate(payload)

        assert any(e["loc"] == ("reasoning",) for e in exc_info.value.errors())
