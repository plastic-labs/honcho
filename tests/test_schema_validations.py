import pytest
from pydantic import ValidationError

from src.schemas import (
    DocumentCreate,
    MessageCreate,
    PeerCreate,
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
        long_content = (
            "a" * 50001  # Increase repetitions to ensure we exceed 8192 tokens
        )

        with pytest.raises(ValidationError) as exc_info:
            MessageCreate(content=long_content, peer_id="12345", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestDocumentValidations:
    def test_valid_document_create(self):
        doc = DocumentCreate(content="test content", metadata={})
        assert doc.content == "test content"
        assert doc.metadata == {}

    def test_document_content_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(content="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_document_content_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentCreate(content="a" * 100001, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"
