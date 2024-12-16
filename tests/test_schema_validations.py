import pytest
from pydantic import ValidationError
from src.schemas import (
    AppCreate,
    UserCreate,
    MessageCreate,
    MetamessageCreate,
    CollectionCreate,
    DocumentCreate,
    DocumentQuery,
    MessageBatchCreate,
)


class TestAppValidations:
    def test_valid_app_create(self):
        app = AppCreate(name="test", metadata={})
        assert app.name == "test"
        assert app.metadata == {}

    def test_app_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            AppCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_app_name_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            AppCreate(name="a" * 101, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_app_invalid_metadata_type(self):
        with pytest.raises(ValidationError) as exc_info:
            AppCreate(name="test", metadata="not a dict")
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "dict_type"


class TestUserValidations:
    def test_valid_user_create(self):
        user = UserCreate(name="test", metadata={})
        assert user.name == "test"
        assert user.metadata == {}

    def test_user_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_user_name_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(name="a" * 101, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestMessageValidations:
    def test_valid_message_create(self):
        msg = MessageCreate(content="test", is_user=True, metadata={})
        assert msg.content == "test"
        assert msg.is_user is True
        assert msg.metadata == {}

    def test_message_content_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            MessageCreate(content="a" * 50001, is_user=True, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_message_invalid_is_user_type(self):
        with pytest.raises(ValidationError) as exc_info:
            MessageCreate(content="test", is_user="not a bool", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "bool_parsing"


class TestMetamessageValidations:
    def test_valid_metamessage_create(self):
        meta = MetamessageCreate(
            metamessage_type="test",
            content="test content",
            message_id="123",
            metadata={},
        )
        assert meta.metamessage_type == "test"
        assert meta.content == "test content"
        assert meta.message_id == "123"

    def test_metamessage_type_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            MetamessageCreate(
                metamessage_type="",
                content="test",
                message_id="123",
                metadata={},
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_metamessage_type_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            MetamessageCreate(
                metamessage_type="a" * 51,
                content="test",
                message_id="123",
                metadata={},
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_metamessage_content_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            MetamessageCreate(
                metamessage_type="test",
                content="a" * 50001,
                message_id="123",
                metadata={},
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"


class TestCollectionValidations:
    def test_valid_collection_create(self):
        collection = CollectionCreate(name="test", metadata={})
        assert collection.name == "test"
        assert collection.metadata == {}

    def test_collection_name_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            CollectionCreate(name="", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_collection_name_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            CollectionCreate(name="a" * 101, metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_collection_name_honcho(self):
        with pytest.raises(ValidationError) as exc_info:
            CollectionCreate(name="honcho", metadata={})
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "value_error"


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


class TestDocumentQueryValidations:
    def test_valid_document_query(self):
        query = DocumentQuery(query="test query", top_k=5)
        assert query.query == "test query"
        assert query.top_k == 5

    def test_query_too_short(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentQuery(query="", top_k=5)
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_short"

    def test_query_too_long(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentQuery(query="a" * 1001, top_k=5)
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "string_too_long"

    def test_top_k_too_small(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentQuery(query="test", top_k=0)
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "greater_than_equal"

    def test_top_k_too_large(self):
        with pytest.raises(ValidationError) as exc_info:
            DocumentQuery(query="test", top_k=51)
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "less_than_equal"


class TestMessageBatchValidations:
    def test_valid_message_batch(self):
        batch = MessageBatchCreate(
            messages=[
                MessageCreate(content="test", is_user=True, metadata={})
            ]
        )
        assert len(batch.messages) == 1

    def test_message_batch_too_large(self):
        with pytest.raises(ValidationError) as exc_info:
            MessageBatchCreate(
                messages=[
                    MessageCreate(content="test", is_user=True, metadata={})
                    for _ in range(101)
                ]
            )
        error_dict = exc_info.value.errors()[0]
        assert error_dict["type"] == "too_long" 