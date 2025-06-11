"""Tests for the TOM embeddings module and CollectionEmbeddingStore."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src import schemas
from src.deriver.tom.embeddings import CollectionEmbeddingStore


class TestCollectionEmbeddingStoreInitialization:
    """Test CollectionEmbeddingStore initialization."""

    def test_initialization(self):
        """Test basic initialization of CollectionEmbeddingStore."""
        app_id = str(uuid4())
        user_id = str(uuid4())
        collection_id = str(uuid4())

        store = CollectionEmbeddingStore(app_id, user_id, collection_id)

        assert store.app_id == app_id
        assert store.user_id == user_id
        assert store.collection_id == collection_id


class TestSaveFacts:
    """Test fact saving functionality."""

    @pytest_asyncio.fixture
    async def embedding_store(self, sample_data):
        """Create an embedding store for testing."""
        test_app, test_user = sample_data
        collection_id = str(uuid4())
        return CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)

    @pytest.mark.asyncio
    async def test_save_facts_basic(self, embedding_store):
        """Test basic fact saving functionality."""
        facts = [
            "User is a Python developer",
            "User works remotely",
            "User loves coffee"
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                mock_create_doc.return_value = None

                await embedding_store.save_facts(facts)

                # Should create a document for each fact
                assert mock_create_doc.call_count == 3
                
                # Verify each call
                for i, fact in enumerate(facts):
                    call_args = mock_create_doc.call_args_list[i]
                    assert call_args[1]["app_id"] == embedding_store.app_id
                    assert call_args[1]["user_id"] == embedding_store.user_id
                    assert call_args[1]["collection_id"] == embedding_store.collection_id
                    assert call_args[1]["document"].content == fact
                    assert abs(call_args[1]["duplicate_threshold"] - 0.15) < 1e-10  # 1 - 0.85

    @pytest.mark.asyncio
    async def test_save_facts_with_message_id(self, embedding_store):
        """Test saving facts with message ID metadata."""
        facts = ["User prefers PyTorch over TensorFlow"]
        message_id = str(uuid4())

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                mock_create_doc.return_value = None

                await embedding_store.save_facts(facts, message_id=message_id)

                # Verify message_id is included in metadata
                call_args = mock_create_doc.call_args_list[0]
                document = call_args[1]["document"]
                assert document.metadata == {"message_id": message_id}

    @pytest.mark.asyncio
    async def test_save_facts_custom_similarity_threshold(self, embedding_store):
        """Test saving facts with custom similarity threshold."""
        facts = ["User enjoys debugging"]
        similarity_threshold = 0.9

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                mock_create_doc.return_value = None

                await embedding_store.save_facts(facts, similarity_threshold=similarity_threshold)

                # Verify duplicate threshold is calculated correctly
                call_args = mock_create_doc.call_args_list[0]
                assert abs(call_args[1]["duplicate_threshold"] - 0.1) < 1e-10  # 1 - 0.9

    @pytest.mark.asyncio
    async def test_save_facts_handles_document_creation_error(self, embedding_store):
        """Test that fact saving handles document creation errors gracefully."""
        facts = [
            "User is a Python developer",
            "This fact will fail",
            "User works remotely"
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                # Mock second call to raise an exception
                def mock_create_side_effect(*args, **kwargs):
                    if "This fact will fail" in str(kwargs.get("document", "")):
                        raise Exception("Database error")
                    return None

                mock_create_doc.side_effect = mock_create_side_effect

                # Should not raise exception (errors are handled gracefully)
                await embedding_store.save_facts(facts)

                # Should still attempt to create all documents
                assert mock_create_doc.call_count == 3

    @pytest.mark.asyncio
    async def test_save_facts_empty_list(self, embedding_store):
        """Test saving empty fact list."""
        facts = []

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                await embedding_store.save_facts(facts)

                # Should not call create_document
                mock_create_doc.assert_not_called()


class TestGetRelevantFacts:
    """Test fact retrieval functionality."""

    @pytest_asyncio.fixture
    async def embedding_store(self, sample_data):
        """Create an embedding store for testing."""
        test_app, test_user = sample_data
        collection_id = str(uuid4())
        return CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)

    @pytest.mark.asyncio
    async def test_get_relevant_facts_basic(self, embedding_store):
        """Test basic fact retrieval functionality."""
        query = "What programming languages does the user know?"
        
        # Mock documents returned from query
        mock_documents = [
            MagicMock(content="User is proficient in Python"),
            MagicMock(content="User has experience with JavaScript"),
            MagicMock(content="User knows SQL")
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                mock_query_docs.return_value = mock_documents

                facts = await embedding_store.get_relevant_facts(query)

                # Verify query parameters
                mock_query_docs.assert_called_once_with(
                    mock_db,
                    app_id=embedding_store.app_id,
                    user_id=embedding_store.user_id,
                    collection_id=embedding_store.collection_id,
                    query=query,
                    max_distance=0.3,
                    top_k=5
                )

                # Verify returned facts
                expected_facts = [
                    "User is proficient in Python",
                    "User has experience with JavaScript", 
                    "User knows SQL"
                ]
                assert facts == expected_facts

    @pytest.mark.asyncio
    async def test_get_relevant_facts_custom_parameters(self, embedding_store):
        """Test fact retrieval with custom parameters."""
        query = "What does the user do for work?"
        top_k = 10
        max_distance = 0.2

        mock_documents = [MagicMock(content="User is a software engineer")]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                mock_query_docs.return_value = mock_documents

                facts = await embedding_store.get_relevant_facts(
                    query, top_k=top_k, max_distance=max_distance
                )

                # Verify custom parameters were used
                mock_query_docs.assert_called_once_with(
                    mock_db,
                    app_id=embedding_store.app_id,
                    user_id=embedding_store.user_id,
                    collection_id=embedding_store.collection_id,
                    query=query,
                    max_distance=max_distance,
                    top_k=top_k
                )

    @pytest.mark.asyncio
    async def test_get_relevant_facts_no_results(self, embedding_store):
        """Test fact retrieval when no relevant facts are found."""
        query = "What is the user's favorite food?"

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                mock_query_docs.return_value = []  # No results

                facts = await embedding_store.get_relevant_facts(query)

                assert facts == []

    @pytest.mark.asyncio
    async def test_get_relevant_facts_empty_query(self, embedding_store):
        """Test fact retrieval with empty query."""
        query = ""

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                mock_query_docs.return_value = []

                facts = await embedding_store.get_relevant_facts(query)

                # Should still call query_documents with empty query
                mock_query_docs.assert_called_once()
                assert facts == []


class TestRemoveDuplicates:
    """Test duplicate detection and removal functionality."""

    @pytest_asyncio.fixture
    async def embedding_store(self, sample_data):
        """Create an embedding store for testing."""
        test_app, test_user = sample_data
        collection_id = str(uuid4())
        return CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)

    @pytest.mark.asyncio
    async def test_remove_duplicates_no_duplicates(self, embedding_store):
        """Test duplicate removal when no duplicates exist."""
        facts = [
            "User is a Python developer",
            "User works remotely",
            "User loves coffee"
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                mock_get_dupes.return_value = []  # No duplicates

                unique_facts = await embedding_store.remove_duplicates(facts)

                # All facts should be considered unique
                assert unique_facts == facts
                
                # Should check each fact for duplicates
                assert mock_get_dupes.call_count == 3

    @pytest.mark.asyncio
    async def test_remove_duplicates_with_duplicates(self, embedding_store):
        """Test duplicate removal when duplicates exist."""
        facts = [
            "User is a Python developer",
            "User codes in Python",  # Similar to first fact
            "User works remotely"
        ]

        # Mock duplicate document for second fact
        mock_duplicate_doc = MagicMock()
        mock_duplicate_doc.content = "User is a Python programmer"

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                def mock_get_dupes_side_effect(db, app_id, user_id, collection_id, content, similarity_threshold):
                    if "codes in Python" in content:
                        return [mock_duplicate_doc]  # Duplicate found
                    return []  # No duplicates

                mock_get_dupes.side_effect = mock_get_dupes_side_effect

                unique_facts = await embedding_store.remove_duplicates(facts)

                # Should remove the duplicate fact
                expected_unique = [
                    "User is a Python developer",
                    "User works remotely"
                ]
                assert unique_facts == expected_unique

    @pytest.mark.asyncio
    async def test_remove_duplicates_custom_threshold(self, embedding_store):
        """Test duplicate removal with custom similarity threshold."""
        facts = ["User enjoys programming"]
        similarity_threshold = 0.9

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                mock_get_dupes.return_value = []

                await embedding_store.remove_duplicates(facts, similarity_threshold=similarity_threshold)

                # Verify custom threshold was passed
                mock_get_dupes.assert_called_once_with(
                    mock_db,
                    app_id=embedding_store.app_id,
                    user_id=embedding_store.user_id,
                    collection_id=embedding_store.collection_id,
                    content=facts[0],
                    similarity_threshold=similarity_threshold
                )

    @pytest.mark.asyncio
    async def test_remove_duplicates_handles_errors(self, embedding_store):
        """Test that duplicate checking handles errors gracefully."""
        facts = [
            "User is a Python developer",
            "This fact will cause an error",
            "User works remotely"
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                def mock_get_dupes_side_effect(*args, **kwargs):
                    if "cause an error" in kwargs.get("content", ""):
                        raise Exception("Database connection error")
                    return []

                mock_get_dupes.side_effect = mock_get_dupes_side_effect

                unique_facts = await embedding_store.remove_duplicates(facts)

                # Should include all facts (error results in keeping the fact)
                assert unique_facts == facts
                assert mock_get_dupes.call_count == 3

    @pytest.mark.asyncio
    async def test_remove_duplicates_empty_list(self, embedding_store):
        """Test duplicate removal with empty fact list."""
        facts = []

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                unique_facts = await embedding_store.remove_duplicates(facts)

                assert unique_facts == []
                mock_get_dupes.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_duplicates_logs_duplicate_found(self, embedding_store):
        """Test that duplicate detection logs when duplicates are found."""
        facts = ["User loves Python programming"]
        
        mock_duplicate_doc = MagicMock()
        mock_duplicate_doc.content = "User enjoys Python development"

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                mock_get_dupes.return_value = [mock_duplicate_doc]

                with patch("src.deriver.tom.embeddings.logger.debug") as mock_log:
                    unique_facts = await embedding_store.remove_duplicates(facts)

                    # Should log the duplicate detection
                    mock_log.assert_called_once()
                    log_message = mock_log.call_args[0][0]
                    assert "Duplicate found" in log_message
                    assert mock_duplicate_doc.content in log_message
                    assert facts[0] in log_message

                    # Should not include the duplicate fact
                    assert unique_facts == []


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple operations."""

    @pytest_asyncio.fixture
    async def embedding_store(self, sample_data):
        """Create an embedding store for testing."""
        test_app, test_user = sample_data
        collection_id = str(uuid4())
        return CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)

    @pytest.mark.asyncio
    async def test_full_workflow_save_and_retrieve(self, embedding_store):
        """Test complete workflow of saving facts and retrieving them."""
        # First save some facts
        facts_to_save = [
            "User is a senior Python developer",
            "User has 5 years of experience with FastAPI",
            "User prefers async programming"
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                mock_create_doc.return_value = None

                await embedding_store.save_facts(facts_to_save)

                # Verify all facts were saved
                assert mock_create_doc.call_count == 3

        # Then retrieve relevant facts
        query = "What is the user's programming experience?"
        mock_documents = [
            MagicMock(content="User is a senior Python developer"),
            MagicMock(content="User has 5 years of experience with FastAPI")
        ]

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                mock_query_docs.return_value = mock_documents

                retrieved_facts = await embedding_store.get_relevant_facts(query)

                expected_facts = [
                    "User is a senior Python developer",
                    "User has 5 years of experience with FastAPI"
                ]
                assert retrieved_facts == expected_facts

    @pytest.mark.asyncio
    async def test_duplicate_removal_before_saving(self, embedding_store):
        """Test the typical workflow of removing duplicates before saving."""
        facts_to_check = [
            "User is a Python developer",
            "User writes code in Python",  # Potential duplicate
            "User works from home"
        ]

        # Mock existing duplicate
        mock_duplicate_doc = MagicMock()
        mock_duplicate_doc.content = "User is proficient in Python"

        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes:
                def mock_get_dupes_side_effect(*args, **kwargs):
                    if "writes code in Python" in kwargs.get("content", ""):
                        return [mock_duplicate_doc]
                    return []

                mock_get_dupes.side_effect = mock_get_dupes_side_effect

                # Remove duplicates
                unique_facts = await embedding_store.remove_duplicates(facts_to_check)

                expected_unique = [
                    "User is a Python developer",
                    "User works from home"
                ]
                assert unique_facts == expected_unique

        # Now save the unique facts
        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                mock_create_doc.return_value = None

                await embedding_store.save_facts(unique_facts)

                # Should only save the unique facts
                assert mock_create_doc.call_count == 2

    @pytest.mark.asyncio
    async def test_error_recovery_in_workflow(self, embedding_store):
        """Test error recovery across multiple operations."""
        facts = [
            "User is experienced with machine learning",
            "User uses scikit-learn and pandas"
        ]

        # Test save_facts with partial failure
        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc:
                def mock_create_side_effect(*args, **kwargs):
                    if "scikit-learn" in str(kwargs.get("document", "")):
                        raise Exception("Database error")
                    return None

                mock_create_doc.side_effect = mock_create_side_effect

                # Should handle the error gracefully
                await embedding_store.save_facts(facts)

                # Should attempt to save both facts
                assert mock_create_doc.call_count == 2

        # Test get_relevant_facts after partial save
        with patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db:
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None

            with patch("src.deriver.tom.embeddings.crud.query_documents") as mock_query_docs:
                # Only return the successfully saved fact
                mock_query_docs.return_value = [
                    MagicMock(content="User is experienced with machine learning")
                ]

                retrieved_facts = await embedding_store.get_relevant_facts("machine learning")

                assert retrieved_facts == ["User is experienced with machine learning"]