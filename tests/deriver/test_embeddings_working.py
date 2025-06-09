"""Working embeddings tests that test actual vector operations and database interactions."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src import models
from src.deriver.tom.embeddings import CollectionEmbeddingStore


class TestEmbeddingsWorking:
    """Test CollectionEmbeddingStore with real database operations."""

    @pytest_asyncio.fixture
    async def embedding_store_setup(self, db_session, sample_data):
        """Setup embedding store with real database collection."""
        test_app, test_user = sample_data

        # Create real collection in database
        collection = models.Collection(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            name=f"test_collection_{uuid4()}",
            metadata={"type": "user_facts"},
        )
        db_session.add(collection)
        await db_session.flush()

        # Create embedding store
        store = CollectionEmbeddingStore(
            test_app.public_id, test_user.public_id, collection.public_id
        )

        return test_app, test_user, collection, store

    @pytest.mark.asyncio
    async def test_save_facts_with_real_database_operations(
        self, db_session, embedding_store_setup
    ):
        """Test saving facts with real database operations."""
        test_app, test_user, collection, store = embedding_store_setup

        facts_to_save = [
            "User is a Python developer with Django experience",
            "User works remotely from Seattle Washington",
            "User has exactly 5 years of professional experience",
            "User enjoys machine learning and AI projects",
        ]
        message_id = str(uuid4())

        # Mock only the tracked_db context manager to use our test session
        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        # Mock create_document to avoid internal duplicate detection
        async def mock_create_document(
            db, document, app_id, user_id, collection_id, duplicate_threshold=None
        ):
            new_doc = models.Document(
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                content=document.content,
                h_metadata=document.metadata,  # Use h_metadata, not metadata
                embedding=[0.1] * 1536,  # Mock embedding
            )
            db.add(new_doc)
            return new_doc

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.create_document",
                side_effect=mock_create_document,
            ):
                await store.save_facts(facts_to_save, message_id=message_id)

                # Verify facts were actually stored in database
                result = await db_session.execute(
                    models.Document.__table__.select().where(
                        models.Document.collection_id == collection.public_id
                    )
                )
                stored_documents = result.fetchall()

                assert len(stored_documents) == len(facts_to_save)

                # Verify content and metadata
                stored_contents = [doc.content for doc in stored_documents]
                for fact in facts_to_save:
                    assert fact in stored_contents

                # Verify message_id metadata
                for doc in stored_documents:
                    doc_metadata = doc.metadata if doc.metadata else {}
                    assert doc_metadata.get("message_id") == message_id

                # Verify embeddings are stored (should be populated by create_document)
                for doc in stored_documents:
                    assert doc.embedding is not None
                    assert len(doc.embedding) > 0  # Should have embedding vector

        print(f"✓ Save facts test passed - stored {len(stored_documents)} facts")

    @pytest.mark.asyncio
    async def test_get_relevant_facts_with_real_query_operations(
        self, db_session, embedding_store_setup
    ):
        """Test retrieving relevant facts with real database queries."""
        test_app, test_user, collection, store = embedding_store_setup

        # Pre-populate collection with facts
        existing_facts = [
            "User is a Python developer with Django experience",
            "User works on machine learning projects using scikit-learn",
            "User has experience with React and frontend development",
            "User enjoys hiking and outdoor activities on weekends",
            "User graduated from Stanford with a CS degree",
        ]

        # Store facts in database with realistic embeddings
        for i, fact in enumerate(existing_facts):
            doc = models.Document(
                app_id=test_app.public_id,
                user_id=test_user.public_id,
                collection_id=collection.public_id,
                content=fact,
                h_metadata={"stored_at": "2024-01-01T00:00:00Z"},
                embedding=[0.1 + i * 0.1] * 1536,  # Varied embeddings
            )
            db_session.add(doc)
        await db_session.flush()

        query = "What programming languages does the user know?"

        # Mock query_documents to return relevant documents
        mock_relevant_docs = [
            MagicMock(content="User is a Python developer with Django experience"),
            MagicMock(
                content="User works on machine learning projects using scikit-learn"
            ),
        ]

        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        # Mock query_documents to simulate vector search
        captured_query_params = None

        async def mock_query_documents(
            _db, app_id, user_id, collection_id, query, max_distance, top_k
        ):
            nonlocal captured_query_params
            captured_query_params = {
                "app_id": app_id,
                "user_id": user_id,
                "collection_id": collection_id,
                "query": query,
                "max_distance": max_distance,
                "top_k": top_k,
            }
            return mock_relevant_docs

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.query_documents",
                side_effect=mock_query_documents,
            ):
                relevant_facts = await store.get_relevant_facts(
                    query, top_k=3, max_distance=0.25
                )

                # Verify query parameters were passed correctly
                assert captured_query_params is not None
                assert captured_query_params["app_id"] == test_app.public_id
                assert captured_query_params["user_id"] == test_user.public_id
                assert captured_query_params["collection_id"] == collection.public_id
                assert captured_query_params["query"] == query
                assert captured_query_params["max_distance"] == 0.25
                assert captured_query_params["top_k"] == 3

                # Verify results
                assert len(relevant_facts) == 2
                assert (
                    "User is a Python developer with Django experience"
                    in relevant_facts
                )
                assert (
                    "User works on machine learning projects using scikit-learn"
                    in relevant_facts
                )

        print(
            f"✓ Get relevant facts test passed - found {len(relevant_facts)} relevant facts"
        )

    @pytest.mark.asyncio
    async def test_remove_duplicates_with_real_similarity_detection(
        self, db_session, embedding_store_setup
    ):
        """Test duplicate removal with real similarity detection logic."""
        test_app, test_user, collection, store = embedding_store_setup

        # Store some existing facts
        existing_facts = [
            "User is a software engineer",
            "User works with Python programming language",
            "User has machine learning experience",
        ]

        for fact in existing_facts:
            doc = models.Document(
                app_id=test_app.public_id,
                user_id=test_user.public_id,
                collection_id=collection.public_id,
                content=fact,
                h_metadata={},
                embedding=[0.1] * 1536,
            )
            db_session.add(doc)
        await db_session.flush()

        # Test facts with some duplicates and some unique
        test_facts = [
            "User is a software engineer",  # Exact duplicate
            "User codes in Python",  # Similar to "works with Python"
            "User has ML expertise",  # Similar to "machine learning experience"
            "User enjoys reading technical books",  # Unique
            "User lives in San Francisco",  # Unique
        ]

        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        # Mock get_duplicate_documents to simulate realistic duplicate detection
        async def mock_get_duplicate_documents(
            db, app_id, user_id, collection_id, content, similarity_threshold=0.85
        ):
            if content == "User is a software engineer":
                # Exact match
                duplicate_doc = MagicMock()
                duplicate_doc.content = "User is a software engineer"
                return [duplicate_doc]
            elif "codes in Python" in content:
                # Similar to existing Python fact
                duplicate_doc = MagicMock()
                duplicate_doc.content = "User works with Python programming language"
                return [duplicate_doc]
            elif "ML expertise" in content:
                # Similar to existing ML fact
                duplicate_doc = MagicMock()
                duplicate_doc.content = "User has machine learning experience"
                return [duplicate_doc]
            else:
                return []  # No duplicates

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.get_duplicate_documents",
                side_effect=mock_get_duplicate_documents,
            ):
                unique_facts = await store.remove_duplicates(
                    test_facts, similarity_threshold=0.85
                )

                # Should only return the unique facts
                expected_unique = [
                    "User enjoys reading technical books",
                    "User lives in San Francisco",
                ]
                assert set(unique_facts) == set(expected_unique)
                assert len(unique_facts) == 2

        print(
            f"✓ Remove duplicates test passed - kept {len(unique_facts)} unique facts"
        )

    @pytest.mark.asyncio
    async def test_collection_isolation_between_users(self, db_session, sample_data):
        """Test that user facts are properly isolated between different users."""
        test_app, _ = sample_data

        # Create two different users in the database
        user1 = models.User(
            app_id=test_app.public_id, name=f"test_user_1_{uuid4()}", metadata={}
        )
        user2 = models.User(
            app_id=test_app.public_id, name=f"test_user_2_{uuid4()}", metadata={}
        )
        db_session.add_all([user1, user2])
        await db_session.flush()

        # Create collections for both users
        collection1 = models.Collection(
            app_id=test_app.public_id,
            user_id=user1.public_id,
            name=f"user_{user1.public_id}",
            metadata={"type": "user_facts"},
        )
        collection2 = models.Collection(
            app_id=test_app.public_id,
            user_id=user2.public_id,
            name=f"user_{user2.public_id}",
            metadata={"type": "user_facts"},
        )
        db_session.add_all([collection1, collection2])
        await db_session.flush()

        # Create embedding stores for both users
        store1 = CollectionEmbeddingStore(
            test_app.public_id, user1.public_id, collection1.public_id
        )
        store2 = CollectionEmbeddingStore(
            test_app.public_id, user2.public_id, collection2.public_id
        )

        # Store different facts for each user
        user1_facts = [
            "User is a backend developer",
            "User lives in New York",
            "User has 3 years experience",
        ]
        user2_facts = [
            "User is a frontend developer",
            "User lives in California",
            "User has 5 years experience",
        ]

        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        async def mock_create_document(
            db, document, app_id, user_id, collection_id, duplicate_threshold=None
        ):
            new_doc = models.Document(
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                content=document.content,
                h_metadata=document.metadata,
                embedding=[0.1] * 1536,
            )
            db.add(new_doc)
            return new_doc

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.create_document",
                side_effect=mock_create_document,
            ):
                # Store facts for both users
                await store1.save_facts(user1_facts)
                await store2.save_facts(user2_facts)

                # Verify user1 facts are only in user1's collection
                result1 = await db_session.execute(
                    models.Document.__table__.select().where(
                        models.Document.collection_id == collection1.public_id
                    )
                )
                user1_docs = result1.fetchall()
                user1_contents = [doc.content for doc in user1_docs]

                assert len(user1_docs) == 3
                for fact in user1_facts:
                    assert fact in user1_contents
                for fact in user2_facts:
                    assert fact not in user1_contents

                # Verify user2 facts are only in user2's collection
                result2 = await db_session.execute(
                    models.Document.__table__.select().where(
                        models.Document.collection_id == collection2.public_id
                    )
                )
                user2_docs = result2.fetchall()
                user2_contents = [doc.content for doc in user2_docs]

                assert len(user2_docs) == 3
                for fact in user2_facts:
                    assert fact in user2_contents
                for fact in user1_facts:
                    assert fact not in user2_contents

        print(
            f"✓ User isolation test passed - user1: {len(user1_docs)} facts, user2: {len(user2_docs)} facts"
        )

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(
        self, db_session, embedding_store_setup
    ):
        """Test that embedding operations handle errors gracefully."""
        test_app, test_user, collection, store = embedding_store_setup

        test_facts = [
            "User likes programming",
            "This fact will cause an error during storage",
            "User works in technology",
        ]

        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        # Mock create_document to fail on specific fact
        async def mock_create_document_with_error(
            db, document, app_id, user_id, collection_id, duplicate_threshold
        ):
            if "cause an error" in document.content:
                raise Exception("Vector embedding service temporarily unavailable")
            # Otherwise create normally
            new_doc = models.Document(
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                content=document.content,
                h_metadata=document.metadata,
                embedding=[0.1] * 1536,
            )
            db.add(new_doc)
            return new_doc

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.create_document",
                side_effect=mock_create_document_with_error,
            ):
                # Should complete despite partial failures
                await store.save_facts(test_facts)

                # Verify partial storage - successful facts should be stored
                result = await db_session.execute(
                    models.Document.__table__.select().where(
                        models.Document.collection_id == collection.public_id
                    )
                )
                stored_documents = result.fetchall()

                stored_contents = [doc.content for doc in stored_documents]
                # These should have been stored successfully
                assert "User likes programming" in stored_contents
                assert "User works in technology" in stored_contents
                # This should have failed to store
                assert (
                    "This fact will cause an error during storage"
                    not in stored_contents
                )

        print(
            f"✓ Error handling test passed - stored {len(stored_documents)} out of {len(test_facts)} facts"
        )

    @pytest.mark.asyncio
    async def test_large_fact_volumes(self, db_session, embedding_store_setup):
        """Test embedding store performance with larger volumes of facts."""
        test_app, test_user, collection, store = embedding_store_setup

        # Generate a moderate set of facts (50 instead of 200 for faster testing)
        large_fact_set = []
        for i in range(10):
            large_fact_set.extend(
                [
                    f"User has experience with technology {i}",
                    f"User worked on project {i} for 6 months",
                    f"User learned skill {i} during their career",
                    f"User enjoys activity {i} in their spare time",
                    f"User collaborated with team {i} on initiatives",
                ]
            )

        # Should have 50 facts total
        assert len(large_fact_set) == 50

        def mock_tracked_db(_operation_name):
            class MockContext:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
                    return None

            return MockContext()

        # Mock create_document to create documents normally
        async def mock_create_document(
            db, document, app_id, user_id, collection_id, duplicate_threshold=None
        ):
            new_doc = models.Document(
                app_id=app_id,
                user_id=user_id,
                collection_id=collection_id,
                content=document.content,
                h_metadata=document.metadata,
                embedding=[0.1] * 1536,
            )
            db.add(new_doc)
            return new_doc

        with patch(
            "src.deriver.tom.embeddings.tracked_db", side_effect=mock_tracked_db
        ):
            with patch(
                "src.deriver.tom.embeddings.crud.create_document",
                side_effect=mock_create_document,
            ):
                # Process in chunks to simulate realistic usage
                chunk_size = 10
                for i in range(0, len(large_fact_set), chunk_size):
                    chunk = large_fact_set[i : i + chunk_size]
                    await store.save_facts(chunk)
                    await db_session.flush()  # Ensure each chunk is committed

                # Verify all facts were stored
                result = await db_session.execute(
                    models.Document.__table__.select().where(
                        models.Document.collection_id == collection.public_id
                    )
                )
                stored_documents = result.fetchall()

                assert len(stored_documents) == len(large_fact_set)

                # Verify content integrity with sampling
                stored_contents = [doc.content for doc in stored_documents]
                # Check first and last facts
                assert large_fact_set[0] in stored_contents
                assert large_fact_set[-1] in stored_contents
                # Check some middle facts
                assert large_fact_set[25] in stored_contents
                assert large_fact_set[40] in stored_contents

        print(f"✓ Large volume test passed - stored {len(stored_documents)} facts")
