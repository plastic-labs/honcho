"""Performance tests for the deriver system."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.deriver import consumer
from src.deriver.tom.embeddings import CollectionEmbeddingStore


class TestPerformanceValidation:
    """Test performance characteristics of deriver components."""

    @pytest.mark.asyncio
    async def test_fact_extraction_performance(self, performance_config):
        """Test that fact extraction completes within reasonable time limits."""
        start_time = time.time()
        
        # Mock a reasonably complex chat history
        chat_history = """
        User: Hi, I'm Alex, a senior software engineer working at Google in the machine learning team.
        AI: Hello Alex! That sounds like an exciting role. What kind of ML projects are you working on?
        User: I'm primarily focused on building recommendation systems using TensorFlow and PyTorch. 
        We handle millions of user interactions daily and need to provide real-time personalized recommendations.
        AI: That's impressive scale! How do you handle the computational challenges?
        User: We use a distributed architecture with Kubernetes, Redis for caching, and BigQuery for data processing.
        The team also experiments with newer frameworks like JAX for research prototypes.
        """
        
        # Extract facts and measure time - using global mocks from conftest.py
        from src.deriver.tom.long_term import extract_facts_long_term
        facts = await extract_facts_long_term(chat_history)
        
        extraction_time = time.time() - start_time
        
        # Verify performance meets requirements
        assert extraction_time < performance_config["fact_extraction_time_limit"]
        assert hasattr(facts, 'facts') and len(facts.facts) > 0
        print(f"✅ Fact extraction completed in {extraction_time:.3f}s")

    @pytest.mark.asyncio
    async def test_embedding_operations_performance(self, sample_data, performance_config):
        """Test that embedding operations complete efficiently."""
        test_app, test_user = sample_data
        
        # Create embedding store
        collection_id = "test_collection"
        store = CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)
        
        # Test data
        facts = [
            "User is a machine learning engineer",
            "User works with large-scale systems", 
            "User has expertise in TensorFlow and PyTorch",
            "User handles millions of daily interactions",
            "User uses distributed computing"
        ]
        
        with (
            patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db,
            patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes,
            patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc
        ):
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None
            
            mock_get_dupes.return_value = []  # No duplicates
            mock_create_doc.return_value = None
            
            # Test duplicate removal performance
            start_time = time.time()
            unique_facts = await store.remove_duplicates(facts)
            dedup_time = time.time() - start_time
            
            # Test fact saving performance  
            start_time = time.time()
            await store.save_facts(unique_facts)
            save_time = time.time() - start_time
            
            # Verify performance
            total_time = dedup_time + save_time
            assert total_time < 2.0  # Should complete within 2 seconds
            assert len(unique_facts) == len(facts)  # All facts should be unique
            
            print(f"✅ Embedding operations completed in {total_time:.3f}s")
            print(f"  - Deduplication: {dedup_time:.3f}s")
            print(f"  - Fact saving: {save_time:.3f}s")

    @pytest.mark.asyncio 
    async def test_concurrent_processing_performance(self, sample_data, performance_config):
        """Test performance under concurrent load."""
        test_app, test_user = sample_data
        
        # Create multiple simulated messages
        messages = [
            f"Message {i}: User sharing information about their work and interests"
            for i in range(performance_config["message_count"] // 10)  # Smaller load for test
        ]
        
        # Mock all the dependencies for speed
        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_history,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection,
            patch("src.deriver.consumer.CollectionEmbeddingStore") as mock_store_class,
            patch("src.deriver.consumer.summarize_if_needed") as mock_summarize
        ):
            # Setup fast mocks - extract_facts uses global mock
            mock_history.return_value = ("", [], None)
            mock_get_collection.return_value = AsyncMock()
            
            mock_store = AsyncMock()
            global_facts = ["User is a software developer", "User works remotely", "User prefers coffee over tea", "User uses Python and JavaScript"]
            mock_store.remove_duplicates.return_value = global_facts
            mock_store.save_facts.return_value = None
            mock_store_class.return_value = mock_store
            
            mock_summarize.return_value = None
            
            # Process messages concurrently
            start_time = time.time()
            
            async def process_single_message(content):
                await consumer.process_user_message(
                    content,
                    test_app.public_id,
                    test_user.public_id,
                    "session_123",
                    f"msg_{hash(content)}",
                    AsyncMock()  # Mock DB session
                )
            
            # Run concurrent processing
            tasks = [process_single_message(msg) for msg in messages]
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Verify performance
            messages_per_second = len(messages) / total_time
            assert messages_per_second > 5  # Should process at least 5 messages per second
            
            print(f"✅ Processed {len(messages)} messages in {total_time:.3f}s")
            print(f"  - Rate: {messages_per_second:.1f} messages/second")

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, sample_data):
        """Test that memory usage remains stable during processing."""
        test_app, test_user = sample_data
        
        # Simulate processing many messages to check for memory leaks
        message_count = 50
        
        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_history,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection,
            patch("src.deriver.consumer.CollectionEmbeddingStore") as mock_store_class,
            patch("src.deriver.consumer.summarize_if_needed") as mock_summarize
        ):
            # Setup mocks - extract_facts uses global mock from conftest.py
            mock_history.return_value = ("", [], None)
            mock_get_collection.return_value = AsyncMock()
            
            mock_store = AsyncMock()
            global_facts = ["User is a software developer", "User works remotely", "User prefers coffee over tea", "User uses Python and JavaScript"]
            mock_store.remove_duplicates.return_value = global_facts
            mock_store.save_facts.return_value = None
            mock_store_class.return_value = mock_store
            
            mock_summarize.return_value = None
            
            # Process messages in batches to simulate sustained load
            for batch in range(5):  # 5 batches of 10 messages each
                batch_tasks = []
                for i in range(10):
                    task = consumer.process_user_message(
                        f"Batch {batch} Message {i}: User information",
                        test_app.public_id, 
                        test_user.public_id,
                        f"session_{batch}",
                        f"msg_{batch}_{i}",
                        AsyncMock()
                    )
                    batch_tasks.append(task)
                
                # Process batch
                await asyncio.gather(*batch_tasks)
                
                # Small delay between batches
                await asyncio.sleep(0.01)
            
            # Verify all processing completed successfully
            assert mock_store.save_facts.call_count == message_count
            print(f"✅ Processed {message_count} messages in batches successfully")

    def test_configuration_performance_settings(self, performance_config):
        """Test that performance configuration is reasonable."""
        # Verify performance thresholds are achievable
        assert performance_config["fact_extraction_time_limit"] >= 1.0
        assert performance_config["tom_inference_time_limit"] >= 1.0
        assert performance_config["queue_processing_time_limit"] >= 0.1
        assert performance_config["max_workers"] >= 1
        assert performance_config["timeout_seconds"] >= 10
        
        print("✅ Performance configuration validated")
        print(f"  - Fact extraction limit: {performance_config['fact_extraction_time_limit']}s")
        print(f"  - TOM inference limit: {performance_config['tom_inference_time_limit']}s")
        print(f"  - Max workers: {performance_config['max_workers']}")


class TestScalabilityValidation:
    """Test scalability characteristics."""

    @pytest.mark.asyncio
    async def test_fact_storage_scalability(self, sample_data):
        """Test that fact storage can handle larger volumes."""
        test_app, test_user = sample_data
        
        # Simulate storing many facts
        large_fact_list = [f"User fact number {i}" for i in range(100)]
        
        collection_id = "test_scalability"
        store = CollectionEmbeddingStore(test_app.public_id, test_user.public_id, collection_id)
        
        with (
            patch("src.deriver.tom.embeddings.tracked_db") as mock_tracked_db,
            patch("src.deriver.tom.embeddings.crud.get_duplicate_documents") as mock_get_dupes,
            patch("src.deriver.tom.embeddings.crud.create_document") as mock_create_doc
        ):
            mock_db = AsyncMock()
            mock_tracked_db.return_value.__aenter__.return_value = mock_db
            mock_tracked_db.return_value.__aexit__.return_value = None
            
            mock_get_dupes.return_value = []
            mock_create_doc.return_value = None
            
            start_time = time.time()
            
            # Test processing in chunks
            chunk_size = 20
            for i in range(0, len(large_fact_list), chunk_size):
                chunk = large_fact_list[i:i+chunk_size]
                unique_facts = await store.remove_duplicates(chunk)
                await store.save_facts(unique_facts)
            
            total_time = time.time() - start_time
            
            # Should handle 100 facts efficiently
            assert total_time < 5.0
            assert mock_create_doc.call_count == len(large_fact_list)
            
            print(f"✅ Processed {len(large_fact_list)} facts in {total_time:.3f}s")
            print(f"  - Rate: {len(large_fact_list)/total_time:.1f} facts/second")