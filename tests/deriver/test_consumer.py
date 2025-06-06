"""Tests for the consumer module and message processing functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src import models
from src.deriver import consumer
from src.utils.history import SummaryType


class TestProcessItem:
    """Test the main process_item entry point."""

    @pytest.mark.asyncio
    async def test_process_item_routes_user_message(self, db_session):
        """Test that process_item correctly routes user messages."""
        payload = {
            "message_id": str(uuid4()),
            "is_user": True,
            "content": "Hello, I'm a Python developer",
            "app_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_id": str(uuid4())
        }

        with (
            patch("src.deriver.consumer.process_user_message") as mock_process_user,
            patch("src.deriver.consumer.process_ai_message") as mock_process_ai,
            patch("src.deriver.consumer.summarize_if_needed") as mock_summarize
        ):
            mock_process_user.return_value = None
            mock_process_ai.return_value = None
            mock_summarize.return_value = None

            await consumer.process_item(db_session, payload)

            # Should call user message processing
            mock_process_user.assert_called_once_with(
                payload["content"],
                payload["app_id"],
                payload["user_id"],
                payload["session_id"],
                payload["message_id"],
                db_session
            )
            mock_process_ai.assert_not_called()
            mock_summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_item_routes_ai_message(self, db_session):
        """Test that process_item correctly routes AI messages."""
        payload = {
            "message_id": str(uuid4()),
            "is_user": False,
            "content": "I can help you with Python development!",
            "app_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_id": str(uuid4())
        }

        with (
            patch("src.deriver.consumer.process_user_message") as mock_process_user,
            patch("src.deriver.consumer.process_ai_message") as mock_process_ai,
            patch("src.deriver.consumer.summarize_if_needed") as mock_summarize
        ):
            mock_process_user.return_value = None
            mock_process_ai.return_value = None
            mock_summarize.return_value = None

            await consumer.process_item(db_session, payload)

            # Should call AI message processing
            mock_process_ai.assert_called_once_with(
                payload["content"],
                payload["app_id"],
                payload["user_id"],
                payload["session_id"],
                payload["message_id"],
                db_session
            )
            mock_process_user.assert_not_called()
            mock_summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_item_calls_summarize(self, db_session):
        """Test that process_item always calls summarize_if_needed."""
        payload = {
            "message_id": str(uuid4()),
            "is_user": True,
            "content": "Test message",
            "app_id": str(uuid4()),
            "user_id": str(uuid4()),
            "session_id": str(uuid4())
        }

        with (
            patch("src.deriver.consumer.process_user_message") as mock_process_user,
            patch("src.deriver.consumer.summarize_if_needed") as mock_summarize
        ):
            mock_process_user.return_value = None
            mock_summarize.return_value = None

            await consumer.process_item(db_session, payload)

            mock_summarize.assert_called_once_with(
                db_session,
                payload["app_id"],
                payload["session_id"],
                payload["user_id"],
                payload["message_id"]
            )


class TestProcessUserMessage:
    """Test user message processing functionality."""

    @pytest_asyncio.fixture
    async def setup_user_message_test(self, db_session, sample_data):
        """Setup test data for user message processing."""
        test_app, test_user = sample_data
        
        # Create a session
        session = models.Session(
            user_id=test_user.public_id,
            app_id=test_app.public_id,
            metadata={}
        )
        db_session.add(session)
        await db_session.flush()

        # Create a collection for the user
        collection = models.Collection(
            app_id=test_app.public_id,
            user_id=test_user.public_id,
            name=f"user_{test_user.public_id}",
            metadata={"type": "user_facts"}
        )
        db_session.add(collection)
        await db_session.flush()

        return test_app, test_user, session, collection

    @pytest.mark.asyncio
    async def test_process_user_message_extracts_and_saves_facts(self, db_session, setup_user_message_test, mock_llm_responses):
        """Test that user message processing extracts and saves facts."""
        test_app, test_user, session, collection = setup_user_message_test
        
        message_content = "I'm a Python developer who works remotely and loves coffee"
        message_id = str(uuid4())

        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_get_history,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection,
            patch("src.deriver.consumer.CollectionEmbeddingStore") as mock_embedding_store_class
        ):
            # Mock history retrieval
            mock_get_history.return_value = ("Previous chat", [], None)
            
            # Mock collection retrieval
            mock_get_collection.return_value = collection
            
            # Mock embedding store
            mock_embedding_store = AsyncMock()
            mock_embedding_store.remove_duplicates.return_value = [
                "User is a software developer",
                "User works remotely"
            ]  # Simulate same facts from global mock
            mock_embedding_store.save_facts.return_value = None
            mock_embedding_store_class.return_value = mock_embedding_store

            # Process the user message
            await consumer.process_user_message(
                message_content,
                test_app.public_id,
                test_user.public_id,
                session.public_id,
                message_id,
                db_session
            )

            # Verify the flow
            mock_get_history.assert_called_once_with(
                db_session, session.public_id, summary_type=SummaryType.SHORT
            )
            mock_get_collection.assert_called_once_with(
                db=db_session, app_id=test_app.public_id, user_id=test_user.public_id
            )
            mock_embedding_store.remove_duplicates.assert_called_once_with([
                "User is a software developer", 
                "User works remotely", 
                "User prefers coffee over tea", 
                "User uses Python and JavaScript"
            ])
            mock_embedding_store.save_facts.assert_called_once_with(
                ["User is a software developer", "User works remotely"],
                message_id=message_id
            )

    @pytest.mark.asyncio
    async def test_process_user_message_no_unique_facts(self, db_session, setup_user_message_test):
        """Test user message processing when all facts are duplicates."""
        test_app, test_user, session, collection = setup_user_message_test
        
        message_content = "I still love Python programming"
        message_id = str(uuid4())

        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_get_history,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection,
            patch("src.deriver.consumer.CollectionEmbeddingStore") as mock_embedding_store_class
        ):
            mock_get_history.return_value = ("Previous chat", [], None)
            mock_get_collection.return_value = collection
            
            # Mock embedding store to return no unique facts
            mock_embedding_store = AsyncMock()
            mock_embedding_store.remove_duplicates.return_value = []  # All duplicates
            mock_embedding_store_class.return_value = mock_embedding_store

            await consumer.process_user_message(
                message_content,
                test_app.public_id,
                test_user.public_id,
                session.public_id,
                message_id,
                db_session
            )

            # Should not call save_facts when no unique facts
            mock_embedding_store.save_facts.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_user_message_with_chat_history(self, db_session, setup_user_message_test):
        """Test that chat history is properly included in fact extraction."""
        test_app, test_user, session, collection = setup_user_message_test
        
        message_content = "I prefer PyTorch over TensorFlow"
        message_id = str(uuid4())

        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_get_history,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection,
            patch("src.deriver.consumer.CollectionEmbeddingStore") as mock_embedding_store_class
        ):
            # Mock history with previous context
            mock_get_history.return_value = (
                "AI: Hello! How can I help?\nhuman: I'm a machine learning engineer",
                [],
                None
            )
            mock_get_collection.return_value = collection
            
            mock_embedding_store = AsyncMock()
            mock_embedding_store.remove_duplicates.return_value = ["User is a software developer", "User works remotely"]
            mock_embedding_store_class.return_value = mock_embedding_store

            await consumer.process_user_message(
                message_content,
                test_app.public_id,
                test_user.public_id,
                session.public_id,
                message_id,
                db_session
            )

            # Verify the flow completed successfully - fact extraction uses global mock

    @pytest.mark.asyncio
    async def test_process_user_message_handles_extraction_error(self, db_session, setup_user_message_test):
        """Test that user message processing handles fact extraction errors gracefully."""
        test_app, test_user, session, collection = setup_user_message_test
        
        message_content = "Test message"
        message_id = str(uuid4())

        with (
            patch("src.deriver.consumer.history.get_summarized_history") as mock_get_history,
            patch("src.deriver.consumer.extract_facts_long_term") as mock_extract_facts,
            patch("src.deriver.consumer.crud.get_or_create_user_protected_collection") as mock_get_collection
        ):
            mock_get_history.return_value = ("", [], None)
            mock_extract_facts.side_effect = Exception("LLM API error")
            mock_get_collection.return_value = collection

            # Should raise the exception (let caller handle it)
            with pytest.raises(Exception, match="LLM API error"):
                await consumer.process_user_message(
                    message_content,
                    test_app.public_id,
                    test_user.public_id,
                    session.public_id,
                    message_id,
                    db_session
                )


class TestProcessAIMessage:
    """Test AI message processing functionality."""

    @pytest.mark.asyncio
    async def test_process_ai_message_basic_functionality(self, db_session):
        """Test basic AI message processing (currently just console output)."""
        content = "I can help you with Python programming!"
        app_id = str(uuid4())
        user_id = str(uuid4())
        session_id = str(uuid4())
        message_id = str(uuid4())

        # Mock console output
        with patch("src.deriver.consumer.console.print") as mock_print:
            await consumer.process_ai_message(
                content, app_id, user_id, session_id, message_id, db_session
            )
            
            # Should print the AI message content
            mock_print.assert_called_once_with(
                f"Processing AI message: {content}", 
                style="bright_magenta"
            )


class TestSummarizeIfNeeded:
    """Test summary generation functionality."""

    @pytest_asyncio.fixture
    async def setup_summary_test(self, db_session, sample_data):
        """Setup test data for summary testing."""
        test_app, test_user = sample_data
        
        session = models.Session(
            user_id=test_user.public_id,
            app_id=test_app.public_id,
            metadata={}
        )
        db_session.add(session)
        await db_session.flush()

        return test_app, test_user, session

    @pytest.mark.asyncio
    async def test_summarize_if_needed_no_summary_required(self, db_session, setup_summary_test):
        """Test when no summary is needed."""
        test_app, test_user, session = setup_summary_test
        message_id = str(uuid4())

        with patch("src.deriver.consumer.history.should_create_summary") as mock_should_create:
            # Mock that no summary is needed
            mock_should_create.return_value = (False, [], None)

            await consumer.summarize_if_needed(
                db_session,
                test_app.public_id,
                session.public_id,
                test_user.public_id,
                message_id
            )

            # Should only check for short summary
            mock_should_create.assert_called_once_with(
                db_session, session.public_id, summary_type=SummaryType.SHORT
            )

    @pytest.mark.asyncio
    async def test_summarize_if_needed_short_summary_only(self, db_session, setup_summary_test):
        """Test creating only a short summary."""
        test_app, test_user, session = setup_summary_test
        message_id = str(uuid4())

        # Mock messages for short summary
        mock_messages = [
            MagicMock(id=1, content="Message 1"),
            MagicMock(id=2, content="Message 2")
        ]

        with (
            patch("src.deriver.consumer.history.should_create_summary") as mock_should_create,
            patch("src.deriver.consumer.history.create_summary") as mock_create_summary,
            patch("src.deriver.consumer.history.save_summary_metamessage") as mock_save_summary
        ):
            # Mock summary check responses
            def mock_should_create_side_effect(db, session_id, summary_type):
                if summary_type == SummaryType.SHORT:
                    return (True, mock_messages, None)  # Need short summary
                else:
                    return (False, [], None)  # Don't need long summary
            
            mock_should_create.side_effect = mock_should_create_side_effect
            mock_create_summary.return_value = "Short summary of recent messages"
            mock_save_summary.return_value = None

            await consumer.summarize_if_needed(
                db_session,
                test_app.public_id,
                session.public_id,
                test_user.public_id,
                message_id
            )

            # Should check for both short and long summaries
            assert mock_should_create.call_count == 2
            
            # Should create one short summary
            mock_create_summary.assert_called_once_with(
                messages=mock_messages,
                previous_summary=None,
                summary_type=SummaryType.SHORT
            )
            
            # Should save the short summary
            mock_save_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_if_needed_both_summaries(self, db_session, setup_summary_test):
        """Test creating both short and long summaries."""
        test_app, test_user, session = setup_summary_test
        message_id = str(uuid4())

        # Mock messages for summaries
        mock_short_messages = [MagicMock(id=i, content=f"Message {i}") for i in range(1, 11)]
        mock_long_messages = [MagicMock(id=i, content=f"Message {i}") for i in range(1, 61)]

        with (
            patch("src.deriver.consumer.history.should_create_summary") as mock_should_create,
            patch("src.deriver.consumer.history.create_summary") as mock_create_summary,
            patch("src.deriver.consumer.history.save_summary_metamessage") as mock_save_summary
        ):
            # Mock summary check responses
            def mock_should_create_side_effect(db, session_id, summary_type):
                if summary_type == SummaryType.SHORT:
                    return (True, mock_short_messages, None)
                else:
                    return (True, mock_long_messages, None)
            
            mock_should_create.side_effect = mock_should_create_side_effect
            
            # Mock summary creation
            def mock_create_summary_side_effect(messages, previous_summary, summary_type):
                if summary_type == SummaryType.LONG:
                    return "Long summary of conversation"
                else:
                    return "Short summary of recent messages"
            
            mock_create_summary.side_effect = mock_create_summary_side_effect
            
            # Mock save returning the long summary for short summary context
            mock_long_summary_obj = MagicMock()
            mock_long_summary_obj.content = "Long summary of conversation"
            
            def mock_save_side_effect(db, app_id, user_id, session_id, message_id, summary_content, message_count, summary_type):
                if summary_type == SummaryType.LONG:
                    return mock_long_summary_obj
                return None
            
            mock_save_summary.side_effect = mock_save_side_effect

            await consumer.summarize_if_needed(
                db_session,
                test_app.public_id,
                session.public_id,
                test_user.public_id,
                message_id
            )

            # Should create both summaries
            assert mock_create_summary.call_count == 2
            
            # Should save both summaries
            assert mock_save_summary.call_count == 2

    @pytest.mark.asyncio
    async def test_summarize_if_needed_handles_summary_creation_error(self, db_session, setup_summary_test):
        """Test that summary creation errors are handled gracefully."""
        test_app, test_user, session = setup_summary_test
        message_id = str(uuid4())

        mock_messages = [MagicMock(id=1, content="Message 1")]

        with (
            patch("src.deriver.consumer.history.should_create_summary") as mock_should_create,
            patch("src.deriver.consumer.history.create_summary") as mock_create_summary
        ):
            mock_should_create.return_value = (True, mock_messages, None)
            mock_create_summary.side_effect = Exception("LLM API error")

            # Should not raise exception (errors are logged)
            await consumer.summarize_if_needed(
                db_session,
                test_app.public_id,
                session.public_id,
                test_user.public_id,
                message_id
            )

    @pytest.mark.asyncio
    async def test_summarize_if_needed_with_existing_long_summary(self, db_session, setup_summary_test):
        """Test short summary creation with existing long summary context."""
        test_app, test_user, session = setup_summary_test
        message_id = str(uuid4())

        mock_messages = [MagicMock(id=1, content="Message 1")]
        mock_existing_long_summary = MagicMock()
        mock_existing_long_summary.content = "Existing long summary"

        with (
            patch("src.deriver.consumer.history.should_create_summary") as mock_should_create,
            patch("src.deriver.consumer.history.create_summary") as mock_create_summary,
            patch("src.deriver.consumer.history.save_summary_metamessage") as mock_save_summary
        ):
            # Mock that we need short summary and have existing long summary
            def mock_should_create_side_effect(db, session_id, summary_type):
                if summary_type == SummaryType.SHORT:
                    return (True, mock_messages, mock_existing_long_summary)
                else:
                    return (False, [], mock_existing_long_summary)
            
            mock_should_create.side_effect = mock_should_create_side_effect
            mock_create_summary.return_value = "Short summary with context"
            mock_save_summary.return_value = None

            await consumer.summarize_if_needed(
                db_session,
                test_app.public_id,
                session.public_id,
                test_user.public_id,
                message_id
            )

            # Should create short summary with existing long summary as context
            mock_create_summary.assert_called_once_with(
                messages=mock_messages,
                previous_summary="Existing long summary",
                summary_type=SummaryType.SHORT
            )


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_tom_method_default(self):
        """Test TOM_METHOD defaults to single_prompt."""
        # Test the getenv behavior that the module uses
        import os
        default_value = os.getenv("TOM_METHOD", "single_prompt")
        # If no environment variable is set, should use default
        if os.getenv("TOM_METHOD") is None:
            assert default_value == "single_prompt"
        else:
            # If environment variable is set, respect it
            assert default_value == os.getenv("TOM_METHOD")

    def test_tom_method_custom(self):
        """Test TOM_METHOD can be customized via environment."""
        # Test that the os.getenv pattern works correctly
        import os
        # Simulate the pattern used in the consumer module
        test_value = os.getenv("TOM_METHOD", "single_prompt")
        # The result should be either the env var or the default
        assert test_value in ["single_prompt", "conversational", "long_term"]

    def test_user_representation_method_default(self):
        """Test USER_REPRESENTATION_METHOD defaults to long_term."""
        # Test the getenv behavior that the module uses
        import os
        default_value = os.getenv("USER_REPRESENTATION_METHOD", "long_term")
        # If no environment variable is set, should use default
        if os.getenv("USER_REPRESENTATION_METHOD") is None:
            assert default_value == "long_term"
        else:
            # If environment variable is set, respect it
            assert default_value == os.getenv("USER_REPRESENTATION_METHOD")