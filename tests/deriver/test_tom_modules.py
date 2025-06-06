"""Tests for TOM (Theory of Mind) inference modules."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from src.deriver.tom import (
    get_tom_inference,
    get_user_representation
)
from src.deriver.tom.single_prompt import (
    get_tom_inference_single_prompt,
    get_user_representation_single_prompt
)
from src.deriver.tom.conversational import (
    get_tom_inference_conversational,
    get_user_representation_conversational
)
from src.deriver.tom.long_term import (
    get_user_representation_long_term,
    extract_facts_long_term
)


class TestTOMRouter:
    """Test the main TOM routing functions in __init__.py."""

    @pytest.mark.asyncio
    async def test_get_tom_inference_routes_to_conversational(self):
        """Test routing to conversational TOM inference method."""
        chat_history = "User: I'm a Python developer\nAI: How long have you been coding?"
        session_id = str(uuid4())
        user_representation = "User is technical"

        with patch("src.deriver.tom.get_tom_inference_conversational") as mock_conversational:
            mock_conversational.return_value = "Conversational TOM response"

            result = await get_tom_inference(
                chat_history, session_id, user_representation, method="conversational"
            )

            mock_conversational.assert_called_once_with(
                chat_history, session_id, user_representation
            )
            assert result == "Conversational TOM response"

    @pytest.mark.asyncio
    async def test_get_tom_inference_routes_to_single_prompt(self):
        """Test routing to single prompt TOM inference method."""
        chat_history = "User: I love machine learning\nAI: What frameworks do you use?"
        session_id = str(uuid4())

        with patch("src.deriver.tom.get_tom_inference_single_prompt") as mock_single_prompt:
            mock_single_prompt.return_value = "Single prompt TOM response"

            result = await get_tom_inference(
                chat_history, session_id, method="single_prompt"
            )

            mock_single_prompt.assert_called_once_with(
                chat_history, session_id, "None"
            )
            assert result == "Single prompt TOM response"

    @pytest.mark.asyncio
    async def test_get_tom_inference_invalid_method_raises_error(self):
        """Test that invalid TOM inference method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method: invalid_method"):
            await get_tom_inference(
                "chat history", "session_id", method="invalid_method"
            )

    @pytest.mark.asyncio
    async def test_get_user_representation_routes_to_conversational(self):
        """Test routing to conversational user representation method."""
        chat_history = "User: I work in AI research"
        session_id = str(uuid4())
        tom_inference = "User is excited about AI"

        with patch("src.deriver.tom.get_user_representation_conversational") as mock_conversational:
            mock_conversational.return_value = "Conversational representation"

            result = await get_user_representation(
                chat_history, session_id, tom_inference=tom_inference, method="conversational"
            )

            mock_conversational.assert_called_once_with(
                chat_history, session_id, "None", tom_inference
            )
            assert result == "Conversational representation"

    @pytest.mark.asyncio
    async def test_get_user_representation_routes_to_long_term(self):
        """Test routing to long term user representation method."""
        chat_history = "User: I've been programming for 5 years"
        session_id = str(uuid4())

        with patch("src.deriver.tom.get_user_representation_long_term") as mock_long_term:
            mock_long_term.return_value = "Long term representation"

            result = await get_user_representation(
                chat_history, session_id, method="long_term"
            )

            mock_long_term.assert_called_once_with(
                chat_history, session_id, "None", "None"
            )
            assert result == "Long term representation"

    @pytest.mark.asyncio
    async def test_get_user_representation_invalid_method_raises_error(self):
        """Test that invalid user representation method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method: unknown_method"):
            await get_user_representation(
                "chat history", "session_id", method="unknown_method"
            )

    @pytest.mark.asyncio
    async def test_tom_inference_with_kwargs(self):
        """Test that kwargs are properly passed through to TOM methods."""
        chat_history = "User: Test message"
        session_id = str(uuid4())
        extra_param = "test_value"

        with patch("src.deriver.tom.get_tom_inference_single_prompt") as mock_single_prompt:
            mock_single_prompt.return_value = "Response with kwargs"

            await get_tom_inference(
                chat_history, session_id, method="single_prompt", extra_param=extra_param
            )

            # Verify kwargs were passed through
            mock_single_prompt.assert_called_once_with(
                chat_history, session_id, "None", extra_param=extra_param
            )


class TestSinglePromptMethods:
    """Test the single prompt TOM inference methods."""

    @pytest.mark.asyncio
    async def test_get_tom_inference_single_prompt_basic(self, mock_model_clients):
        """Test basic single prompt TOM inference."""
        chat_history = "User: I'm feeling stressed about work\nAI: What's causing the stress?"
        session_id = str(uuid4())

        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            result = await get_tom_inference_single_prompt(chat_history, session_id)

            # Verify model client was called with correct parameters
            mock_client = mock_model_clients["single_prompt"]
            mock_client.generate.assert_called_once()
            call_kwargs = mock_client.generate.call_args[1]
            
            assert call_kwargs["max_tokens"] == 1000
            assert call_kwargs["temperature"] == 0
            assert call_kwargs["use_caching"] is True
            # The system prompt should contain key TOM instruction phrases
            assert "system" in call_kwargs
            system_prompt = call_kwargs["system"]
            assert "theory of mind" in system_prompt.lower() or "prediction" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_get_tom_inference_single_prompt_with_user_representation(self, mock_model_clients):
        """Test single prompt TOM inference with existing user representation."""
        chat_history = "User: I changed my mind about the project"
        session_id = str(uuid4())
        user_representation = "User is decisive and goal-oriented"

        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            await get_tom_inference_single_prompt(
                chat_history, session_id, user_representation
            )

            # Verify user representation was included in messages
            mock_client = mock_model_clients["single_prompt"]
            call_args = mock_client.generate.call_args[1]
            messages = call_args["messages"]
            
            # Should have two messages: main analysis + user representation context
            assert len(messages) == 2
            assert user_representation in str(messages)

    @pytest.mark.asyncio
    async def test_get_tom_inference_single_prompt_handles_error(self, mock_model_clients):
        """Test that single prompt TOM inference handles LLM errors."""
        chat_history = "User: Test message"
        session_id = str(uuid4())

        # Mock the model client to raise an exception
        mock_model_clients["single_prompt"].generate.side_effect = Exception("LLM API Error")

        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None
            
            with patch("src.deriver.tom.single_prompt.sentry_sdk.capture_exception") as mock_capture:
                with pytest.raises(Exception, match="LLM API Error"):
                    await get_tom_inference_single_prompt(chat_history, session_id)
                
                # Verify error was captured
                mock_capture.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_representation_single_prompt_basic(self, mock_model_clients):
        """Test basic single prompt user representation."""
        chat_history = "User: I'm a data scientist\nAI: What tools do you use?"
        session_id = str(uuid4())
        tom_inference = "User is passionate about data science"

        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            result = await get_user_representation_single_prompt(
                chat_history, session_id, tom_inference=tom_inference
            )

            # Verify correct system prompt was used
            mock_client = mock_model_clients["single_prompt"] 
            call_kwargs = mock_client.generate.call_args[1]
            # The system prompt should contain user representation instructions
            assert "system" in call_kwargs
            system_prompt = call_kwargs["system"]
            assert "user representation" in system_prompt.lower() or "factual" in system_prompt.lower()
            
            # Verify TOM inference was included in context
            messages = call_kwargs["messages"]
            assert tom_inference in str(messages)

    @pytest.mark.asyncio
    async def test_get_user_representation_single_prompt_all_inputs(self, mock_model_clients):
        """Test single prompt user representation with all optional inputs."""
        chat_history = "User: I've been learning React lately"
        session_id = str(uuid4())
        user_representation = "User is a full-stack developer"
        tom_inference = "User is eager to learn new technologies"

        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            await get_user_representation_single_prompt(
                chat_history, session_id, user_representation, tom_inference
            )

            # Verify all inputs were included in the context
            mock_client = mock_model_clients["single_prompt"]
            call_kwargs = mock_client.generate.call_args[1]
            messages = call_kwargs["messages"]
            
            message_content = str(messages)
            assert chat_history in message_content
            assert user_representation in message_content
            assert tom_inference in message_content


class TestConversationalMethods:
    """Test the conversational TOM inference methods."""

    @pytest.mark.asyncio
    async def test_get_tom_inference_conversational_basic(self):
        """Test basic conversational TOM inference."""
        chat_history = "User: I'm learning to cook\nAI: That's exciting! What dishes interest you?"
        session_id = str(uuid4())
        user_representation = "User enjoys trying new things"

        # Mock the Anthropic client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="<prediction>User is enthusiastic about cooking</prediction>")]

        with patch("src.deriver.tom.conversational.anthropic.messages.create") as mock_create:
            mock_create.return_value = mock_message
            
            with patch("src.deriver.tom.conversational.sentry_sdk.start_transaction") as mock_transaction:
                mock_transaction.return_value.__enter__.return_value = MagicMock()
                mock_transaction.return_value.__exit__.return_value = None

                result = await get_tom_inference_conversational(
                    chat_history, session_id, user_representation
                )

                # Verify Anthropic client was called
                mock_create.assert_called_once()
                call_kwargs = mock_create.call_args[1]
                
                assert call_kwargs["model"] == "claude-3-5-sonnet-20240620"
                assert call_kwargs["max_tokens"] == 1000
                assert call_kwargs["temperature"] == 0
                
                # Verify chat history and user representation were included  
                messages = call_kwargs["messages"]
                message_content = str(messages)
                # Check for key parts of the chat history and user representation
                assert "learning to cook" in message_content.lower()
                assert "enjoys trying new things" in message_content.lower() or user_representation in message_content

                assert result == "<prediction>User is enthusiastic about cooking</prediction>"

    @pytest.mark.asyncio
    async def test_get_tom_inference_conversational_complex_prompting(self):
        """Test that conversational method uses complex metanarrative prompting."""
        chat_history = "User: I'm having trouble with my team\nAI: What kind of challenges are you facing?"
        session_id = str(uuid4())

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="User seems frustrated with team dynamics")]

        with patch("src.deriver.tom.conversational.anthropic.messages.create") as mock_create:
            mock_create.return_value = mock_message
            
            with patch("src.deriver.tom.conversational.sentry_sdk.start_transaction") as mock_transaction:
                mock_transaction.return_value.__enter__.return_value = MagicMock()
                mock_transaction.return_value.__exit__.return_value = None

                await get_tom_inference_conversational(chat_history, session_id)

                # Verify complex prompting structure
                call_kwargs = mock_create.call_args[1]
                messages = call_kwargs["messages"]
                
                # Should have multiple role-playing messages
                assert len(messages) >= 5
                
                # Verify OOC (out of character) setup is included
                message_content = str(messages)
                assert "OOC" in message_content
                assert "experiment" in message_content.lower()

    @pytest.mark.asyncio
    async def test_get_user_representation_conversational_basic(self):
        """Test basic conversational user representation."""
        chat_history = "User: I work in finance but I'm passionate about art"
        session_id = str(uuid4())
        tom_inference = "User has diverse interests spanning analytical and creative domains"

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="<representation>User balances analytical work with creative pursuits</representation>")]

        with patch("src.deriver.tom.conversational.anthropic.messages.create") as mock_create:
            mock_create.return_value = mock_message
            
            with patch("src.deriver.tom.conversational.sentry_sdk.start_transaction") as mock_transaction:
                mock_transaction.return_value.__enter__.return_value = MagicMock()
                mock_transaction.return_value.__exit__.return_value = None

                result = await get_user_representation_conversational(
                    chat_history, session_id, tom_inference=tom_inference
                )

                # Verify TOM inference was included in the prompt
                call_kwargs = mock_create.call_args[1]
                messages = call_kwargs["messages"]
                assert tom_inference in str(messages)

                assert result == "<representation>User balances analytical work with creative pursuits</representation>"

    @pytest.mark.asyncio
    async def test_get_user_representation_conversational_with_existing_representation(self):
        """Test conversational user representation with existing representation."""
        chat_history = "User: I've started learning piano"
        session_id = str(uuid4())
        user_representation = "User enjoys creative hobbies"
        tom_inference = "User is expanding creative skills"

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Updated representation with piano learning")]

        with patch("src.deriver.tom.conversational.anthropic.messages.create") as mock_create:
            mock_create.return_value = mock_message
            
            with patch("src.deriver.tom.conversational.sentry_sdk.start_transaction") as mock_transaction:
                mock_transaction.return_value.__enter__.return_value = MagicMock()
                mock_transaction.return_value.__exit__.return_value = None

                await get_user_representation_conversational(
                    chat_history, session_id, user_representation, tom_inference
                )

                # Verify all inputs were included
                call_kwargs = mock_create.call_args[1]
                messages = call_kwargs["messages"]
                message_content = str(messages)
                
                assert chat_history in message_content
                assert user_representation in message_content
                assert tom_inference in message_content


class TestLongTermMethods:
    """Test the long term TOM methods."""

    @pytest.mark.asyncio
    async def test_extract_facts_long_term_basic(self, mock_model_clients, mock_llm_responses):
        """Test basic fact extraction from chat history."""
        chat_history = "User: I'm a software engineer at Google and I love hiking on weekends"

        # Mock the response with proper XML format
        mock_response = f'<facts>{mock_llm_responses["fact_extraction"]}</facts>'
        mock_model_clients["long_term"].generate.return_value = mock_response

        with patch("src.deriver.tom.long_term.parse_xml_content") as mock_parse_xml:
            mock_parse_xml.return_value = mock_llm_responses["fact_extraction"]

            facts = await extract_facts_long_term(chat_history)

            # Verify model client was called
            mock_client = mock_model_clients["long_term"]
            mock_client.generate.assert_called_once()
            
            call_kwargs = mock_client.generate.call_args[1]
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["use_caching"] is True
            
            # Verify the system prompt includes the chat history
            messages = call_kwargs["messages"]
            message_content = str(messages)
            # Check for key parts of the chat history
            assert "software engineer" in message_content.lower()
            assert "google" in message_content.lower() or "hiking" in message_content.lower()

            # Verify facts were extracted correctly
            expected_facts = [
                "User is a software developer",
                "User works remotely", 
                "User prefers coffee over tea",
                "User uses Python and JavaScript"
            ]
            assert facts == expected_facts

    @pytest.mark.asyncio
    async def test_extract_facts_long_term_handles_json_error(self, mock_model_clients):
        """Test that fact extraction handles JSON parsing errors gracefully."""
        chat_history = "User: I like programming"

        # Mock malformed response
        mock_model_clients["long_term"].generate.return_value = "Invalid JSON response"

        with patch("src.deriver.tom.long_term.parse_xml_content") as mock_parse_xml:
            mock_parse_xml.return_value = "Not valid JSON"

            facts = await extract_facts_long_term(chat_history)

            # Should return empty list on error
            assert facts == []

    @pytest.mark.asyncio
    async def test_extract_facts_long_term_handles_missing_facts_key(self, mock_model_clients):
        """Test that fact extraction handles missing 'facts' key in response."""
        chat_history = "User: Test message"

        # Mock response with missing facts key
        invalid_response = json.dumps({"other_key": "some_value"})
        mock_model_clients["long_term"].generate.return_value = f"<facts>{invalid_response}</facts>"

        with patch("src.deriver.tom.long_term.parse_xml_content") as mock_parse_xml:
            mock_parse_xml.return_value = invalid_response

            facts = await extract_facts_long_term(chat_history)

            # Should return empty list on KeyError
            assert facts == []

    @pytest.mark.asyncio
    async def test_get_user_representation_long_term_basic(self, mock_model_clients):
        """Test basic long term user representation."""
        chat_history = "User: I'm starting a new job next week"
        session_id = str(uuid4())
        facts = ["User is a software engineer", "User is changing jobs"]

        mock_response = "CURRENT STATE:\n- Active Context: Starting new job\n<KNOWN_FACTS>\nTENTATIVE PATTERNS:\n- High confidence: Career-focused"
        mock_model_clients["long_term"].generate.return_value = mock_response

        result = await get_user_representation_long_term(
            chat_history, session_id, facts=facts
        )

        # Verify model client was called
        mock_client = mock_model_clients["long_term"]
        mock_client.generate.assert_called_once()
        
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["use_caching"] is True
        
        # Verify chat history was included
        messages = call_kwargs["messages"]
        assert chat_history in str(messages)

        # Verify facts were injected into the response
        assert "User is a software engineer" in result
        assert "User is changing jobs" in result
        assert "<KNOWN_FACTS>" not in result  # Should be replaced

    @pytest.mark.asyncio
    async def test_get_user_representation_long_term_with_all_inputs(self, mock_model_clients):
        """Test long term user representation with all optional inputs."""
        chat_history = "User: I'm excited about the new project"
        session_id = str(uuid4())
        user_representation = "User is enthusiastic about work"
        tom_inference = "User is feeling motivated"
        facts = ["User works in tech", "User enjoys new challenges"]

        mock_response = "Updated representation with <KNOWN_FACTS> placeholder"
        mock_model_clients["long_term"].generate.return_value = mock_response

        result = await get_user_representation_long_term(
            chat_history, session_id, user_representation, tom_inference, facts
        )

        # Verify all inputs were included in the context
        mock_client = mock_model_clients["long_term"]
        call_kwargs = mock_client.generate.call_args[1]
        messages = call_kwargs["messages"]
        message_content = str(messages)
        
        assert chat_history in message_content
        assert user_representation in message_content
        assert tom_inference in message_content

        # Verify facts injection worked
        assert "User works in tech" in result
        assert "User enjoys new challenges" in result

    @pytest.mark.asyncio
    async def test_get_user_representation_long_term_empty_facts(self, mock_model_clients):
        """Test long term user representation with empty facts list."""
        chat_history = "User: Hello there"
        session_id = str(uuid4())

        mock_response = "Basic representation with <KNOWN_FACTS> placeholder"
        mock_model_clients["long_term"].generate.return_value = mock_response

        result = await get_user_representation_long_term(chat_history, session_id)

        # Verify empty facts are handled gracefully
        assert "PERSISTENT INFORMATION:\n" in result
        assert "<KNOWN_FACTS>" not in result

    @pytest.mark.asyncio
    async def test_get_user_representation_long_term_none_inputs(self, mock_model_clients):
        """Test long term user representation with None inputs."""
        chat_history = "User: Test message"
        session_id = str(uuid4())

        mock_response = "Representation with <KNOWN_FACTS>"
        mock_model_clients["long_term"].generate.return_value = mock_response

        result = await get_user_representation_long_term(
            chat_history, session_id, user_representation="None", tom_inference="None"
        )

        # Verify None inputs are handled (not included in context)
        mock_client = mock_model_clients["long_term"]
        call_kwargs = mock_client.generate.call_args[1]
        messages = call_kwargs["messages"]
        message_content = str(messages)
        
        # "None" values should not be included in context strings
        assert "EXISTING USER REPRESENTATION - INCOMPLETE" not in message_content
        assert "PREDICTION OF USER MENTAL STATE" not in message_content


class TestTOMIntegration:
    """Test integration scenarios across TOM methods."""

    @pytest.mark.asyncio
    async def test_method_configuration_via_environment(self, mock_model_clients):
        """Test that TOM methods can be configured via environment variables."""
        chat_history = "User: I'm learning data science"
        session_id = str(uuid4())

        # Test single_prompt method
        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            await get_tom_inference(chat_history, session_id, method="single_prompt")
            
            # Verify single prompt was called
            mock_model_clients["single_prompt"].generate.assert_called()

    @pytest.mark.asyncio
    async def test_error_handling_across_methods(self, mock_model_clients):
        """Test error handling consistency across different TOM methods."""
        chat_history = "User: Error test"
        session_id = str(uuid4())

        # Test single_prompt error handling
        mock_model_clients["single_prompt"].generate.side_effect = Exception("API Error")
        
        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None
            
            with patch("src.deriver.tom.single_prompt.sentry_sdk.capture_exception"):
                with pytest.raises(Exception):
                    await get_tom_inference(chat_history, session_id, method="single_prompt")

    @pytest.mark.asyncio
    async def test_response_format_consistency(self, mock_model_clients, mock_llm_responses):
        """Test that different methods return appropriately formatted responses."""
        chat_history = "User: I'm a product manager"
        session_id = str(uuid4())

        # Test single prompt response format
        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            single_prompt_result = await get_tom_inference(
                chat_history, session_id, method="single_prompt"
            )
            
            # Verify response is returned correctly
            assert single_prompt_result == mock_llm_responses["tom_single_prompt"]

        # Test long term fact extraction format
        with patch("src.deriver.tom.long_term.parse_xml_content") as mock_parse_xml:
            mock_parse_xml.return_value = mock_llm_responses["fact_extraction"]
            
            facts = await extract_facts_long_term(chat_history)
            
            # Verify facts are returned as list of strings
            assert isinstance(facts, list)
            assert all(isinstance(fact, str) for fact in facts)

    @pytest.mark.asyncio
    async def test_caching_behavior_across_methods(self, mock_model_clients):
        """Test that caching is properly enabled across different methods."""
        chat_history = "User: Testing caching"
        session_id = str(uuid4())

        # Test single prompt caching
        with patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_transaction:
            mock_transaction.return_value.__enter__.return_value = MagicMock()
            mock_transaction.return_value.__exit__.return_value = None

            await get_tom_inference(chat_history, session_id, method="single_prompt")
            
            call_kwargs = mock_model_clients["single_prompt"].generate.call_args[1]
            assert call_kwargs["use_caching"] is True

        # Test long term fact extraction caching
        with patch("src.deriver.tom.long_term.parse_xml_content") as mock_parse_xml:
            mock_parse_xml.return_value = '{"facts": []}'
            
            await extract_facts_long_term(chat_history)
            
            call_kwargs = mock_model_clients["long_term"].generate.call_args[1]
            assert call_kwargs["use_caching"] is True

    @pytest.mark.asyncio
    async def test_observability_integration(self, mock_model_clients):
        """Test that observability tools (Sentry, Langfuse) are properly integrated."""
        chat_history = "User: Testing observability"
        session_id = str(uuid4())

        # Test single prompt observability
        with (
            patch("src.deriver.tom.single_prompt.sentry_sdk.start_transaction") as mock_sentry,
            patch("src.deriver.tom.single_prompt.langfuse_context.update_current_observation") as mock_langfuse
        ):
            mock_sentry.return_value.__enter__.return_value = MagicMock()
            mock_sentry.return_value.__exit__.return_value = None

            await get_tom_inference(chat_history, session_id, method="single_prompt")

            # Verify Sentry transaction was started
            mock_sentry.assert_called_once_with(op="tom-inference", name="ToM Inference")
            
            # Verify Langfuse observation was updated
            mock_langfuse.assert_called_once()

        # Test conversational method observability
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Test response")]

        with (
            patch("src.deriver.tom.conversational.anthropic.messages.create") as mock_create,
            patch("src.deriver.tom.conversational.sentry_sdk.start_transaction") as mock_sentry,
            patch("src.deriver.tom.conversational.langfuse_context.update_current_observation") as mock_langfuse
        ):
            mock_create.return_value = mock_message
            mock_sentry.return_value.__enter__.return_value = MagicMock()
            mock_sentry.return_value.__exit__.return_value = None

            await get_tom_inference(chat_history, session_id, method="conversational")

            # Verify observability integration
            mock_sentry.assert_called_once()
            mock_langfuse.assert_called_once()