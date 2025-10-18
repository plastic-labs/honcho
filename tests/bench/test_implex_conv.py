"""
Unit tests for ImplexConv test runner.

Tests the conversation parsing, data loading, and core functionality
of the ImplexConv benchmark script without requiring a running Honcho instance.
"""

# pyright: reportPrivateUsage=false

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.bench.implex_conv import ImplexConvRunner


class TestConversationParsing:
    """Test conversation text parsing into message format."""

    def test_parse_simple_conversation(self):
        """Test parsing a basic two-turn conversation."""
        runner = ImplexConvRunner()

        conv_text = """Speaker1: Hello there!
Assistant: Hi, how can I help you today?

Speaker1: I need some advice.
Assistant: Of course, I'm here to help."""

        messages = runner._parse_conversation(conv_text)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello there!"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi, how can I help you today?"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "I need some advice."
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "Of course, I'm here to help."

    def test_parse_conversation_with_multiline_messages(self):
        """Test parsing messages that span multiple lines within a turn."""
        runner = ImplexConvRunner()

        conv_text = """Speaker1: I have a question about sports.
Do you have any recommendations?
Assistant: Sure! I'd be happy to help you with that.
Let me think about some good options.

Speaker1: Thanks!
Assistant: You're welcome!"""

        messages = runner._parse_conversation(conv_text)

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert "question about sports" in messages[0]["content"]
        assert "recommendations" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert "happy to help" in messages[1]["content"]
        assert "good options" in messages[1]["content"]

    def test_parse_conversation_empty_lines(self):
        """Test parsing handles empty lines correctly."""
        runner = ImplexConvRunner()

        conv_text = """

Speaker1: Hello
Assistant: Hi there

Speaker1: Goodbye
Assistant: See you later"""

        messages = runner._parse_conversation(conv_text)

        assert len(messages) == 4
        assert all(msg["role"] in ["user", "assistant"] for msg in messages)
        assert all(msg["content"].strip() for msg in messages)

    def test_parse_conversation_no_double_newline(self):
        """Test parsing when turns are not separated by double newlines."""
        runner = ImplexConvRunner()

        # Real-world data sometimes has inconsistent formatting
        conv_text = """Speaker1: First message
Assistant: First response
Speaker1: Second message
Assistant: Second response"""

        messages = runner._parse_conversation(conv_text)

        # Should still parse correctly based on role markers
        assert len(messages) >= 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_parse_conversation_empty_content_after_marker(self):
        """Test that messages with empty content after role markers are skipped."""
        runner = ImplexConvRunner()

        conv_text = """Speaker1:
Assistant: Valid response

Speaker1: Valid message
Assistant: """

        messages = runner._parse_conversation(conv_text)

        # Should only have 2 messages (the ones with actual content)
        assert len(messages) == 2
        assert messages[0]["content"] == "Valid response"
        assert messages[1]["content"] == "Valid message"

    def test_parse_conversation_no_empty_strings(self):
        """Test that all parsed messages have non-empty content."""
        runner = ImplexConvRunner()

        conv_text = """Speaker1: First message
Assistant: Second message

Speaker1:
Assistant: Third message"""

        messages = runner._parse_conversation(conv_text)

        # All messages should have non-empty content
        assert len(messages) == 3
        assert all(len(msg["content"]) > 0 for msg in messages)
        assert all(msg["content"].strip() for msg in messages)


class TestDataLoading:
    """Test loading and validating ImplexConv data files."""

    def test_load_opposed_data(self):
        """Test loading the opposed reasoning dataset."""
        runner = ImplexConvRunner(reasoning_type="opposed")
        test_file = Path("tests/bench/implex_conv_data/ImplexConv_opposed.json")

        if not test_file.exists():
            pytest.skip("ImplexConv_opposed.json not found")

        examples = runner.load_test_file(test_file)

        assert len(examples) > 0
        assert isinstance(examples[0], dict)
        assert "conversation" in examples[0]
        assert "qa" in examples[0]

    def test_load_supportive_data(self):
        """Test loading the supportive reasoning dataset."""
        runner = ImplexConvRunner(reasoning_type="supportive")
        test_file = Path("tests/bench/implex_conv_data/ImplexConv_supportive.json")

        if not test_file.exists():
            pytest.skip("ImplexConv_supportive.json not found")

        examples = runner.load_test_file(test_file)

        assert len(examples) > 0
        assert isinstance(examples[0], dict)
        assert "conversation" in examples[0]
        assert "qa" in examples[0]

    def test_validate_question_structure(self):
        """Test that questions have the required fields."""
        runner = ImplexConvRunner(reasoning_type="opposed")
        test_file = Path("tests/bench/implex_conv_data/ImplexConv_opposed.json")

        if not test_file.exists():
            pytest.skip("ImplexConv_opposed.json not found")

        examples = runner.load_test_file(test_file)
        first_example = examples[0]

        # Check QA structure
        assert len(first_example["qa"]) > 0
        question_data = first_example["qa"][0]

        assert "question" in question_data
        assert "answer" in question_data
        assert "retrieved_conv_ids" in question_data
        assert isinstance(question_data["retrieved_conv_ids"], list)

    def test_validate_conversation_structure(self):
        """Test that conversations have the expected structure."""
        runner = ImplexConvRunner(reasoning_type="opposed")
        test_file = Path("tests/bench/implex_conv_data/ImplexConv_opposed.json")

        if not test_file.exists():
            pytest.skip("ImplexConv_opposed.json not found")

        examples = runner.load_test_file(test_file)
        first_example = examples[0]

        # Check conversation structure
        conversations = first_example["conversation"]
        assert isinstance(conversations, dict)
        assert len(conversations) > 0

        # Each conversation should be a string
        for conv_id, conv_text in conversations.items():
            assert isinstance(conv_id, str)
            assert isinstance(conv_text, str)
            assert len(conv_text) > 0


class TestTokenCalculation:
    """Test token counting for efficiency metrics."""

    def test_calculate_total_tokens(self):
        """Test token calculation from conversations."""
        runner = ImplexConvRunner()

        conversations = {
            "0": "Speaker1: Hello\nAssistant: Hi there",
            "1": "Speaker1: How are you?\nAssistant: I'm good!",
        }

        total_tokens = runner._calculate_total_tokens(conversations)

        assert total_tokens > 0
        assert isinstance(total_tokens, int)

    def test_calculate_tokens_empty_conversation(self):
        """Test token calculation with empty conversations."""
        runner = ImplexConvRunner()

        conversations = {"0": ""}

        total_tokens = runner._calculate_total_tokens(conversations)

        assert total_tokens == 0


class TestFormatting:
    """Test output formatting utilities."""

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        runner = ImplexConvRunner()

        assert runner._format_duration(45.67) == "45.67s"
        assert runner._format_duration(1.23) == "1.23s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        runner = ImplexConvRunner()

        assert runner._format_duration(125.0) == "2m05s"
        assert runner._format_duration(60.0) == "1m00s"
        assert runner._format_duration(119.5) == "2m00s"  # Rounds up

    def test_format_duration_edge_cases(self):
        """Test duration formatting edge cases."""
        runner = ImplexConvRunner()

        assert runner._format_duration(0) == "0.00s"
        assert runner._format_duration(59.99) == "59.99s"
        assert runner._format_duration(60.01) == "1m00s"


class TestHonchoURLGeneration:
    """Test Honcho instance URL generation for load balancing."""

    def test_get_honcho_url_single_instance(self):
        """Test URL generation with a single instance."""
        runner = ImplexConvRunner(base_api_port=8000, pool_size=1)

        # All examples should use the same instance
        assert runner.get_honcho_url_for_index(0) == "http://localhost:8000"
        assert runner.get_honcho_url_for_index(1) == "http://localhost:8000"
        assert runner.get_honcho_url_for_index(10) == "http://localhost:8000"

    def test_get_honcho_url_multiple_instances(self):
        """Test URL generation with multiple instances (round-robin)."""
        runner = ImplexConvRunner(base_api_port=8000, pool_size=3)

        # Should distribute across 3 instances
        assert runner.get_honcho_url_for_index(0) == "http://localhost:8000"
        assert runner.get_honcho_url_for_index(1) == "http://localhost:8001"
        assert runner.get_honcho_url_for_index(2) == "http://localhost:8002"
        assert runner.get_honcho_url_for_index(3) == "http://localhost:8000"  # Wraps
        assert runner.get_honcho_url_for_index(4) == "http://localhost:8001"

    def test_get_honcho_url_custom_port(self):
        """Test URL generation with custom base port."""
        runner = ImplexConvRunner(base_api_port=9000, pool_size=2)

        assert runner.get_honcho_url_for_index(0) == "http://localhost:9000"
        assert runner.get_honcho_url_for_index(1) == "http://localhost:9001"


@pytest.mark.asyncio
class TestJudgmentLogic:
    """Test LLM-based judgment of responses."""

    async def test_judge_opposed_reasoning_pass(self):
        """Test judgment for opposed reasoning that should pass."""
        runner = ImplexConvRunner(reasoning_type="opposed")

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"passed": true, "found_implicit": true, "reasoning": "Response correctly identifies the constraint"}'
        mock_response.content = [mock_content]

        runner.anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judgment = await runner.judge_implicit_reasoning(
            question="What sports should I do?",
            expected_answer="Low-impact activities",
            actual_response="Given your injury, I'd suggest swimming or yoga",
            implicit_reasoning="broke my leg",
        )

        assert judgment["passed"] is True
        assert judgment["found_implicit"] is True
        assert "reasoning" in judgment

    async def test_judge_opposed_reasoning_fail(self):
        """Test judgment for opposed reasoning that should fail."""
        runner = ImplexConvRunner(reasoning_type="opposed")

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"passed": false, "found_implicit": false, "reasoning": "Response suggests activities that ignore the constraint"}'
        mock_response.content = [mock_content]

        runner.anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judgment = await runner.judge_implicit_reasoning(
            question="What sports should I do?",
            expected_answer="Low-impact activities",
            actual_response="You should try basketball and running!",
            implicit_reasoning="broke my leg",
        )

        assert judgment["passed"] is False
        assert judgment["found_implicit"] is False

    async def test_judge_supportive_reasoning(self):
        """Test judgment for supportive reasoning."""
        runner = ImplexConvRunner(reasoning_type="supportive")

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '{"passed": true, "found_implicit": true, "reasoning": "Response confirms the trait based on evidence"}'
        mock_response.content = [mock_content]

        runner.anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        judgment = await runner.judge_implicit_reasoning(
            question="Do I post facts daily?",
            expected_answer="Yes",
            actual_response="Yes, you've been consistently posting daily facts",
            implicit_reasoning=None,
        )

        assert judgment["passed"] is True
        assert "reasoning" in judgment

    async def test_judge_error_fallback(self):
        """Test judgment fallback on API error."""
        runner = ImplexConvRunner(reasoning_type="opposed")

        # Mock an API error
        runner.anthropic_client.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        judgment = await runner.judge_implicit_reasoning(
            question="Test question",
            expected_answer="test answer",
            actual_response="test answer is here",
            implicit_reasoning=None,
        )

        # Should fall back to string matching
        assert "passed" in judgment
        assert "found_implicit" in judgment
        assert judgment["found_implicit"] is False
        assert "Fallback string matching" in judgment["reasoning"]


class TestConversationToMessages:
    """Test conversion of conversation text to Honcho message format."""

    def test_message_role_mapping(self):
        """Test that Speaker1 maps to user and Assistant maps to assistant."""
        runner = ImplexConvRunner()

        conv_text = """Speaker1: User message
Assistant: Assistant response"""

        messages = runner._parse_conversation(conv_text)

        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_message_content_preservation(self):
        """Test that message content is preserved correctly."""
        runner = ImplexConvRunner()

        test_content = "This is a test message with special chars: !@#$%"
        conv_text = f"""Speaker1: {test_content}
Assistant: Got it!"""

        messages = runner._parse_conversation(conv_text)

        assert messages[0]["content"] == test_content

    def test_real_world_conversation_parsing(self):
        """Test parsing a real conversation from the dataset."""
        runner = ImplexConvRunner(reasoning_type="opposed")
        test_file = Path("tests/bench/implex_conv_data/ImplexConv_opposed.json")

        if not test_file.exists():
            pytest.skip("ImplexConv_opposed.json not found")

        examples = runner.load_test_file(test_file)
        first_conversation = examples[0]["conversation"]["0"]

        messages = runner._parse_conversation(first_conversation)

        # Should have parsed successfully
        assert len(messages) > 0
        # Should alternate between user and assistant (in most cases)
        assert all(msg["role"] in ["user", "assistant"] for msg in messages)
        # Should have non-empty content
        assert all(len(msg["content"]) > 0 for msg in messages)


class TestWorkspaceNaming:
    """Test workspace ID generation."""

    def test_workspace_id_format(self):
        """Test that workspace IDs follow the expected format."""
        example_id = "ex0"
        question_index = 1
        reasoning_type = "opposed"

        expected_workspace = f"{example_id}_q{question_index}_{reasoning_type}"
        assert expected_workspace == "ex0_q1_opposed"

    def test_workspace_id_uniqueness(self):
        """Test that different questions get different workspace IDs."""
        reasoning_type = "opposed"

        workspace_ids: set[str] = set()
        for ex_id in range(3):
            for q_idx in range(2):
                workspace_id = f"ex{ex_id}_q{q_idx}_{reasoning_type}"
                workspace_ids.add(workspace_id)

        # Should have 6 unique workspace IDs
        assert len(workspace_ids) == 6


class TestMetricsCollection:
    """Test metrics collection initialization."""

    def test_metrics_collector_initialized(self):
        """Test that metrics collector is initialized."""
        runner = ImplexConvRunner(reasoning_type="opposed")

        assert runner.metrics_collector is not None

    def test_runner_configuration(self):
        """Test runner configuration options."""
        runner = ImplexConvRunner(
            base_api_port=9000,
            pool_size=5,
            timeout_seconds=300,
            reasoning_type="supportive",
            cleanup_workspace=True,
            use_get_context=False,
        )

        assert runner.base_api_port == 9000
        assert runner.pool_size == 5
        assert runner.timeout_seconds == 300
        assert runner.reasoning_type == "supportive"
        assert runner.cleanup_workspace is True
        assert runner.use_get_context is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
