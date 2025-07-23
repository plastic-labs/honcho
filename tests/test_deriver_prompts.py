"""Tests for src/deriver/prompts.py"""

import datetime

from src.deriver.prompts import critical_analysis_prompt


class TestCriticalAnalysisPrompt:
    """Test cases for critical_analysis_prompt function"""

    def test_critical_analysis_prompt_returns_formatted_content(self):
        """Test that critical_analysis_prompt returns properly formatted content (covers line 35)"""
        # Arrange
        peer_name = "TestUser"
        message_created_at = datetime.datetime(2024, 1, 1, 12, 0, 0)
        context = "User has shown interest in AI"
        history = "User: Hello\nAgent: Hi there!"
        new_turn = "User: How are you?"

        # Act
        result = critical_analysis_prompt(
            peer_name=peer_name,
            message_created_at=message_created_at,
            context=context,
            history=history,
            new_turn=new_turn,
        )

        # Assert - mirascope prompt_template returns a list of BaseMessageParam objects
        assert isinstance(result, list)
        assert len(result) > 0

        # Extract the content from the first message param
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(content, str)
        assert peer_name in content
        assert str(message_created_at) in content
        assert context in content
        assert history in content
        assert new_turn in content

    def test_critical_analysis_prompt_contains_expected_sections(self):
        """Test that the prompt contains all expected sections and formatting"""
        # Arrange
        peer_name = "Alice"
        message_created_at = datetime.datetime(2024, 2, 15, 14, 30, 45)
        context = "Alice enjoys programming"
        history = "Alice: I love coding\nAgent: That's great!"
        new_turn = "Alice: What languages do you recommend?"

        # Act
        result = critical_analysis_prompt(
            peer_name=peer_name,
            message_created_at=message_created_at,
            context=context,
            history=history,
            new_turn=new_turn,
        )

        # Assert - Extract content and check for key sections
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert "You are an agent who critically analyzes user messages" in content
        assert f"The user's name is **{peer_name}**" in content
        assert "IMPORTANT NAMING RULES" in content
        assert "EXPLICIT REASONING" in content
        assert "DEDUCTIVE REASONING" in content
        assert "<current_context>" in content
        assert "<history>" in content
        assert "<new_turn>" in content

    def test_critical_analysis_prompt_with_special_characters(self):
        """Test that the prompt handles special characters properly"""
        # Arrange
        peer_name = "User@123"
        message_created_at = datetime.datetime(2024, 12, 25, 23, 59, 59)
        context = "Context with special chars: !@#$%^&*()"
        history = "User: How's the weather? It's 75°F\nAgent: Nice temperature!"
        new_turn = "User: What about tomorrow's forecast?"

        # Act
        result = critical_analysis_prompt(
            peer_name=peer_name,
            message_created_at=message_created_at,
            context=context,
            history=history,
            new_turn=new_turn,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert context in content
        assert history in content
        assert new_turn in content

    def test_critical_analysis_prompt_with_empty_strings(self):
        """Test that the prompt handles empty strings gracefully"""
        # Arrange
        peer_name = ""
        message_created_at = datetime.datetime(2024, 6, 30, 0, 0, 0)
        context = ""
        history = ""
        new_turn = ""

        # Act
        result = critical_analysis_prompt(
            peer_name=peer_name,
            message_created_at=message_created_at,
            context=context,
            history=history,
            new_turn=new_turn,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert len(content) > 0  # Should still have template content
        assert str(message_created_at) in content

    def test_critical_analysis_prompt_with_multiline_content(self):
        """Test that the prompt handles multiline content properly"""
        # Arrange
        peer_name = "MultilineUser"
        message_created_at = datetime.datetime(2024, 3, 10, 10, 15, 30)
        context = "User likes:\n- Programming\n- Reading\n- Gaming"
        history = "User: I have multiple hobbies\nAgent: Tell me more\nUser: I code and read books"
        new_turn = "User: I also enjoy:\n- Video games\n- Hiking\n- Cooking"

        # Act
        result = critical_analysis_prompt(
            peer_name=peer_name,
            message_created_at=message_created_at,
            context=context,
            history=history,
            new_turn=new_turn,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert "Programming" in content
        assert "Video games" in content
        assert "Tell me more" in content
