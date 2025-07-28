"""Tests for src/dialectic/prompts.py"""

from src.dialectic.prompts import dialectic_prompt, query_generation_prompt


class TestDialecticPrompt:
    """Test cases for dialectic_prompt function"""

    def test_dialectic_prompt_returns_formatted_content_with_additional_context(self):
        """Test that dialectic_prompt returns properly formatted content with additional context (covers line 25)"""
        # Arrange
        query = "What are the user's interests?"
        working_representation = "User likes programming and reading"
        additional_context = "User has expressed interest in AI and machine learning"
        peer_name = "TestUser"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert - mirascope prompt_template returns a list of BaseMessageParam objects
        assert isinstance(result, list)
        assert len(result) > 0

        # Extract the content from the first message param
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(content, str)
        assert peer_name in content
        assert query in content
        assert working_representation in content
        assert additional_context in content
        assert f"The query is about user {peer_name}" in content
        assert "<query>" in content
        assert "<working_representation>" in content
        assert "<global_context>" in content

    def test_dialectic_prompt_returns_formatted_content_without_additional_context(
        self,
    ):
        """Test that dialectic_prompt returns properly formatted content without additional context (covers line 25)"""
        # Arrange
        query = "What does the user prefer for breakfast?"
        working_representation = "User mentioned liking coffee and toast"
        additional_context = None
        peer_name = "Alice"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert query in content
        assert working_representation in content
        assert f"The query is about user {peer_name}" in content
        assert "<query>" in content
        assert "<working_representation>" in content
        # Should NOT contain global_context section when additional_context is None
        assert "<global_context>" not in content

    def test_dialectic_prompt_with_empty_additional_context(self):
        """Test that dialectic_prompt handles empty string additional_context (covers line 25)"""
        # Arrange
        query = "How does the user handle feedback?"
        working_representation = "User seems receptive to constructive criticism"
        additional_context = ""
        peer_name = "Bob"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert query in content
        assert working_representation in content
        # Empty string should be treated as falsy and not include global_context
        assert "<global_context>" not in content

    def test_dialectic_prompt_contains_expected_sections(self):
        """Test that the prompt contains all expected sections and formatting (covers line 25)"""
        # Arrange
        query = "What are the user's communication preferences?"
        working_representation = "User prefers direct communication"
        additional_context = "Historically prefers email over phone calls"
        peer_name = "Charlie"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert - Extract content and check for key sections
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert "You are a context synthesis agent" in content
        assert "## INPUT STRUCTURE" in content
        assert "## OUTPUT FORMAT" in content
        assert "- **Query**: The specific question" in content
        assert "- **Working Representation**: Current session conclusions" in content
        assert "- **Additional Context**: Historical conclusions" in content
        assert "- **Conclusion**: The derived insight" in content
        assert "- **Premises**: Supporting evidence" in content
        assert "- **Type**: Either Explicit or Deductive" in content

    def test_dialectic_prompt_with_special_characters(self):
        """Test that the prompt handles special characters properly (covers line 25)"""
        # Arrange
        query = "How does the user feel about symbols like @, #, $, %?"
        working_representation = "User uses emojis frequently: 😀 🚀 💻"
        additional_context = "User's name contains special chars: User@123"
        peer_name = "User@123"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert "😀 🚀 💻" in content
        assert "@, #, $, %" in content
        assert "User@123" in content

    def test_dialectic_prompt_with_multiline_content(self):
        """Test that the prompt handles multiline content properly (covers line 25)"""
        # Arrange
        query = "What are the user's project management skills?"
        working_representation = """User demonstrates:
- Strong planning abilities
- Good time management
- Clear communication"""
        additional_context = """Previous observations:
- Led successful team projects
- Meets deadlines consistently
- Provides clear updates"""
        peer_name = "ProjectManager"

        # Act
        result = dialectic_prompt(
            query=query,
            working_representation=working_representation,
            additional_context=additional_context,
            peer_name=peer_name,
        )

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert "Strong planning abilities" in content
        assert "Led successful team projects" in content
        assert "Clear communication" in content


class TestQueryGenerationPrompt:
    """Test cases for query_generation_prompt function"""

    def test_query_generation_prompt_returns_formatted_content(self):
        """Test that query_generation_prompt returns properly formatted content (covers line 70)"""
        # Arrange
        query = "What does the user like to do for fun?"
        peer_name = "TestUser"

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert query in content
        assert f"The user's name is {peer_name}" in content
        assert "You are a query expansion agent" in content

    def test_query_generation_prompt_with_special_characters(self):
        """Test that query_generation_prompt handles special characters in query and peer_name (covers line 70)"""
        # Arrange
        query = "How does User@123 handle symbols like @, #, $, %?"
        peer_name = "User@123"

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert query in content
        assert f"The user's name is {peer_name}" in content
        assert "@, #, $, %" in content

    def test_query_generation_prompt_with_multiline_query(self):
        """Test that query_generation_prompt handles multiline queries (covers line 70)"""
        # Arrange
        query = """What are the user's preferences for:
- Communication style
- Work environment
- Learning methods"""
        peer_name = "MultilineUser"

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert "Communication style" in content
        assert "Work environment" in content
        assert "Learning methods" in content
        assert f"The user's name is {peer_name}" in content

    def test_query_generation_prompt_contains_expected_sections(self):
        """Test that query_generation_prompt contains all expected sections and formatting (covers line 70)"""
        # Arrange
        query = "What are the user's career goals?"
        peer_name = "CareerUser"

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert - Extract content and check for key sections
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert "## QUERY EXPANSION STRATEGY FOR SEMANTIC SIMILARITY" in content
        assert "**Your Goal**: Generate 3-5 complementary search queries" in content
        assert "**Semantic Similarity Optimization**:" in content
        assert "1. **Analyze the Application Query**:" in content
        assert "2. **Think Conceptually**:" in content
        assert "3. **Consider Language Patterns in Stored Observations**:" in content
        assert "4. **Vary Semantic Scope**" in content
        assert "**Vocabulary Expansion Techniques**:" in content
        assert "## OUTPUT FORMAT" in content
        assert '{"queries": ["query1", "query2", "query3"]}' in content
        assert "<query>" in content and "</query>" in content

    def test_query_generation_prompt_with_empty_strings(self):
        """Test that query_generation_prompt handles empty strings (covers line 70)"""
        # Arrange
        query = ""
        peer_name = ""

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert (
            "The user's name is " in content
        )  # Empty peer_name should still appear in template
        assert (
            "<query></query>" in content
        )  # Empty query should still appear in template

    def test_query_generation_prompt_with_unicode_characters(self):
        """Test that query_generation_prompt handles unicode characters (covers line 70)"""
        # Arrange
        query = (
            "What does the user think about émojis like 😀 and símbolos like ñ, ü, ç?"
        )
        peer_name = "Ünïcödé_Üser"

        # Act
        result = query_generation_prompt(query=query, peer_name=peer_name)

        # Assert
        content = result[0].content if hasattr(result[0], "content") else str(result[0])

        assert isinstance(result, list)
        assert isinstance(content, str)
        assert peer_name in content
        assert "😀" in content
        assert "ñ, ü, ç" in content
        assert "émojis" in content
        assert "símbolos" in content
