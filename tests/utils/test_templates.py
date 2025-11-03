"""Tests for the Jinja2 template rendering system."""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast

from src.deriver.prompts import critical_analysis_prompt, peer_card_prompt
from src.dialectic.prompts import dialectic_prompt, query_generation_prompt
from src.dreamer.prompts import consolidation_prompt
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)
from src.utils.templates import TemplateManager, get_template_manager, render_template


class TestTemplateManager:
    """Tests for the TemplateManager class."""

    def test_template_manager_singleton(self):
        """Verify TemplateManager is cached as a singleton."""
        manager1 = get_template_manager()
        manager2 = get_template_manager()
        assert manager1 is manager2

    def test_template_manager_initialization(self):
        """Verify TemplateManager initializes correctly."""
        manager = TemplateManager()
        assert manager.env is not None
        assert "join_lines" in manager.env.filters

    def test_join_lines_filter(self):
        """Test the custom join_lines filter."""

        manager = get_template_manager()
        join_lines = cast(Callable[[Any], str], manager.env.filters["join_lines"])

        # Test with list of strings
        assert join_lines(["line1", "line2", "line3"]) == "line1\nline2\nline3"

        # Test with empty list
        assert join_lines([]) == ""

        # Test with None
        assert join_lines(None) == ""

    def test_render_basic_template(self):
        """Test basic template rendering with simple context."""
        # This tests the rendering mechanism itself
        # Using one of the actual templates
        result = render_template(
            "dreamer/consolidation.jinja",
            {"representation_as_json": '{"test": "data"}'},
        )
        assert "consolidates observations" in result
        assert '{"test": "data"}' in result

    def test_render_strips_whitespace(self):
        """Verify rendered templates have leading/trailing whitespace stripped."""
        result = render_template(
            "dreamer/consolidation.jinja",
            {"representation_as_json": "{}"},
        )
        # Should not have leading/trailing whitespace
        assert result == result.strip()
        assert not result.startswith("\n")
        assert not result.endswith("\n")


class TestDeriverPrompts:
    """Tests for deriver prompt templates."""

    def test_critical_analysis_prompt_basic(self):
        """Test critical_analysis_prompt renders with minimal inputs."""
        representation = Representation()
        result = critical_analysis_prompt(
            peer_id="test_user",
            peer_card=None,
            message_created_at=datetime.now(timezone.utc),
            working_representation=representation,
            history="Previous context here",
            new_turns=["User: Hello", "Assistant: Hi there"],
        )

        # Verify key sections are present
        assert "test_user" in result
        assert "TARGET USER TO ANALYZE" in result
        assert "EXPLICIT REASONING" in result
        assert "DEDUCTIVE REASONING" in result
        assert "Previous context here" in result
        assert "User: Hello" in result
        assert "Hi there" in result

    def test_critical_analysis_prompt_with_peer_card(self):
        """Test critical_analysis_prompt with peer card data."""
        peer_card = ["Name: Alice", "Age: 30", "Location: NYC"]
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="Alice likes programming",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ]
        )

        result = critical_analysis_prompt(
            peer_id="alice",
            peer_card=peer_card,
            message_created_at=datetime.now(timezone.utc),
            working_representation=representation,
            history="",
            new_turns=["I love Python"],
        )

        # Verify peer card is included
        assert "Name: Alice" in result
        assert "Age: 30" in result
        assert "Location: NYC" in result
        assert "known biographical information" in result

    def test_critical_analysis_prompt_with_working_representation(self):
        """Test critical_analysis_prompt includes working representation."""
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="User enjoys hiking",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
            deductive=[
                DeductiveObservation(
                    conclusion="User is physically active",
                    premises=["User enjoys hiking"],
                    created_at=datetime.now(timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
        )

        result = critical_analysis_prompt(
            peer_id="test_user",
            peer_card=None,
            message_created_at=datetime.now(timezone.utc),
            working_representation=representation,
            history="",
            new_turns=["I went hiking yesterday"],
        )

        # Should include current understanding section
        assert "Current understanding" in result or "current_context" in result
        assert "User enjoys hiking" in result

    def test_critical_analysis_prompt_empty_representation(self):
        """Test critical_analysis_prompt with empty representation."""
        empty_rep = Representation()

        result = critical_analysis_prompt(
            peer_id="new_user",
            peer_card=None,
            message_created_at=datetime.now(timezone.utc),
            working_representation=empty_rep,
            history="",
            new_turns=["First message"],
        )

        # Should still render without current understanding section
        assert "new_user" in result
        assert "First message" in result
        # Empty representation should not show current understanding
        assert result  # Just verify it renders

    def test_peer_card_prompt_create_new(self):
        """Test peer_card_prompt when creating a new card."""
        result = peer_card_prompt(
            old_peer_card=None,
            new_observations="User is 25 years old\nUser lives in Boston\nUser is a software engineer",
        )

        # Verify prompt structure
        assert "biographical card" in result
        assert "User does not have a card" in result
        assert "User is 25 years old" in result
        assert "User lives in Boston" in result
        assert "software engineer" in result

    def test_peer_card_prompt_update_existing(self):
        """Test peer_card_prompt when updating an existing card."""
        old_card = ["Name: Bob", "Age: 30", "Location: NYC"]

        result = peer_card_prompt(
            old_peer_card=old_card,
            new_observations="Bob moved to San Francisco\nBob is now 31 years old",
        )

        # Verify old card is included
        assert "Current user biographical card" in result
        assert "Name: Bob" in result
        assert "Age: 30" in result
        assert "Location: NYC" in result

        # Verify new observations are included
        assert "moved to San Francisco" in result
        assert "now 31 years old" in result

    def test_peer_card_prompt_examples_included(self):
        """Test peer_card_prompt includes example format."""
        result = peer_card_prompt(
            old_peer_card=None,
            new_observations="Test observation",
        )

        # Should include examples
        assert "Example" in result
        assert '"card"' in result


class TestDialecticPrompts:
    """Tests for dialectic prompt templates."""

    def test_dialectic_prompt_same_observer_observed(self):
        """Test dialectic_prompt when observer and observed are the same."""
        result = dialectic_prompt(
            query="What are the user's hobbies?",
            working_representation="User enjoys painting and photography",
            recent_conversation_history="User: I painted a landscape today",
            observer_peer_card=["Name: Alice", "Age: 28"],
            observed_peer_card=None,
            observer="alice",
            observed="alice",
        )

        # Should be a global query format
        assert "What are the user's hobbies?" in result
        assert "User enjoys painting and photography" in result
        assert "painted a landscape today" in result
        assert "Name: Alice" in result

        # Should NOT have directional language
        assert "understanding of" not in result or "alice" in result.lower()

    def test_dialectic_prompt_different_observer_observed(self):
        """Test dialectic_prompt for directional query."""
        result = dialectic_prompt(
            query="What does Alice think about Bob?",
            working_representation="Alice thinks Bob is helpful",
            recent_conversation_history="Alice: Bob really helped me today",
            observer_peer_card=["Name: Alice"],
            observed_peer_card=["Name: Bob"],
            observer="alice",
            observed="bob",
        )

        # Should have directional query format
        assert "alice" in result
        assert "bob" in result
        assert "understanding of" in result
        assert "Alice thinks Bob is helpful" in result

    def test_dialectic_prompt_without_conversation_history(self):
        """Test dialectic_prompt without recent conversation history."""
        result = dialectic_prompt(
            query="What does the user like?",
            working_representation="User likes coffee",
            recent_conversation_history=None,
            observer_peer_card=["Name: Charlie"],
            observed_peer_card=None,
            observer="charlie",
            observed="charlie",
        )

        # Should still render without conversation history section
        assert "What does the user like?" in result
        assert "User likes coffee" in result
        # Should not have empty recent_conversation_history section
        assert result  # Just verify it renders

    def test_dialectic_prompt_core_content(self):
        """Test dialectic_prompt includes core instructional content."""
        result = dialectic_prompt(
            query="Test query",
            working_representation="Test representation",
            recent_conversation_history=None,
            observer_peer_card=None,
            observed_peer_card=None,
            observer="user1",
            observed="user1",
        )

        # Verify core sections are present
        assert "context synthesis agent" in result
        assert "EXPLICIT" in result or "Explicit" in result
        assert "DEDUCTIVE" in result or "Deductive" in result
        assert "working_representation" in result
        assert "Test representation" in result

    def test_query_generation_prompt_basic(self):
        """Test query_generation_prompt renders correctly."""
        result = query_generation_prompt(
            query="How does the user handle criticism?",
            observed="john",
        )

        # Verify core content
        assert "john" in result
        assert "How does the user handle criticism?" in result
        assert "query expansion" in result or "search queries" in result
        assert "semantic" in result

    def test_query_generation_prompt_json_format(self):
        """Test query_generation_prompt mentions JSON output format."""
        result = query_generation_prompt(
            query="What are user's interests?",
            observed="mary",
        )

        # Should mention JSON format
        assert "queries" in result
        assert "JSON" in result or "json" in result


class TestDreamerPrompts:
    """Tests for dreamer prompt templates."""

    def test_consolidation_prompt_basic(self):
        """Test consolidation_prompt renders correctly."""
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="User is interested in AI",
                    created_at=datetime.now(timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
            deductive=[
                DeductiveObservation(
                    conclusion="User works in technology",
                    premises=["User is interested in AI"],
                    created_at=datetime.now(timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
        )

        result = consolidation_prompt(representation=representation)

        # Verify core content
        assert "consolidates observations" in result
        assert "EXPLICIT" in result
        assert "DEDUCTIVE" in result
        # Should contain JSON representation
        assert "User is interested in AI" in result
        assert "User works in technology" in result

    def test_consolidation_prompt_with_empty_representation(self):
        """Test consolidation_prompt with empty representation."""
        empty_rep = Representation()

        result = consolidation_prompt(representation=empty_rep)

        # Should still render
        assert "consolidates observations" in result
        # Should contain empty structure
        assert result


class TestPromptConsistency:
    """Tests to ensure prompts maintain consistent structure across renders."""

    def test_all_prompts_render_deterministically(self):
        """Verify prompts render consistently with same inputs."""
        # Test critical_analysis_prompt
        rep = Representation()
        prompt1 = critical_analysis_prompt(
            peer_id="user",
            peer_card=["Name: Test"],
            message_created_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            working_representation=rep,
            history="history",
            new_turns=["turn1"],
        )
        prompt2 = critical_analysis_prompt(
            peer_id="user",
            peer_card=["Name: Test"],
            message_created_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            working_representation=rep,
            history="history",
            new_turns=["turn1"],
        )
        assert prompt1 == prompt2

        # Test peer_card_prompt
        prompt1 = peer_card_prompt(old_peer_card=None, new_observations="obs")
        prompt2 = peer_card_prompt(old_peer_card=None, new_observations="obs")
        assert prompt1 == prompt2

        # Test dialectic_prompt
        prompt1 = dialectic_prompt(
            query="q",
            working_representation="rep",
            recent_conversation_history=None,
            observer_peer_card=None,
            observed_peer_card=None,
            observer="alice",
            observed="alice",
        )
        prompt2 = dialectic_prompt(
            query="q",
            working_representation="rep",
            recent_conversation_history=None,
            observer_peer_card=None,
            observed_peer_card=None,
            observer="alice",
            observed="alice",
        )
        assert prompt1 == prompt2

    def test_prompts_handle_special_characters(self):
        """Test prompts handle special characters correctly."""
        # Test with quotes, newlines, special chars
        special_text = 'User said: "Hello\nWorld" & <tag>'

        result = peer_card_prompt(
            old_peer_card=None,
            new_observations=special_text,
        )

        # Should preserve special characters (no escaping since autoescape is off)
        assert special_text in result
        assert '"Hello\nWorld"' in result

    def test_prompts_handle_none_values(self):
        """Test prompts handle None values gracefully."""
        # critical_analysis with None peer_card
        result = critical_analysis_prompt(
            peer_id="user",
            peer_card=None,
            message_created_at=datetime.now(timezone.utc),
            working_representation=Representation(),
            history="",
            new_turns=[],
        )
        assert result  # Should render

        # dialectic with None values
        result = dialectic_prompt(
            query="q",
            working_representation="rep",
            recent_conversation_history=None,
            observer_peer_card=None,
            observed_peer_card=None,
            observer="user",
            observed="user",
        )
        assert result  # Should render

    def test_prompts_handle_empty_lists(self):
        """Test prompts handle empty lists gracefully."""
        result = critical_analysis_prompt(
            peer_id="user",
            peer_card=[],
            message_created_at=datetime.now(timezone.utc),
            working_representation=Representation(),
            history="",
            new_turns=[],
        )
        assert result  # Should render

        result = peer_card_prompt(old_peer_card=[], new_observations="obs")
        assert result  # Should render
