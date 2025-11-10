"""Tests for the Jinja2 template rendering system."""

import json
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
        created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = critical_analysis_prompt(
            peer_id="test_user",
            peer_card=None,
            message_created_at=created_at,
            working_representation=representation,
            history="Previous context here",
            new_turns=["User: Hello", "Assistant: Hi there"],
        )

        # Verify key sections are present
        assert "test_user" in result
        assert "TARGET USER TO ANALYZE" in result
        assert "EXPLICIT REASONING" in result
        assert "DEDUCTIVE REASONING" in result
        assert "2025-01-01 00:00:00+00:00" in result
        assert "<history>\nPrevious context here\n</history>" in result
        assert "<new_turns>\nUser: Hello\nAssistant: Hi there\n</new_turns>" in result
        assert "<peer_card>" not in result
        assert "<current_context>" not in result

    def test_critical_analysis_prompt_with_peer_card(self):
        """Test critical_analysis_prompt with peer card data."""
        peer_card = ["Name: Alice", "Age: 30", "Location: NYC"]
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="Alice likes programming",
                    created_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
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
        peer_card_block = (
            "<peer_card>\nName: Alice\nAge: 30\nLocation: NYC\n</peer_card>"
        )
        assert peer_card_block in result
        assert "known biographical information" in result
        assert "<current_context>" in result

    def test_critical_analysis_prompt_with_working_representation(self):
        """Test critical_analysis_prompt includes working representation."""
        explicit_created_at = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        deductive_created_at = datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc)
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="User enjoys hiking",
                    created_at=explicit_created_at,
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
            deductive=[
                DeductiveObservation(
                    conclusion="User is physically active",
                    premises=["User enjoys hiking"],
                    created_at=deductive_created_at,
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
        assert "<current_context>" in result
        assert (
            "EXPLICIT:\n\n1. [2025-01-01 12:00:00+00:00] User enjoys hiking" in result
        )
        assert (
            "DEDUCTIVE:\n\n1. [2025-01-01 12:05:00+00:00] User is physically active"
            in result
        )
        assert "    - User enjoys hiking" in result
        assert "</current_context>" in result
        assert "<peer_card>" not in result

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
        assert "<current_context>" not in result

    def test_peer_card_prompt_create_new(self):
        """Test peer_card_prompt when creating a new card."""
        result = peer_card_prompt(
            old_peer_card=None,
            new_observations="User is 25 years old\nUser lives in Boston\nUser is a software engineer",
        )

        # Verify prompt structure
        assert "biographical card" in result
        assert (
            "User does not have a card. Create one with any key observations." in result
        )
        observations_block = (
            "New observations:\n\n"
            "User is 25 years old\n"
            "User lives in Boston\n"
            "User is a software engineer\n"
        )
        assert observations_block in result

    def test_peer_card_prompt_update_existing(self):
        """Test peer_card_prompt when updating an existing card."""
        old_card = ["Name: Bob", "Age: 30", "Location: NYC"]

        result = peer_card_prompt(
            old_peer_card=old_card,
            new_observations="Bob moved to San Francisco\nBob is now 31 years old",
        )

        # Verify old card is included
        existing_card_block = (
            "Current user biographical card:\n"
            "Name: Bob\n"
            "Age: 30\n"
            "Location: NYC\n"
        )
        assert existing_card_block in result

        # Verify new observations are included
        observations_block = (
            "New observations:\n\n"
            "Bob moved to San Francisco\n"
            "Bob is now 31 years old\n"
        )
        assert observations_block in result

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
        assert "<query>What are the user's hobbies?</query>" in result
        assert (
            "<working_representation>User enjoys painting and photography</working_representation>"
            in result
        )
        assert (
            "<recent_conversation_history>\n"
            "User: I painted a landscape today\n"
            "</recent_conversation_history>"
        ) in result
        assert "The query is about user alice." in result
        assert (
            "The user's known biographical information:\n" "Name: Alice\n" "Age: 28\n"
        ) in result

        # Should NOT have directional language
        assert "alice's understanding of" not in result
        assert "The target's known biographical information" not in result

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
        assert "alice's understanding of bob" in result.lower()
        assert (
            "<working_representation>Alice thinks Bob is helpful</working_representation>"
            in result
        )
        assert (
            "The user's known biographical information:\n" "Name: Alice\n"
        ) in result
        assert (
            "The target's known biographical information:\n" "Name: Bob\n"
        ) in result

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
        assert (
            "<working_representation>User likes coffee</working_representation>"
            in result
        )
        # Should not have empty recent_conversation_history section
        assert "<recent_conversation_history>" not in result

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
        assert "## INPUT STRUCTURE" in result
        assert "## OUTPUT FORMAT" in result
        assert (
            "<working_representation>Test representation</working_representation>"
            in result
        )
        assert "<query>Test query</query>" in result

    def test_query_generation_prompt_basic(self):
        """Test query_generation_prompt renders correctly."""
        result = query_generation_prompt(
            query="How does the user handle criticism?",
            observed="john",
        )

        # Verify core content
        assert "john" in result
        assert "query expansion agent" in result
        assert "<query>How does the user handle criticism?</query>" in result
        assert "semantic search over an embedding store" in result

    def test_query_generation_prompt_json_format(self):
        """Test query_generation_prompt mentions JSON output format."""
        result = query_generation_prompt(
            query="What are user's interests?",
            observed="mary",
        )

        # Should mention JSON format
        assert "queries" in result
        assert "Respond with 3-5 search queries as a JSON object" in result
        assert "<query>What are user's interests?</query>" in result


class TestDreamerPrompts:
    """Tests for dreamer prompt templates."""

    def test_consolidation_prompt_basic(self):
        """Test consolidation_prompt renders correctly."""
        representation = Representation(
            explicit=[
                ExplicitObservation(
                    content="User is interested in AI",
                    created_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test",
                )
            ],
            deductive=[
                DeductiveObservation(
                    conclusion="User works in technology",
                    premises=["User is interested in AI"],
                    created_at=datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc),
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
        json_block = result[result.index("{") :]
        data = json.loads(json_block)
        assert data["explicit"][0]["content"] == "User is interested in AI"
        assert data["deductive"][0]["conclusion"] == "User works in technology"
        assert data["deductive"][0]["premises"] == ["User is interested in AI"]

    def test_consolidation_prompt_with_empty_representation(self):
        """Test consolidation_prompt with empty representation."""
        empty_rep = Representation()

        result = consolidation_prompt(representation=empty_rep)

        # Should still render
        assert "consolidates observations" in result
        # Should contain empty structure
        json_block = result[result.index("{") :]
        data = json.loads(json_block)
        assert data == {"explicit": [], "deductive": []}


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
