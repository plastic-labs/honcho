"""Tests for the developer feedback channel (Phase 4 of Agentic FDE)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.feedback import (
    INTERVIEW_QUESTIONS,
    FeedbackChange,
    FeedbackLLMResponse,
    build_feedback_prompt,
    config_is_empty,
    is_simple_greeting,
    process_feedback,
)
from src.schemas import (
    ConfigChange,
    FeedbackRequest,
    FeedbackResponse,
    IntrospectionReport,
    IntrospectionSignals,
    IntrospectionSuggestion,
    WorkspaceAgentConfig,
)


class TestFeedbackSchemas:
    """Test the feedback-related Pydantic schemas."""

    def test_feedback_request_basic(self):
        """Test basic FeedbackRequest creation."""
        request = FeedbackRequest(message="Hello")
        assert request.message == "Hello"
        assert request.include_introspection is False

    def test_feedback_request_with_introspection(self):
        """Test FeedbackRequest with introspection enabled."""
        request = FeedbackRequest(
            message="Configure my workspace", include_introspection=True
        )
        assert request.message == "Configure my workspace"
        assert request.include_introspection is True

    def test_feedback_request_validation(self):
        """Test FeedbackRequest validation."""
        # Empty message should fail
        with pytest.raises(ValidationError):
            FeedbackRequest(message="")

        # Very long message should fail
        with pytest.raises(ValidationError):
            FeedbackRequest(message="x" * 10001)

    def test_config_change(self):
        """Test ConfigChange schema."""
        change = ConfigChange(
            field="deriver_rules",
            previous_value="old rule",
            new_value="new rule",
        )
        assert change.field == "deriver_rules"
        assert change.previous_value == "old rule"
        assert change.new_value == "new rule"

    def test_config_change_dialectic(self):
        """Test ConfigChange for dialectic_rules."""
        change = ConfigChange(
            field="dialectic_rules",
            previous_value="",
            new_value="Be concise",
        )
        assert change.field == "dialectic_rules"

    def test_feedback_response(self):
        """Test FeedbackResponse schema."""
        response = FeedbackResponse(
            message="Configuration updated",
            understood_intent="Add focus on emotions",
            changes_made=[
                ConfigChange(
                    field="deriver_rules",
                    previous_value="",
                    new_value="Focus on emotions",
                )
            ],
            current_config=WorkspaceAgentConfig(deriver_rules="Focus on emotions"),
        )
        assert response.message == "Configuration updated"
        assert len(response.changes_made) == 1
        assert response.current_config.deriver_rules == "Focus on emotions"


class TestHelperFunctions:
    """Test helper functions for feedback processing."""

    def test_is_simple_greeting_true_cases(self):
        """Test cases that should be detected as simple greetings."""
        greetings = [
            "hello",
            "Hello",
            "HELLO",
            "hi",
            "Hi there",
            "hey",
            "Hey!",
            "good morning",
            "Good afternoon",
            "howdy",
            "help",
            "Help me",
            "how do i configure",
            "configure",
            "setup",
            "get started",
        ]
        for greeting in greetings:
            assert is_simple_greeting(
                greeting
            ), f"Expected '{greeting}' to be a greeting"

    def test_is_simple_greeting_false_cases(self):
        """Test cases that should NOT be detected as simple greetings."""
        non_greetings = [
            "I'm building a journaling app and want to focus on emotions",
            "The deriver should extract technical facts only",
            "x" * 101,  # Too long
            "Please configure the workspace to focus on user preferences",
        ]
        for msg in non_greetings:
            assert not is_simple_greeting(msg), f"Expected '{msg}' to NOT be a greeting"

    def test_config_is_empty_true(self):
        """Test detecting empty configuration."""
        config = WorkspaceAgentConfig()
        assert config_is_empty(config)

        config = WorkspaceAgentConfig(deriver_rules="", dialectic_rules="")
        assert config_is_empty(config)

        config = WorkspaceAgentConfig(deriver_rules="   ", dialectic_rules="  ")
        assert config_is_empty(config)

    def test_config_is_empty_false(self):
        """Test detecting non-empty configuration."""
        config = WorkspaceAgentConfig(deriver_rules="Some rule")
        assert not config_is_empty(config)

        config = WorkspaceAgentConfig(dialectic_rules="Another rule")
        assert not config_is_empty(config)


class TestBuildFeedbackPrompt:
    """Test the prompt building function."""

    def test_basic_prompt(self):
        """Test basic prompt without introspection."""
        prompt = build_feedback_prompt(
            message="Focus on emotions",
            current_config=WorkspaceAgentConfig(),
        )
        assert "Focus on emotions" in prompt
        assert "(empty - using defaults)" in prompt
        assert "Developer Message" in prompt

    def test_prompt_with_existing_config(self):
        """Test prompt includes existing configuration."""
        config = WorkspaceAgentConfig(
            deriver_rules="Extract technical facts",
            dialectic_rules="Be concise",
        )
        prompt = build_feedback_prompt(
            message="Add emotion tracking",
            current_config=config,
        )
        assert "Extract technical facts" in prompt
        assert "Be concise" in prompt

    def test_prompt_with_introspection(self):
        """Test prompt includes introspection report when provided."""
        import datetime

        report = IntrospectionReport(
            workspace_name="test",
            generated_at=datetime.datetime.now(datetime.timezone.utc),
            performance_summary="Good performance overall",
            identified_issues=["High abstention rate"],
            suggestions=[
                IntrospectionSuggestion(
                    target="deriver_rules",
                    current_value="",
                    suggested_value="Focus more",
                    rationale="Would reduce abstentions",
                    confidence="high",
                )
            ],
            signals=IntrospectionSignals(),
        )

        prompt = build_feedback_prompt(
            message="Help me improve",
            current_config=WorkspaceAgentConfig(),
            introspection_report=report,
        )
        assert "Good performance overall" in prompt
        assert "High abstention rate" in prompt
        assert "Would reduce abstentions" in prompt


class TestProcessFeedback:
    """Test the main feedback processing function."""

    @pytest.mark.asyncio
    async def test_interview_mode_trigger(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that interview mode is triggered for empty config + greeting."""
        workspace, _ = sample_data

        request = FeedbackRequest(message="Hello")
        response = await process_feedback(db_session, workspace.name, request)

        assert INTERVIEW_QUESTIONS in response.message
        assert "First-time setup" in response.understood_intent
        assert len(response.changes_made) == 0

    @pytest.mark.asyncio
    async def test_interview_mode_not_triggered_with_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that interview mode is NOT triggered when config exists."""
        workspace, _ = sample_data

        # Set up existing config
        config = WorkspaceAgentConfig(deriver_rules="Existing rule")
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        # Mock the LLM call (structured output)
        mock_response = MagicMock()
        mock_response.content = FeedbackLLMResponse(
            message="I see you already have rules set up.",
            understood_intent="Greeting with existing config",
            changes=[],
        )

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = FeedbackRequest(message="Hello")
            response = await process_feedback(db_session, workspace.name, request)

            # Should NOT be interview mode
            assert INTERVIEW_QUESTIONS not in response.message

    @pytest.mark.asyncio
    async def test_config_update(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that config changes are applied correctly."""
        workspace, _ = sample_data

        # Mock the LLM call to return a config change (structured output)
        mock_response = MagicMock()
        mock_response.content = FeedbackLLMResponse(
            message="I've configured the workspace to focus on emotions.",
            understood_intent="Configure deriver for emotion tracking",
            changes=[
                FeedbackChange(
                    field="deriver_rules",
                    new_value="Focus on emotional content and feelings",
                )
            ],
        )

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = FeedbackRequest(
                message="I'm building a journaling app, focus on emotions"
            )
            response = await process_feedback(db_session, workspace.name, request)

            assert len(response.changes_made) == 1
            assert response.changes_made[0].field == "deriver_rules"
            assert "emotion" in response.changes_made[0].new_value.lower()

            # Verify config was actually saved
            saved_config = await crud.get_workspace_agent_config(
                db_session, workspace.name
            )
            assert "emotion" in saved_config.deriver_rules.lower()

    @pytest.mark.asyncio
    async def test_question_no_changes(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that questions don't result in config changes."""
        workspace, _ = sample_data

        # Set up existing config
        config = WorkspaceAgentConfig(deriver_rules="Existing rule")
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        # Mock LLM to return answer without changes (structured output)
        mock_response = MagicMock()
        mock_response.content = FeedbackLLMResponse(
            message="Your current deriver rule is: 'Existing rule'",
            understood_intent="Question about current config",
            changes=[],
        )

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = FeedbackRequest(message="What are my current rules?")
            response = await process_feedback(db_session, workspace.name, request)

            assert len(response.changes_made) == 0
            # Config should be unchanged
            assert response.current_config.deriver_rules == "Existing rule"

    @pytest.mark.asyncio
    async def test_llm_error_handling(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test graceful handling of LLM errors."""
        workspace, _ = sample_data

        # Set initial config
        config = WorkspaceAgentConfig(deriver_rules="Existing rule")
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            side_effect=Exception("LLM service unavailable"),
        ):
            request = FeedbackRequest(message="Update my config")
            response = await process_feedback(db_session, workspace.name, request)

            # Should return error message
            assert "error" in response.message.lower()
            assert len(response.changes_made) == 0
            # Config should be unchanged
            assert response.current_config.deriver_rules == "Existing rule"

    @pytest.mark.asyncio
    async def test_invalid_response_type(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test handling of unexpected response type from LLM."""
        workspace, _ = sample_data

        # Set initial config
        config = WorkspaceAgentConfig(deriver_rules="Existing rule")
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        # Mock returns something unexpected (string instead of FeedbackLLMResponse)
        mock_response = MagicMock()
        mock_response.content = "This is not a FeedbackLLMResponse"

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = FeedbackRequest(message="Update my config")
            response = await process_feedback(db_session, workspace.name, request)

            # Should return error message (caught by generic exception handler)
            assert "error" in response.message.lower()
            assert len(response.changes_made) == 0

    @pytest.mark.asyncio
    async def test_incremental_update(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that new rules are added to existing rules."""
        workspace, _ = sample_data

        # Set initial config
        config = WorkspaceAgentConfig(deriver_rules="Track emotions")
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        # Mock LLM to append new rule (structured output)
        mock_response = MagicMock()
        mock_response.content = FeedbackLLMResponse(
            message="Added goal tracking to existing rules.",
            understood_intent="Add goal tracking while preserving emotion tracking",
            changes=[
                FeedbackChange(
                    field="deriver_rules",
                    new_value="Track emotions\nAlso track goals and aspirations",
                )
            ],
        )

        with patch(
            "src.feedback.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            request = FeedbackRequest(message="Also track goals")
            response = await process_feedback(db_session, workspace.name, request)

            assert len(response.changes_made) == 1
            # Should contain both old and new rules
            assert "emotions" in response.changes_made[0].new_value.lower()
            assert "goals" in response.changes_made[0].new_value.lower()


# Note: API endpoint tests are in tests/routes/test_feedback.py
