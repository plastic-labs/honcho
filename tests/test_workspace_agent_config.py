"""Tests for workspace agent config functionality (Phase 2 of Agentic FDE)."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.deriver.prompts import minimal_deriver_prompt
from src.dialectic.prompts import agent_system_prompt
from src.schemas import WorkspaceAgentConfig


class TestWorkspaceAgentConfigSchema:
    """Test the WorkspaceAgentConfig Pydantic schema."""

    def test_default_values(self):
        """Test that default values are empty strings."""
        config = WorkspaceAgentConfig()
        assert config.deriver_rules == ""
        assert config.dialectic_rules == ""

    def test_custom_values(self):
        """Test setting custom values."""
        config = WorkspaceAgentConfig(
            deriver_rules="Focus on technical facts only",
            dialectic_rules="Be concise in responses",
        )
        assert config.deriver_rules == "Focus on technical facts only"
        assert config.dialectic_rules == "Be concise in responses"

    def test_model_dump(self):
        """Test serialization to dict."""
        config = WorkspaceAgentConfig(
            deriver_rules="Rule 1",
            dialectic_rules="Rule 2",
        )
        data = config.model_dump()
        assert data == {
            "deriver_rules": "Rule 1",
            "dialectic_rules": "Rule 2",
        }

    def test_model_validate(self):
        """Test deserialization from dict."""
        data = {
            "deriver_rules": "Custom rule",
            "dialectic_rules": "Another rule",
        }
        config = WorkspaceAgentConfig.model_validate(data)
        assert config.deriver_rules == "Custom rule"
        assert config.dialectic_rules == "Another rule"

    def test_model_validate_empty_dict(self):
        """Test deserialization from empty dict uses defaults."""
        config = WorkspaceAgentConfig.model_validate({})
        assert config.deriver_rules == ""
        assert config.dialectic_rules == ""


class TestDeriverPromptCustomRules:
    """Test custom rules injection in deriver prompt."""

    def test_empty_custom_rules_unchanged(self):
        """Test that empty custom_rules produces unchanged output."""
        prompt_without = minimal_deriver_prompt(
            peer_id="user123",
            messages="Hello world",
        )
        prompt_with_empty = minimal_deriver_prompt(
            peer_id="user123",
            messages="Hello world",
            custom_rules="",
        )
        assert prompt_without == prompt_with_empty

    def test_custom_rules_injected(self):
        """Test that custom_rules are injected into prompt."""
        custom_rules = "Focus on technical facts\nIgnore casual greetings"
        prompt = minimal_deriver_prompt(
            peer_id="user123",
            messages="Hello world",
            custom_rules=custom_rules,
        )
        assert "ADDITIONAL RULES (workspace-specific):" in prompt
        assert "Focus on technical facts" in prompt
        assert "Ignore casual greetings" in prompt

    def test_custom_rules_position(self):
        """Test that custom rules appear after RULES section."""
        custom_rules = "My custom rule"
        prompt = minimal_deriver_prompt(
            peer_id="user123",
            messages="Hello world",
            custom_rules=custom_rules,
        )
        # Find positions
        rules_pos = prompt.find("RULES:")
        custom_pos = prompt.find("ADDITIONAL RULES (workspace-specific):")
        examples_pos = prompt.find("EXAMPLES:")

        # Custom rules should be between RULES and EXAMPLES
        assert rules_pos < custom_pos < examples_pos


class TestDialecticPromptCustomRules:
    """Test custom rules injection in dialectic prompt."""

    def test_empty_custom_rules_unchanged(self):
        """Test that empty custom_rules produces no additional section."""
        prompt_without = agent_system_prompt(
            observer="agent",
            observed="user123",
            observer_peer_card=None,
            observed_peer_card=None,
        )
        prompt_with_empty = agent_system_prompt(
            observer="agent",
            observed="user123",
            observer_peer_card=None,
            observed_peer_card=None,
            custom_rules="",
        )
        # Both should not have the ADDITIONAL GUIDELINES section
        assert "ADDITIONAL GUIDELINES (workspace-specific)" not in prompt_without
        assert "ADDITIONAL GUIDELINES (workspace-specific)" not in prompt_with_empty

    def test_custom_rules_injected(self):
        """Test that custom_rules are injected into prompt."""
        custom_rules = "Always prioritize recent information\nBe skeptical of old data"
        prompt = agent_system_prompt(
            observer="agent",
            observed="user123",
            observer_peer_card=None,
            observed_peer_card=None,
            custom_rules=custom_rules,
        )
        assert "ADDITIONAL GUIDELINES (workspace-specific)" in prompt
        assert "Always prioritize recent information" in prompt
        assert "Be skeptical of old data" in prompt

    def test_custom_rules_with_peer_cards(self):
        """Test that custom_rules work with peer cards enabled."""
        custom_rules = "Custom rule here"
        prompt = agent_system_prompt(
            observer="agent",
            observed="user123",
            observer_peer_card=["Agent is helpful"],
            observed_peer_card=["User likes coffee"],
            custom_rules=custom_rules,
        )
        # Both peer cards and custom rules should be present
        assert "Agent is helpful" in prompt
        assert "User likes coffee" in prompt
        assert "ADDITIONAL GUIDELINES (workspace-specific)" in prompt
        assert "Custom rule here" in prompt


class TestWorkspaceAgentConfigCRUD:
    """Test CRUD operations for workspace agent config."""

    @pytest.mark.asyncio
    async def test_get_workspace_agent_config_default(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting config for workspace without _agent_config returns defaults."""
        workspace, _ = sample_data

        config = await crud.get_workspace_agent_config(db_session, workspace.name)

        assert config.deriver_rules == ""
        assert config.dialectic_rules == ""

    @pytest.mark.asyncio
    async def test_set_and_get_workspace_agent_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test setting and retrieving workspace agent config."""
        workspace, _ = sample_data

        # Set config
        config = WorkspaceAgentConfig(
            deriver_rules="Extract technical facts only",
            dialectic_rules="Be concise",
        )
        await crud.set_workspace_agent_config(db_session, workspace.name, config)

        # Get config
        retrieved = await crud.get_workspace_agent_config(db_session, workspace.name)

        assert retrieved.deriver_rules == "Extract technical facts only"
        assert retrieved.dialectic_rules == "Be concise"

    @pytest.mark.asyncio
    async def test_set_workspace_agent_config_preserves_other_metadata(
        self,
        db_session: AsyncSession,
    ):
        """Test that setting agent config preserves existing metadata."""
        # Create workspace with existing metadata
        workspace_name = str(generate_nanoid())
        workspace = models.Workspace(
            name=workspace_name,
            h_metadata={"custom_key": "custom_value", "another": 123},
        )
        db_session.add(workspace)
        await db_session.flush()

        # Set agent config
        config = WorkspaceAgentConfig(deriver_rules="Test rule")
        await crud.set_workspace_agent_config(db_session, workspace_name, config)

        # Verify both old metadata and new config are present
        await db_session.refresh(workspace)
        assert workspace.h_metadata.get("custom_key") == "custom_value"
        assert workspace.h_metadata.get("another") == 123
        assert "_agent_config" in workspace.h_metadata
        assert workspace.h_metadata["_agent_config"]["deriver_rules"] == "Test rule"

    @pytest.mark.asyncio
    async def test_update_workspace_agent_config(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test updating workspace agent config."""
        workspace, _ = sample_data

        # Set initial config
        config1 = WorkspaceAgentConfig(deriver_rules="Initial rule")
        await crud.set_workspace_agent_config(db_session, workspace.name, config1)

        # Update config
        config2 = WorkspaceAgentConfig(
            deriver_rules="Updated rule",
            dialectic_rules="New dialectic rule",
        )
        await crud.set_workspace_agent_config(db_session, workspace.name, config2)

        # Verify update
        retrieved = await crud.get_workspace_agent_config(db_session, workspace.name)
        assert retrieved.deriver_rules == "Updated rule"
        assert retrieved.dialectic_rules == "New dialectic rule"
