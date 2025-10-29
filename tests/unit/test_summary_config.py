"""Tests for configurable summary thresholds."""

import pytest
from pydantic import ValidationError

from src import models, schemas
from src.utils.config_helpers import get_summary_config


class TestSummaryConfigSchemas:
    """Test schema validation for summary configuration."""

    def test_workspace_create_with_valid_summary_config(self):
        """Test creating workspace with valid summary configuration."""
        workspace = schemas.WorkspaceCreate(
            name="test-workspace",
            messages_per_short_summary=10,
            messages_per_long_summary=30,
        )
        assert workspace.configuration["messages_per_short_summary"] == 10
        assert workspace.configuration["messages_per_long_summary"] == 30

    def test_workspace_create_short_equals_long_valid(self):
        """Test that short summary threshold can equal long summary threshold."""
        workspace = schemas.WorkspaceCreate(
            name="test-workspace",
            messages_per_short_summary=20,
            messages_per_long_summary=20,
        )
        assert workspace.configuration["messages_per_short_summary"] == 20
        assert workspace.configuration["messages_per_long_summary"] == 20

    def test_workspace_create_short_greater_than_long_invalid(self):
        """Test that short > long raises validation error."""
        # When long is below minimum, field validation catches it first
        with pytest.raises(ValidationError):
            schemas.WorkspaceCreate(
                name="test-workspace",
                messages_per_short_summary=30,
                messages_per_long_summary=10,
            )

        # Test with valid individual values but short > long
        with pytest.raises(ValidationError) as exc_info:
            schemas.WorkspaceCreate(
                name="test-workspace",
                messages_per_short_summary=40,
                messages_per_long_summary=30,
            )
        assert "messages_per_short_summary must be less than" in str(exc_info.value)

    def test_workspace_create_below_minimum_threshold(self):
        """Test that values below minimum threshold raise validation error."""
        with pytest.raises(ValidationError):
            schemas.WorkspaceCreate(
                name="test-workspace",
                messages_per_short_summary=5,  # Below minimum of 10
            )

    def test_workspace_update_with_valid_summary_config(self):
        """Test updating workspace with valid summary configuration."""
        workspace_update = schemas.WorkspaceUpdate(
            messages_per_short_summary=15,
            messages_per_long_summary=45,
        )
        assert workspace_update.configuration is not None
        assert workspace_update.configuration["messages_per_short_summary"] == 15
        assert workspace_update.configuration["messages_per_long_summary"] == 45

    def test_workspace_update_partial_config(self):
        """Test updating only one threshold."""
        workspace_update = schemas.WorkspaceUpdate(
            messages_per_short_summary=20,
        )
        assert workspace_update.configuration is not None
        assert workspace_update.configuration["messages_per_short_summary"] == 20
        assert "messages_per_long_summary" not in workspace_update.configuration

    def test_session_create_with_valid_summary_config(self):
        """Test creating session with valid summary configuration."""
        session = schemas.SessionCreate(
            name="test-session",
            messages_per_short_summary=10,
            messages_per_long_summary=30,
        )
        assert session.configuration is not None
        assert session.configuration["messages_per_short_summary"] == 10
        assert session.configuration["messages_per_long_summary"] == 30

    def test_session_update_with_valid_summary_config(self):
        """Test updating session with valid summary configuration."""
        session_update = schemas.SessionUpdate(
            messages_per_short_summary=12,
            messages_per_long_summary=36,
        )
        assert session_update.configuration is not None
        assert session_update.configuration["messages_per_short_summary"] == 12
        assert session_update.configuration["messages_per_long_summary"] == 36

    def test_session_create_short_greater_than_long_invalid(self):
        """Test that short > long raises validation error for sessions."""
        with pytest.raises(ValidationError) as exc_info:
            schemas.SessionCreate(
                name="test-session",
                messages_per_short_summary=40,
                messages_per_long_summary=20,
            )
        assert "messages_per_short_summary must be less than" in str(exc_info.value)


class TestConfigResolutionHierarchy:
    """Test configuration resolution with hierarchical fallback."""

    def test_session_config_overrides_workspace_config(self):
        """Test that session configuration takes precedence over workspace."""
        # Create workspace with config
        workspace = models.Workspace(
            name="test-workspace",
            configuration={
                "messages_per_short_summary": 10,
                "messages_per_long_summary": 30,
            },
        )

        # Create session with different config
        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={
                "messages_per_short_summary": 15,
                "messages_per_long_summary": 45,
            },
        )

        short, long = get_summary_config(session, workspace)
        assert short == 15  # From session
        assert long == 45  # From session

    def test_workspace_config_used_when_session_has_none(self):
        """Test that workspace configuration is used when session has no config."""
        workspace = models.Workspace(
            name="test-workspace",
            configuration={
                "messages_per_short_summary": 12,
                "messages_per_long_summary": 36,
            },
        )

        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={},  # No summary config
        )

        short, long = get_summary_config(session, workspace)
        assert short == 12  # From workspace
        assert long == 36  # From workspace

    def test_global_defaults_used_when_no_config(self):
        """Test that global defaults are used when neither session nor workspace have config."""
        from src.config import settings

        workspace = models.Workspace(
            name="test-workspace",
            configuration={},
        )

        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={},
        )

        short, long = get_summary_config(session, workspace)
        assert short == settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
        assert long == settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY

    def test_session_partial_override(self):
        """Test that session can override only one threshold, falling back for the other."""
        workspace = models.Workspace(
            name="test-workspace",
            configuration={
                "messages_per_short_summary": 10,
                "messages_per_long_summary": 30,
            },
        )

        # Session only specifies short summary
        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={
                "messages_per_short_summary": 15,
                # No messages_per_long_summary
            },
        )

        short, long = get_summary_config(session, workspace)
        assert short == 15  # From session
        assert long == 30  # From workspace

    def test_workspace_partial_override(self):
        """Test that workspace can specify only one threshold, falling back to global for the other."""
        from src.config import settings

        workspace = models.Workspace(
            name="test-workspace",
            configuration={
                "messages_per_long_summary": 50,
                # No messages_per_short_summary
            },
        )

        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={},
        )

        short, long = get_summary_config(session, workspace)
        assert short == settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY  # From global
        assert long == 50  # From workspace

    def test_no_workspace_provided(self):
        """Test resolution when workspace is not provided."""
        from src.config import settings

        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={
                "messages_per_short_summary": 25,
            },
        )

        short, long = get_summary_config(session, workspace=None)
        assert short == 25  # From session
        assert long == settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY  # From global

    def test_validation_fails_for_invalid_resolved_values(self):
        """Test that validation catches invalid values in stored configuration."""
        workspace = models.Workspace(
            name="test-workspace",
            configuration={},
        )

        # Simulate corrupted data (should not happen with schema validation)
        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={
                "messages_per_short_summary": 0,  # Invalid
            },
        )

        with pytest.raises(ValueError, match="Invalid messages_per_short_summary"):
            get_summary_config(session, workspace)

    def test_validation_catches_short_greater_than_long(self):
        """Test that resolution validates short <= long after resolution."""
        # Workspace has valid config
        workspace = models.Workspace(
            name="test-workspace",
            configuration={
                "messages_per_short_summary": 10,
                "messages_per_long_summary": 30,
            },
        )

        # Session overrides to create invalid combination
        # Session short = 40, but workspace long = 30
        session = models.Session(
            name="test-session",
            workspace_name="test-workspace",
            configuration={
                "messages_per_short_summary": 40,  # Greater than workspace long
                # Uses workspace's long of 30
            },
        )

        # Should raise validation error after resolution
        with pytest.raises(ValueError, match="messages_per_short_summary.*must be <"):
            get_summary_config(session, workspace)
