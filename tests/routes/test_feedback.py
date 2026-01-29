"""API endpoint tests for the developer feedback channel."""

from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from src import models
from src.schemas import FeedbackResponse, WorkspaceAgentConfig


def test_feedback_endpoint_basic(
    client: TestClient,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Test basic feedback endpoint functionality."""
    workspace, _ = sample_data

    # Mock the process_feedback function
    mock_response = FeedbackResponse(
        message="Configuration updated",
        understood_intent="Test intent",
        changes_made=[],
        current_config=WorkspaceAgentConfig(),
    )

    with patch(
        "src.routers.workspaces.process_feedback",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        response = client.post(
            f"/v3/workspaces/{workspace.name}/feedback",
            json={"message": "Test message"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Configuration updated"
        assert "current_config" in data


def test_feedback_endpoint_with_introspection(
    client: TestClient,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Test feedback endpoint with introspection flag."""
    workspace, _ = sample_data

    mock_response = FeedbackResponse(
        message="Used introspection data",
        understood_intent="Test intent",
        changes_made=[],
        current_config=WorkspaceAgentConfig(),
    )

    with (
        patch(
            "src.routers.workspaces.process_feedback",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_process,
        patch(
            "src.routers.workspaces.get_latest_introspection_report",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_introspection,
    ):
        response = client.post(
            f"/v3/workspaces/{workspace.name}/feedback",
            json={"message": "Help me", "include_introspection": True},
        )
        assert response.status_code == 200

        # Verify introspection was fetched
        mock_introspection.assert_called_once()
        # Verify process_feedback was called
        mock_process.assert_called_once()


def test_feedback_endpoint_validation_error(
    client: TestClient,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Test feedback endpoint with invalid input."""
    workspace, _ = sample_data

    # Empty message should fail validation
    response = client.post(
        f"/v3/workspaces/{workspace.name}/feedback",
        json={"message": ""},
    )
    assert response.status_code == 422  # Validation error
