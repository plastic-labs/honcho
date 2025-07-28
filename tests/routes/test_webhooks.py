import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.models import Peer, WebhookStatus, Workspace
from src.webhooks.events import WebhookEventType


@pytest.mark.asyncio
async def test_create_endpoint(client: TestClient, sample_data: tuple[Workspace, Peer]):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://example.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"


@pytest.mark.asyncio
async def test_create_endpoint_invalid_url(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    client.headers["Authorization"] = "Bearer invalid-token"
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "192.168.1.1", "events": ["queue.empty"]},
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "Invalid URL format" in error["msg"]
    assert error["type"] == "value_error"


@pytest.mark.asyncio
async def test_create_endpoint_missing_workspace(client: TestClient):
    response = client.post(
        "/v2/workspaces/test-workspace/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    print(response.json())
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace test-workspace not found"}


@pytest.mark.asyncio
async def test_create_or_enable_webhook_success(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    create_endpoint_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com"},
    )
    assert create_endpoint_response.status_code == 200
    response_json = create_endpoint_response.json()
    assert response_json["url"] == "http://example.com"

    create_webhook_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert create_webhook_response.status_code == 200
    response_json = create_webhook_response.json()
    assert response_json["event"] == "queue.empty"


@pytest.mark.asyncio
async def test_create_or_enable_webhook_already_exists(
    client: TestClient, db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    create_endpoint_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    assert create_endpoint_response.status_code == 200
    response_json = create_endpoint_response.json()
    assert response_json["url"] == "http://example.com"

    from src.crud.webhook import update_webhook_status

    await update_webhook_status(
        db_session,
        test_workspace.name,
        schemas.WebhookUpdate(
            event=WebhookEventType.QUEUE_EMPTY, status=WebhookStatus.DISABLED
        ),
    )

    create_webhook_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert create_webhook_response.status_code == 200
    response_json = create_webhook_response.json()
    assert response_json["event"] == "queue.empty"
    assert response_json["status"] == "enabled"


@pytest.mark.asyncio
async def test_create_or_enable_webhook_no_workspace(
    client: TestClient,
):
    response = client.post(
        "/v2/workspaces/test-workspace/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace test-workspace not found"}


@pytest.mark.asyncio
async def test_create_or_enable_webhook_no_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    create_webhook_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert create_webhook_response.status_code == 404
    print(create_webhook_response.json())
    assert create_webhook_response.json() == {
        "detail": f"Webhook endpoint not set for workspace {test_workspace.name}"
    }


@pytest.mark.asyncio
async def test_get_webhook_configuration(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    assert response.status_code == 200

    get_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert get_response.status_code == 200
    response_json = get_response.json()
    assert response_json["url"] == "http://example.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"


@pytest.mark.asyncio
async def test_get_webhook_configuration_no_workspace(client: TestClient):
    response = client.get("/v2/workspaces/test-workspace/webhooks")
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace test-workspace not found"}


@pytest.mark.asyncio
async def test_update_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://endpoint1.com", "events": ["queue.empty"]},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://endpoint1.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://endpoint2.com"},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://endpoint2.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"


@pytest.mark.asyncio
async def test_update_webhook_endpoint_no_workspace(client: TestClient):
    response = client.put(
        "/v2/workspaces/test-workspace/webhooks",
        json={"url": "http://endpoint2.com"},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace test-workspace not found"}


@pytest.mark.asyncio
async def test_update_webhook_endpoint_no_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://endpoint2.com"},
    )
    assert response.status_code == 200
    assert response.json() == {"url": "http://endpoint2.com", "events": []}


@pytest.mark.asyncio
async def test_update_webhook_status(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://example.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty", "status": "disabled"},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["event"] == "queue.empty"
    assert response_json["status"] == "disabled"

    response = client.put(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty", "status": "enabled"},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["event"] == "queue.empty"
    assert response_json["status"] == "enabled"


@pytest.mark.asyncio
async def test_add_webhook_event_success(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    # Create endpoint with no events
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": []},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://example.com"
    assert response_json["events"] == []

    # Add webhook event
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert response.status_code == 200
    event_json = response.json()
    assert event_json["event"] == "queue.empty"
    assert event_json["status"] == "enabled"


@pytest.mark.asyncio
async def test_add_webhook_event_no_workspace(client: TestClient):
    response = client.post(
        "/v2/workspaces/test-workspace/webhooks/events",
        json={"event": "queue.empty"},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace test-workspace not found"}


@pytest.mark.asyncio
async def test_delete_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    # Create webhook endpoint
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": "http://example.com", "events": ["queue.empty"]},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] == "http://example.com"
    assert len(response_json["events"]) == 1
    assert response_json["events"][0]["event"] == "queue.empty"

    # Delete webhook endpoint
    response = client.delete(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert response.status_code == 200

    response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["url"] is None
    assert len(response_json["events"]) == 0
