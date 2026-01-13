from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.models import Peer, Workspace


@pytest.mark.asyncio
async def test_create_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "http://example.com/webhook",
        },
    )
    assert response.status_code in [200, 201]
    response_json = response.json()
    assert response_json["url"] == "http://example.com/webhook"
    assert "id" in response_json
    assert response_json["workspace_id"] == test_workspace.name
    assert "created_at" in response_json


@pytest.mark.asyncio
async def test_create_webhook_endpoint_invalid_url(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "192.168.1.1/webhook",
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "Invalid URL format" in error["msg"]
    assert error["type"] == "value_error"


@pytest.mark.asyncio
async def test_create_webhook_endpoint_missing_workspace(client: TestClient):
    response = client.post(
        "/v2/workspaces/nonexistent-workspace/webhooks",
        json={
            "url": "http://example.com/webhook",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace nonexistent-workspace not found"}


@pytest.mark.asyncio
async def test_list_webhook_endpoints_with_data(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    initial_count = len(list_response.json()["items"])

    # Create first endpoint
    response1 = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "http://example1.com/webhook",
        },
    )
    assert response1.status_code in [200, 201]

    # Create second endpoint
    response2 = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "http://example2.com/webhook",
        },
    )
    assert response2.status_code in [200, 201]

    # List endpoints
    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 2

    # Verify both endpoints are returned
    endpoint_urls = [ep["url"] for ep in endpoints]
    assert "http://example1.com/webhook" in endpoint_urls
    assert "http://example2.com/webhook" in endpoint_urls


@pytest.mark.asyncio
async def test_delete_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    # Create webhook endpoint
    create_response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "http://example.com/webhook",
        },
    )
    assert create_response.status_code in [200, 201]
    endpoint = create_response.json()
    endpoint_id = endpoint["id"]

    # Delete webhook endpoint
    delete_response = client.delete(
        f"/v2/workspaces/{test_workspace.name}/webhooks/{endpoint_id}"
    )
    assert delete_response.status_code == 204

    # Verify endpoint is deleted
    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert list_response.status_code == 200
    response_data = list_response.json()
    assert response_data["items"] == []


@pytest.mark.asyncio
async def test_delete_webhook_endpoint_not_found(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.delete(
        f"/v2/workspaces/{test_workspace.name}/webhooks/nonexistent-id"
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_multiple_endpoints_per_workspace(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Test that workspaces can have multiple webhook endpoints"""
    test_workspace, _ = sample_data

    # Create multiple endpoints
    urls = [
        "http://app1.com/webhook",
        "http://app2.com/webhook",
        "http://app3.com/webhook",
    ]

    initial_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    initial_count = len(initial_response.json()["items"])

    created_endpoints: list[Any] = []
    for url in urls:
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/webhooks",
            json={"url": url},
        )
        assert response.status_code in [200, 201]
        created_endpoints.append(response.json())

    # List all endpoints
    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 3

    # Verify all URLs are present
    returned_urls = [ep["url"] for ep in endpoints]
    for url in urls:
        assert url in returned_urls

    # Delete one endpoint
    delete_response = client.delete(
        f"/v2/workspaces/{test_workspace.name}/webhooks/{created_endpoints[0]['id']}"
    )
    assert delete_response.status_code == 204

    # Verify only 2 endpoints remain
    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 2


@pytest.mark.asyncio
async def test_create_duplicate_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    url = "http://example.com/duplicate"

    # Create the endpoint first
    response1 = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": url},
    )
    assert response1.status_code in [200, 201]

    # Try to create it again
    response2 = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={"url": url},
    )
    assert response2.status_code in [200, 201]
    assert response1.json() == response2.json()

    # Verify only one endpoint exists
    list_response = client.get(f"/v2/workspaces/{test_workspace.name}/webhooks")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == 1
    assert endpoints[0]["url"] == url


@pytest.mark.asyncio
async def test_max_webhook_endpoints_per_workspace(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    limit: int = settings.WEBHOOK.MAX_WORKSPACE_LIMIT

    # Create endpoints up to the limit
    for i in range(limit):
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/webhooks",
            json={
                "url": f"http://example{i}.com/webhook",
            },
        )
        assert response.status_code in [200, 201]

    # Try to create one more
    response = client.post(
        f"/v2/workspaces/{test_workspace.name}/webhooks",
        json={
            "url": "http://extra.com/webhook",
        },
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_same_endpoint_in_different_workspaces(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    ws1, _ = sample_data
    ws2_response = client.post("/v2/workspaces", json={"name": "workspace-2"})
    assert ws2_response.status_code in [200, 201]
    ws2 = ws2_response.json()

    url = "http://example.com/shared"

    # Create endpoint in workspace 1
    response1 = client.post(
        f"/v2/workspaces/{ws1.name}/webhooks",
        json={"url": url},
    )
    assert response1.status_code in [200, 201]

    # Create same endpoint in workspace 2
    response2 = client.post(
        f"/v2/workspaces/{ws2['id']}/webhooks",
        json={"url": url},
    )
    assert response2.status_code in [200, 201]

    # Verify they are different resources
    assert response1.json()["id"] != response2.json()["id"]
    assert response1.json()["workspace_id"] == ws1.name
    assert response2.json()["workspace_id"] == ws2["id"]
