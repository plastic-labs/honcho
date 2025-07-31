from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.models import Peer, Workspace


@pytest.mark.asyncio
async def test_create_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data
    response = client.post(
        "/v2/webhooks",
        json={
            "url": "http://example.com/webhook",
            "workspace_id": test_workspace.name,
        },
    )
    assert response.status_code == 200
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
        "/v2/webhooks",
        json={
            "url": "192.168.1.1/webhook",
            "workspace_id": test_workspace.name,
        },
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert "Invalid URL format" in error["msg"]
    assert error["type"] == "value_error"


@pytest.mark.asyncio
async def test_create_webhook_endpoint_missing_workspace(client: TestClient):
    response = client.post(
        "/v2/webhooks",
        json={
            "url": "http://example.com/webhook",
            "workspace_id": "nonexistent-workspace",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace nonexistent-workspace not found"}


@pytest.mark.asyncio
async def test_list_webhook_endpoints_with_data(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    list_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    initial_count = len(list_response.json()["items"])

    # Create first endpoint
    response1 = client.post(
        "/v2/webhooks",
        json={
            "url": "http://example1.com/webhook",
            "workspace_id": test_workspace.name,
        },
    )
    assert response1.status_code == 200

    # Create second endpoint
    response2 = client.post(
        "/v2/webhooks",
        json={
            "url": "http://example2.com/webhook",
            "workspace_id": test_workspace.name,
        },
    )
    assert response2.status_code == 200

    # List endpoints
    list_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 2

    # Verify both endpoints are returned
    endpoint_urls = [ep["url"] for ep in endpoints]
    assert "http://example1.com/webhook" in endpoint_urls
    assert "http://example2.com/webhook" in endpoint_urls


@pytest.mark.asyncio
async def test_list_webhook_endpoints_missing_workspace(client: TestClient):
    response = client.get("/v2/webhooks?workspace_id=nonexistent-workspace")
    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace nonexistent-workspace not found"}


@pytest.mark.asyncio
async def test_delete_webhook_endpoint(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    test_workspace, _ = sample_data

    # Create webhook endpoint
    create_response = client.post(
        "/v2/webhooks",
        json={
            "url": "http://example.com/webhook",
            "workspace_id": test_workspace.name,
        },
    )
    assert create_response.status_code == 200
    endpoint = create_response.json()
    endpoint_id = endpoint["id"]

    # Delete webhook endpoint
    delete_response = client.delete(f"/v2/webhooks/{endpoint_id}")
    assert delete_response.status_code == 200

    # Verify endpoint is deleted
    list_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    assert list_response.status_code == 200
    response_data = list_response.json()
    assert response_data["items"] == []


@pytest.mark.asyncio
async def test_delete_webhook_endpoint_not_found(client: TestClient):
    response = client.delete("/v2/webhooks/nonexistent-id")
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

    initial_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    initial_count = len(initial_response.json()["items"])

    created_endpoints: list[Any] = []
    for url in urls:
        response = client.post(
            "/v2/webhooks",
            json={"url": url, "workspace_id": test_workspace.name},
        )
        assert response.status_code == 200
        created_endpoints.append(response.json())

    # List all endpoints
    list_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 3

    # Verify all URLs are present
    returned_urls = [ep["url"] for ep in endpoints]
    for url in urls:
        assert url in returned_urls

    # Delete one endpoint
    delete_response = client.delete(f"/v2/webhooks/{created_endpoints[0]['id']}")
    assert delete_response.status_code == 200

    # Verify only 2 endpoints remain
    list_response = client.get(f"/v2/webhooks?workspace_id={test_workspace.name}")
    assert list_response.status_code == 200
    response_data = list_response.json()
    endpoints = response_data["items"]
    assert len(endpoints) == initial_count + 2
