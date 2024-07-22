import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models  # Import your SQLAlchemy models


def test_create_app(client):
    name = str(uuid.uuid4())
    response = client.post("/apps", json={"name": name, "metadata": {"key": "value"}})
    print(response)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert data["metadata"] == {"key": "value"}
    assert "id" in data


def test_get_or_create_app(client):
    name = str(uuid.uuid4())
    response = client.get(f"/apps/name/{name}")
    print(response)
    assert response.status_code == 404
    response = client.get(f"/apps/get_or_create/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert "id" in data


def test_get_app_by_id(client, test_data):
    test_app, _ = test_data
    response = client.get(f"/apps/{test_app.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.id)


def test_get_app_by_name(client, test_data):
    test_app, _ = test_data
    response = client.get(f"/apps/name/{test_app.name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.id)


def test_update_app(client, test_data):
    test_app, _ = test_data
    new_name = str(uuid.uuid4())
    response = client.put(
        f"/apps/{test_app.id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == new_name
    assert data["metadata"] == {"new_key": "new_value"}
