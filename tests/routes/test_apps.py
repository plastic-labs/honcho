from nanoid import generate as generate_nanoid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import models  # Import your SQLAlchemy models


def test_create_app(client):
    name = str(generate_nanoid())
    response = client.post("/apps", json={"name": name, "metadata": {"key": "value"}})
    print(response)
    assert response.status_code == 200
    data = response.json()
    print("===================")
    print(data)
    print("===================")
    assert data["name"] == name
    assert data["metadata"] == {"key": "value"}
    assert "id" in data


def test_get_or_create_app(client):
    name = str(generate_nanoid())
    response = client.get(f"/apps/name/{name}")
    assert response.status_code == 404
    response = client.get(f"/apps/get_or_create/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == name
    assert "id" in data


def test_get_or_create_existing_app(client):
    name = str(generate_nanoid())
    response = client.get(f"/apps/name/{name}")
    assert response.status_code == 404
    response = client.post("/apps", json={"name": name, "metadata": {"key": "value"}})
    assert response.status_code == 200
    app1 = response.json()
    response = client.get(f"/apps/get_or_create/{name}")
    assert response.status_code == 200
    app2 = response.json()
    assert app1["name"] == app2["name"]
    assert app1["id"] == app2["id"]
    assert app1["metadata"] == app2["metadata"]


def test_get_app_by_id(client, sample_data):
    test_app, _ = sample_data
    response = client.get(f"/apps/{test_app.public_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.public_id)


def test_get_app_by_name(client, sample_data):
    test_app, _ = sample_data
    response = client.get(f"/apps/name/{test_app.name}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_app.name
    assert data["id"] == str(test_app.public_id)


def test_update_app(client, sample_data):
    test_app, _ = sample_data
    new_name = str(generate_nanoid())
    response = client.put(
        f"/apps/{test_app.public_id}",
        json={"name": new_name, "metadata": {"new_key": "new_value"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == new_name
    assert data["metadata"] == {"new_key": "new_value"}
