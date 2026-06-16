import os
import socket
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from src import models
from src.config import settings
from src.exceptions import HonchoException
from src.routers.keys import router as keys_router
from src.security import create_admin_jwt

ROOT = Path(__file__).resolve().parents[2]
API_RS_MANIFEST = ROOT / "api-rs" / "Cargo.toml"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def rust_api_url(db_engine: AsyncEngine) -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "RUST_API_BIND_ADDRESS": f"127.0.0.1:{port}",
            "DB_CONNECTION_URI": str(db_engine.url),
            "AUTH_USE_AUTH": "false",
            "DB_SCHEMA": "public",
        }
    )
    process = subprocess.Popen(
        ["cargo", "run", "--manifest-path", str(API_RS_MANIFEST), "--quiet"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                raise RuntimeError(f"Rust API exited early:\n{stderr}")
            try:
                response = httpx.get(f"{base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    break
            except Exception as exc:  # noqa: BLE001 - diagnostics for startup loop
                last_error = exc
            time.sleep(0.1)
        else:
            process.terminate()
            _, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Rust API did not become healthy: {last_error}\n{stderr}"
            )

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def rust_api_auth_url(db_engine: AsyncEngine) -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "RUST_API_BIND_ADDRESS": f"127.0.0.1:{port}",
            "DB_CONNECTION_URI": str(db_engine.url),
            "AUTH_USE_AUTH": "true",
            "AUTH_JWT_SECRET": "test-secret",
            "DB_SCHEMA": "public",
        }
    )
    process = subprocess.Popen(
        ["cargo", "run", "--manifest-path", str(API_RS_MANIFEST), "--quiet"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                raise RuntimeError(f"Rust API exited early:\n{stderr}")
            try:
                response = httpx.get(f"{base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    break
            except Exception as exc:  # noqa: BLE001 - diagnostics for startup loop
                last_error = exc
            time.sleep(0.1)
        else:
            process.terminate()
            _, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Rust API did not become healthy: {last_error}\n{stderr}"
            )

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def rust_api_writes_url(db_engine: AsyncEngine) -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "RUST_API_BIND_ADDRESS": f"127.0.0.1:{port}",
            "DB_CONNECTION_URI": str(db_engine.url),
            "AUTH_USE_AUTH": "false",
            "RUST_API_ENABLE_WRITES": "true",
            "DB_SCHEMA": "public",
        }
    )
    process = subprocess.Popen(
        ["cargo", "run", "--manifest-path", str(API_RS_MANIFEST), "--quiet"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                raise RuntimeError(f"Rust API exited early:\n{stderr}")
            try:
                response = httpx.get(f"{base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    break
            except Exception as exc:  # noqa: BLE001 - diagnostics for startup loop
                last_error = exc
            time.sleep(0.1)
        else:
            process.terminate()
            _, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Rust API did not become healthy: {last_error}\n{stderr}"
            )

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def rust_api_auth_writes_url(db_engine: AsyncEngine) -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "RUST_API_BIND_ADDRESS": f"127.0.0.1:{port}",
            "DB_CONNECTION_URI": str(db_engine.url),
            "AUTH_USE_AUTH": "true",
            "AUTH_JWT_SECRET": "test-secret",
            "RUST_API_ENABLE_WRITES": "true",
            "DB_SCHEMA": "public",
        }
    )
    process = subprocess.Popen(
        ["cargo", "run", "--manifest-path", str(API_RS_MANIFEST), "--quiet"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                _, stderr = process.communicate(timeout=1)
                raise RuntimeError(f"Rust API exited early:\n{stderr}")
            try:
                response = httpx.get(f"{base_url}/health", timeout=0.5)
                if response.status_code == 200:
                    break
            except Exception as exc:  # noqa: BLE001 - diagnostics for startup loop
                last_error = exc
            time.sleep(0.1)
        else:
            process.terminate()
            _, stderr = process.communicate(timeout=5)
            raise RuntimeError(
                f"Rust API did not become healthy: {last_error}\n{stderr}"
            )

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _compare(
    client: TestClient,
    rust_api_url: str,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> None:
    python_response = client.request(method, path, json=json_body)
    rust_response = httpx.request(
        method,
        f"{rust_api_url}{path}",
        json=json_body,
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code
    assert rust_response.json() == python_response.json()


def _workspace_contract_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": body["metadata"],
        "configuration": body["configuration"],
    }


def _peer_row_contract_fields(peer: models.Peer) -> dict[str, Any]:
    return {
        "metadata": peer.h_metadata,
        "configuration": peer.configuration,
        "internal_metadata": peer.internal_metadata,
    }


def _session_row_contract_fields(session: models.Session) -> dict[str, Any]:
    return {
        "is_active": session.is_active,
        "metadata": session.h_metadata,
        "configuration": session.configuration,
        "internal_metadata": session.internal_metadata,
    }


def _session_peer_row_contract_fields(
    session_peer: models.SessionPeer,
) -> dict[str, Any]:
    return {
        "configuration": session_peer.configuration,
        "is_active": session_peer.left_at is None,
    }


def _test_jwt(payload: dict[str, Any]) -> str:
    return jwt.encode({"t": "", **payload}, "test-secret", algorithm="HS256")


@pytest.mark.asyncio
async def test_workspace_get_or_create_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_name = f"rust-contract-python-{marker}"
    rust_name = f"rust-contract-rust-{marker}"
    metadata_payload = {
        "contract": marker,
        "kind": "workspace-write",
        "nul": "a\x00b",
        "nested": ["x\x00y"],
    }
    metadata = {
        "contract": marker,
        "kind": "workspace-write",
        "nul": "ab",
        "nested": ["xy"],
    }
    configuration = {"feature1": True, "feature2": False}

    python_response = client.post(
        "/v3/workspaces",
        json={
            "name": python_name,
            "metadata": metadata_payload,
            "configuration": configuration,
        },
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces",
        json={
            "id": rust_name,
            "metadata": metadata_payload,
            "configuration": configuration,
        },
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    python_body = python_response.json()
    rust_body = rust_response.json()
    assert python_body["id"] == python_name
    assert rust_body["id"] == rust_name
    assert _workspace_contract_fields(rust_body) == _workspace_contract_fields(
        python_body
    ) == {
        "metadata": metadata,
        "configuration": configuration,
    }
    assert "created_at" in rust_body

    python_existing = client.post(
        "/v3/workspaces",
        json={
            "name": python_name,
            "metadata": {"ignored": True},
            "configuration": {"ignored": True},
        },
    )
    rust_existing = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces",
        json={
            "name": rust_name,
            "metadata": {"ignored": True},
            "configuration": {"ignored": True},
        },
        timeout=5,
    )

    assert rust_existing.status_code == python_existing.status_code == 200
    python_existing_body = python_existing.json()
    rust_existing_body = rust_existing.json()
    assert python_existing_body["id"] == python_name
    assert rust_existing_body["id"] == rust_name
    assert _workspace_contract_fields(
        rust_existing_body
    ) == _workspace_contract_fields(python_existing_body) == {
        "metadata": metadata,
        "configuration": configuration,
    }

    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == rust_name)
    )
    assert row is not None
    assert row.h_metadata == metadata
    assert row.configuration == configuration


@pytest.mark.asyncio
async def test_workspace_configuration_coercion_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_name = f"rust-contract-python-config-{marker}"
    rust_name = f"rust-contract-rust-config-{marker}"
    request_configuration = {
        "reasoning": {
            "enabled": "false",
            "custom_instructions": "Prefer facts.",
            "ignored": True,
        },
        "peer_card": {"use": "true", "create": "off", "ignored": True},
        "summary": {
            "enabled": "yes",
            "messages_per_short_summary": "10",
            "messages_per_long_summary": "20",
            "ignored": True,
        },
        "dream": {"enabled": "no", "ignored": True},
        "extra": {"kept": True},
    }
    expected_configuration = {
        "reasoning": {"enabled": False, "custom_instructions": "Prefer facts."},
        "peer_card": {"use": True, "create": False},
        "summary": {
            "enabled": True,
            "messages_per_short_summary": 10,
            "messages_per_long_summary": 20,
        },
        "dream": {"enabled": False},
        "extra": {"kept": True},
    }

    python_response = client.post(
        "/v3/workspaces",
        json={"name": python_name, "configuration": request_configuration},
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces",
        json={"name": rust_name, "configuration": request_configuration},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    assert python_response.json()["configuration"] == expected_configuration
    assert rust_response.json()["configuration"] == expected_configuration

    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == rust_name)
    )
    assert row is not None
    assert row.configuration == expected_configuration


def test_workspace_write_validation_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    too_many_keys = {str(index): index for index in range(101)}
    payloads = [
        {},
        {"name": ""},
        {"id": ""},
        {"name": "invalid name"},
        {"id": "invalid name"},
        {"name": "a" * 513},
        {
            "name": "rust-contract-invalid-config",
            "configuration": {"summary": {"messages_per_short_summary": 1}},
        },
        {
            "name": "rust-contract-invalid-reasoning",
            "configuration": {"reasoning": {"enabled": "not-bool"}},
        },
        {
            "name": "rust-contract-invalid-peer-card",
            "configuration": {"peer_card": {"use": "not-bool"}},
        },
        {
            "name": "rust-contract-invalid-dream",
            "configuration": {"dream": {"enabled": "not-bool"}},
        },
        {
            "name": "rust-contract-invalid-depth",
            "metadata": {"a": {"b": {"c": {"d": {"e": {"f": "x"}}}}}},
        },
        {
            "name": "rust-contract-invalid-keys",
            "metadata": too_many_keys,
        },
        {"name": "rust-contract-invalid-metadata-null", "metadata": None},
        {"name": "rust-contract-invalid-metadata-string", "metadata": "bad"},
        {"name": "rust-contract-invalid-metadata-array", "metadata": []},
    ]

    for payload in payloads:
        python_response = client.post("/v3/workspaces", json=payload)
        rust_response = httpx.post(
            f"{rust_api_writes_url}/v3/workspaces",
            json=payload,
            timeout=5,
        )

        assert rust_response.status_code == python_response.status_code == 422
        assert rust_response.json() == python_response.json()


@pytest.mark.asyncio
async def test_workspace_update_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_name = f"rust-contract-python-update-{marker}"
    rust_name = f"rust-contract-rust-update-{marker}"
    initial_metadata = {"initial": marker}
    initial_configuration = {
        "existing": True,
        "summary": {"enabled": False},
    }
    update_metadata_payload = {"updated": marker, "nul": "a\x00b"}
    update_metadata = {"updated": marker, "nul": "ab"}
    update_configuration_payload = {
        "summary": {"enabled": "yes", "ignored": True},
        "new_feature": False,
    }
    expected_configuration = {
        "existing": True,
        "summary": {"enabled": True},
        "new_feature": False,
    }

    python_create = client.post(
        "/v3/workspaces",
        json={
            "name": python_name,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
    )
    rust_create = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces",
        json={
            "name": rust_name,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
        timeout=5,
    )
    assert rust_create.status_code == python_create.status_code == 201

    python_update = client.put(
        f"/v3/workspaces/{python_name}",
        json={
            "metadata": update_metadata_payload,
            "configuration": update_configuration_payload,
        },
    )
    rust_update = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={
            "metadata": update_metadata_payload,
            "configuration": update_configuration_payload,
        },
        timeout=5,
    )

    assert rust_update.status_code == python_update.status_code == 200
    assert python_update.json()["metadata"] == update_metadata
    assert rust_update.json()["metadata"] == update_metadata
    assert python_update.json()["configuration"] == expected_configuration
    assert rust_update.json()["configuration"] == expected_configuration

    python_empty = client.put(f"/v3/workspaces/{python_name}", json={})
    rust_empty = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={},
        timeout=5,
    )
    assert rust_empty.status_code == python_empty.status_code == 200
    assert _workspace_contract_fields(rust_empty.json()) == _workspace_contract_fields(
        python_empty.json()
    )

    python_null = client.put(
        f"/v3/workspaces/{python_name}",
        json={"metadata": None, "configuration": None},
    )
    rust_null = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={"metadata": None, "configuration": None},
        timeout=5,
    )
    assert rust_null.status_code == python_null.status_code == 200
    assert _workspace_contract_fields(rust_null.json()) == _workspace_contract_fields(
        python_null.json()
    )

    python_extra = client.put(
        f"/v3/workspaces/{python_name}",
        json={"id": "ignored", "name": "also-ignored", "configuration": {"reasoning": None}},
    )
    rust_extra = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={"id": "ignored", "name": "also-ignored", "configuration": {"reasoning": None}},
        timeout=5,
    )
    assert rust_extra.status_code == python_extra.status_code == 200
    assert python_extra.json()["id"] == python_name
    assert rust_extra.json()["id"] == rust_name
    assert _workspace_contract_fields(rust_extra.json()) == _workspace_contract_fields(
        python_extra.json()
    )

    python_nested_null = client.put(
        f"/v3/workspaces/{python_name}",
        json={"configuration": {"summary": {"enabled": None}}},
    )
    rust_nested_null = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={"configuration": {"summary": {"enabled": None}}},
        timeout=5,
    )
    expected_nested_null_configuration = {
        "existing": True,
        "summary": {},
        "new_feature": False,
    }
    assert rust_nested_null.status_code == python_nested_null.status_code == 200
    assert python_nested_null.json()["configuration"] == expected_nested_null_configuration
    assert rust_nested_null.json()["configuration"] == expected_nested_null_configuration

    python_clear = client.put(
        f"/v3/workspaces/{python_name}",
        json={"metadata": {}},
    )
    rust_clear = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={"metadata": {}},
        timeout=5,
    )
    assert rust_clear.status_code == python_clear.status_code == 200
    assert python_clear.json()["metadata"] == {}
    assert rust_clear.json()["metadata"] == {}

    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == rust_name)
    )
    assert row is not None
    assert row.h_metadata == {}
    assert row.configuration == expected_nested_null_configuration


@pytest.mark.asyncio
async def test_workspace_update_creates_missing_workspace_like_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_name = f"rust-contract-python-put-create-{marker}"
    rust_name = f"rust-contract-rust-put-create-{marker}"
    metadata = {"created_by": "put", "marker": marker}
    configuration = {"summary": {"enabled": "true"}}
    expected_configuration = {"summary": {"enabled": True}}

    python_response = client.put(
        f"/v3/workspaces/{python_name}",
        json={"metadata": metadata, "configuration": configuration},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_name}",
        json={"metadata": metadata, "configuration": configuration},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_name
    assert rust_response.json()["id"] == rust_name
    assert python_response.json()["metadata"] == metadata
    assert rust_response.json()["metadata"] == metadata
    assert python_response.json()["configuration"] == expected_configuration
    assert rust_response.json()["configuration"] == expected_configuration

    row = await db_session.scalar(
        select(models.Workspace).where(models.Workspace.name == rust_name)
    )
    assert row is not None
    assert row.h_metadata == metadata
    assert row.configuration == expected_configuration


def test_workspace_update_validation_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    workspace = f"rust-contract-update-validation-{generate_nanoid()}"
    payloads: list[Any] = [
        [],
        {"metadata": "bad"},
        {"metadata": []},
        {"configuration": "bad"},
        {"configuration": {"summary": {"messages_per_short_summary": 1}}},
        {"configuration": {"reasoning": {"enabled": "not-bool"}}},
    ]

    for payload in payloads:
        python_response = client.put(f"/v3/workspaces/{workspace}", json=payload)
        rust_response = httpx.put(
            f"{rust_api_writes_url}/v3/workspaces/{workspace}",
            json=payload,
            timeout=5,
        )

        assert rust_response.status_code == python_response.status_code == 422
        assert rust_response.json() == python_response.json()


@pytest.mark.asyncio
async def test_peer_get_or_create_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-ws-{marker}"
    python_peer = f"rust-contract-python-peer-{marker}"
    rust_peer = f"rust-contract-rust-peer-{marker}"
    metadata_payload = {"kind": "peer-write", "nul": "a\x00b"}
    metadata = {"kind": "peer-write", "nul": "ab"}
    configuration = {"feature1": True, "nested": {"kept": True}}

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={
            "name": python_peer,
            "metadata": metadata_payload,
            "configuration": configuration,
        },
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={
            "id": rust_peer,
            "metadata": metadata_payload,
            "configuration": configuration,
        },
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    assert python_response.json()["id"] == python_peer
    assert rust_response.json()["id"] == rust_peer
    assert python_response.json()["workspace_id"] == python_workspace
    assert rust_response.json()["workspace_id"] == rust_workspace
    assert python_response.json()["metadata"] == metadata
    assert rust_response.json()["metadata"] == metadata
    assert python_response.json()["configuration"] == configuration
    assert rust_response.json()["configuration"] == configuration

    replacement_metadata = {"replacement": marker}
    replacement_configuration = {"feature2": False}
    python_existing = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={
            "name": python_peer,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration,
        },
    )
    rust_existing = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={
            "name": rust_peer,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration,
        },
        timeout=5,
    )

    assert rust_existing.status_code == python_existing.status_code == 200
    assert python_existing.json()["metadata"] == replacement_metadata
    assert rust_existing.json()["metadata"] == replacement_metadata
    assert python_existing.json()["configuration"] == replacement_configuration
    assert rust_existing.json()["configuration"] == replacement_configuration

    python_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == python_workspace,
            models.Peer.name == python_peer,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == rust_workspace,
            models.Peer.name == rust_peer,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _peer_row_contract_fields(rust_row) == _peer_row_contract_fields(
        python_row
    ) == {
        "metadata": replacement_metadata,
        "configuration": replacement_configuration,
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_peer_get_or_create_defaults_and_noop_match_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-default-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-default-ws-{marker}"
    python_peer = f"rust-contract-python-peer-default-{marker}"
    rust_peer = f"rust-contract-rust-peer-default-{marker}"

    python_default = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={"name": python_peer},
    )
    rust_default = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={"name": rust_peer},
        timeout=5,
    )
    assert rust_default.status_code == python_default.status_code == 201
    assert _workspace_contract_fields(rust_default.json()) == _workspace_contract_fields(
        python_default.json()
    ) == {"metadata": {}, "configuration": {}}

    replacement_metadata = {"kept": marker}
    replacement_configuration = {"feature": True}
    assert client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={
            "name": python_peer,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration,
        },
    ).status_code == 200
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={
            "name": rust_peer,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration,
        },
        timeout=5,
    ).status_code == 200

    for payload in (
        {"name": python_peer},
        {"name": python_peer, "metadata": None, "configuration": None},
    ):
        python_response = client.post(
            f"/v3/workspaces/{python_workspace}/peers",
            json=payload,
        )
        rust_payload = {**payload, "name": rust_peer}
        rust_response = httpx.post(
            f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
            json=rust_payload,
            timeout=5,
        )

        assert rust_response.status_code == python_response.status_code == 200
        assert _workspace_contract_fields(
            rust_response.json()
        ) == _workspace_contract_fields(python_response.json()) == {
            "metadata": replacement_metadata,
            "configuration": replacement_configuration,
        }

    python_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == python_workspace,
            models.Peer.name == python_peer,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == rust_workspace,
            models.Peer.name == rust_peer,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _peer_row_contract_fields(rust_row) == _peer_row_contract_fields(
        python_row
    ) == {
        "metadata": replacement_metadata,
        "configuration": replacement_configuration,
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_peer_update_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-update-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-update-ws-{marker}"
    python_peer = f"rust-contract-python-peer-update-{marker}"
    rust_peer = f"rust-contract-rust-peer-update-{marker}"
    initial_metadata = {"initial": marker}
    initial_configuration = {"initial_feature": True}
    update_metadata_payload = {"updated": marker, "nul": "x\x00y"}
    update_metadata = {"updated": marker, "nul": "xy"}
    update_configuration = {"updated_feature": False}

    assert client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={
            "name": python_peer,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={
            "name": rust_peer,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
        timeout=5,
    ).status_code == 201

    python_update = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={
            "id": "ignored",
            "name": "also-ignored",
            "metadata": update_metadata_payload,
            "configuration": update_configuration,
        },
    )
    rust_update = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={
            "id": "ignored",
            "name": "also-ignored",
            "metadata": update_metadata_payload,
            "configuration": update_configuration,
        },
        timeout=5,
    )

    assert rust_update.status_code == python_update.status_code == 200
    assert python_update.json()["id"] == python_peer
    assert rust_update.json()["id"] == rust_peer
    assert python_update.json()["metadata"] == update_metadata
    assert rust_update.json()["metadata"] == update_metadata
    assert python_update.json()["configuration"] == update_configuration
    assert rust_update.json()["configuration"] == update_configuration

    python_null = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={"metadata": None, "configuration": None},
    )
    rust_null = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={"metadata": None, "configuration": None},
        timeout=5,
    )
    assert rust_null.status_code == python_null.status_code == 200
    assert _workspace_contract_fields(rust_null.json()) == _workspace_contract_fields(
        python_null.json()
    )

    python_clear = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={"metadata": {}, "configuration": {}},
    )
    rust_clear = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={"metadata": {}, "configuration": {}},
        timeout=5,
    )
    assert rust_clear.status_code == python_clear.status_code == 200
    assert python_clear.json()["metadata"] == {}
    assert rust_clear.json()["metadata"] == {}
    assert python_clear.json()["configuration"] == {}
    assert rust_clear.json()["configuration"] == {}

    python_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == python_workspace,
            models.Peer.name == python_peer,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == rust_workspace,
            models.Peer.name == rust_peer,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _peer_row_contract_fields(rust_row) == _peer_row_contract_fields(
        python_row
    ) == {
        "metadata": {},
        "configuration": {},
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_peer_update_creates_missing_peer_like_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-put-create-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-put-create-ws-{marker}"
    python_peer = f"rust-contract-python-peer-put-create-{marker}"
    rust_peer = f"rust-contract-rust-peer-put-create-{marker}"
    metadata = {"created_by": "put", "marker": marker}
    configuration = {"feature": "created"}

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={"metadata": metadata, "configuration": configuration},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={"metadata": metadata, "configuration": configuration},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_peer
    assert rust_response.json()["id"] == rust_peer
    assert python_response.json()["metadata"] == metadata
    assert rust_response.json()["metadata"] == metadata
    assert python_response.json()["configuration"] == configuration
    assert rust_response.json()["configuration"] == configuration

    python_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == python_workspace,
            models.Peer.name == python_peer,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == rust_workspace,
            models.Peer.name == rust_peer,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _peer_row_contract_fields(rust_row) == _peer_row_contract_fields(
        python_row
    ) == {
        "metadata": metadata,
        "configuration": configuration,
        "internal_metadata": {},
    }


def test_peer_write_validation_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    workspace = f"rust-contract-peer-validation-{generate_nanoid()}"
    peer = f"rust-contract-peer-validation-{generate_nanoid()}"
    too_many_keys = {str(index): index for index in range(101)}
    too_deep_metadata = {"a": {"b": {"c": {"d": {"e": {"f": "x"}}}}}}
    create_payloads: list[Any] = [
        {},
        {"name": ""},
        {"id": "invalid name"},
        {"name": peer, "metadata": "bad"},
        {"name": peer, "metadata": []},
        {"name": peer, "metadata": too_many_keys},
        {"name": peer, "metadata": too_deep_metadata},
        {"name": peer, "configuration": "bad"},
        {"name": peer, "configuration": []},
    ]
    update_payloads: list[Any] = [
        [],
        {"metadata": "bad"},
        {"metadata": []},
        {"metadata": too_many_keys},
        {"metadata": too_deep_metadata},
        {"configuration": "bad"},
        {"configuration": []},
    ]

    for payload in create_payloads:
        python_response = client.post(
            f"/v3/workspaces/{workspace}/peers",
            json=payload,
        )
        rust_response = httpx.post(
            f"{rust_api_writes_url}/v3/workspaces/{workspace}/peers",
            json=payload,
            timeout=5,
        )

        assert rust_response.status_code == python_response.status_code == 422
        assert rust_response.json() == python_response.json()

    for payload in update_payloads:
        python_response = client.put(
            f"/v3/workspaces/{workspace}/peers/{peer}",
            json=payload,
        )
        rust_response = httpx.put(
            f"{rust_api_writes_url}/v3/workspaces/{workspace}/peers/{peer}",
            json=payload,
            timeout=5,
        )

        assert rust_response.status_code == python_response.status_code == 422
        assert rust_response.json() == python_response.json()


def test_peer_write_auth_scopes_match_fastapi(
    client: TestClient,
    rust_api_auth_writes_url: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-auth-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-auth-ws-{marker}"
    python_peer = f"rust-contract-python-peer-auth-{marker}"
    rust_peer = f"rust-contract-rust-peer-auth-{marker}"

    python_admin = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={"name": python_peer},
        headers={"Authorization": f"Bearer {_test_jwt({'ad': True})}"},
    )
    rust_admin = httpx.post(
        f"{rust_api_auth_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={"name": rust_peer},
        headers={"Authorization": f"Bearer {_test_jwt({'ad': True})}"},
        timeout=5,
    )
    assert rust_admin.status_code == python_admin.status_code == 201

    workspace_token = _test_jwt({"w": python_workspace})
    rust_workspace_token = _test_jwt({"w": rust_workspace})
    python_workspace_scoped = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={"name": python_peer, "metadata": {"workspace": True}},
        headers={"Authorization": f"Bearer {workspace_token}"},
    )
    rust_workspace_scoped = httpx.post(
        f"{rust_api_auth_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={"name": rust_peer, "metadata": {"workspace": True}},
        headers={"Authorization": f"Bearer {rust_workspace_token}"},
        timeout=5,
    )
    assert rust_workspace_scoped.status_code == python_workspace_scoped.status_code == 200

    peer_token = _test_jwt({"w": python_workspace, "p": python_peer})
    rust_peer_token = _test_jwt({"w": rust_workspace, "p": rust_peer})
    python_peer_scoped = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={"metadata": {"peer": True}},
        headers={"Authorization": f"Bearer {peer_token}"},
    )
    rust_peer_scoped = httpx.put(
        f"{rust_api_auth_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={"metadata": {"peer": True}},
        headers={"Authorization": f"Bearer {rust_peer_token}"},
        timeout=5,
    )
    assert rust_peer_scoped.status_code == python_peer_scoped.status_code == 200

    python_wrong_workspace = client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={"name": python_peer},
        headers={
            "Authorization": f"Bearer {_test_jwt({'w': python_workspace + '-other'})}"
        },
    )
    rust_wrong_workspace = httpx.post(
        f"{rust_api_auth_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={"name": rust_peer},
        headers={
            "Authorization": f"Bearer {_test_jwt({'w': rust_workspace + '-other'})}"
        },
        timeout=5,
    )
    assert rust_wrong_workspace.status_code == python_wrong_workspace.status_code == 401
    assert rust_wrong_workspace.json() == python_wrong_workspace.json()

    python_wrong_peer = client.put(
        f"/v3/workspaces/{python_workspace}/peers/{python_peer}",
        json={"metadata": {"denied": True}},
        headers={
            "Authorization": f"Bearer {_test_jwt({'p': python_peer + '-other'})}"
        },
    )
    rust_wrong_peer = httpx.put(
        f"{rust_api_auth_writes_url}/v3/workspaces/{rust_workspace}/peers/{rust_peer}",
        json={"metadata": {"denied": True}},
        headers={
            "Authorization": f"Bearer {_test_jwt({'p': rust_peer + '-other'})}"
        },
        timeout=5,
    )
    assert rust_wrong_peer.status_code == python_wrong_peer.status_code == 401
    assert rust_wrong_peer.json() == python_wrong_peer.json()


@pytest.mark.asyncio
async def test_session_get_or_create_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-ws-{marker}"
    python_session = f"rust-contract-python-session-{marker}"
    rust_session = f"rust-contract-rust-session-{marker}"
    metadata_payload = {"kind": "session-write", "nul": "a\x00b"}
    metadata = {"kind": "session-write", "nul": "ab"}
    configuration_payload = {
        "feature": True,
        "summary": {"enabled": "true", "messages_per_short_summary": 20},
    }
    configuration = {
        "feature": True,
        "summary": {"enabled": True, "messages_per_short_summary": 20},
    }

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={
            "id": python_session,
            "metadata": metadata_payload,
            "configuration": configuration_payload,
        },
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={
            "name": rust_session,
            "metadata": metadata_payload,
            "configuration": configuration_payload,
        },
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    assert python_response.json()["id"] == python_session
    assert rust_response.json()["id"] == rust_session
    assert python_response.json()["workspace_id"] == python_workspace
    assert rust_response.json()["workspace_id"] == rust_workspace
    assert python_response.json()["is_active"] is True
    assert rust_response.json()["is_active"] is True
    assert _workspace_contract_fields(rust_response.json()) == _workspace_contract_fields(
        python_response.json()
    ) == {"metadata": metadata, "configuration": configuration}

    replacement_metadata = {"replacement": marker}
    replacement_configuration_payload = {
        "dream": {"enabled": "false"},
        "feature2": False,
    }
    replacement_configuration = {
        **configuration,
        "dream": {"enabled": False},
        "feature2": False,
    }
    python_existing = client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={
            "id": python_session,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration_payload,
        },
    )
    rust_existing = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={
            "id": rust_session,
            "metadata": replacement_metadata,
            "configuration": replacement_configuration_payload,
        },
        timeout=5,
    )

    assert rust_existing.status_code == python_existing.status_code == 200
    assert _workspace_contract_fields(rust_existing.json()) == _workspace_contract_fields(
        python_existing.json()
    ) == {
        "metadata": replacement_metadata,
        "configuration": replacement_configuration,
    }

    python_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == python_workspace,
            models.Session.name == python_session,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == rust_workspace,
            models.Session.name == rust_session,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _session_row_contract_fields(rust_row) == _session_row_contract_fields(
        python_row
    ) == {
        "is_active": True,
        "metadata": replacement_metadata,
        "configuration": replacement_configuration,
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_session_update_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-update-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-update-ws-{marker}"
    python_session = f"rust-contract-python-session-update-{marker}"
    rust_session = f"rust-contract-rust-session-update-{marker}"
    initial_metadata = {"initial": marker}
    initial_configuration = {"initial_feature": True}
    update_metadata_payload = {"updated": marker, "nul": "x\x00y"}
    update_metadata = {"updated": marker, "nul": "xy"}
    update_configuration_payload = {"summary": {"enabled": "false"}}
    update_configuration = {
        "initial_feature": True,
        "summary": {"enabled": False},
    }

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={
            "id": python_session,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={
            "id": rust_session,
            "metadata": initial_metadata,
            "configuration": initial_configuration,
        },
        timeout=5,
    ).status_code == 201

    python_update = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
        json={
            "id": "ignored",
            "name": "also-ignored",
            "metadata": update_metadata_payload,
            "configuration": update_configuration_payload,
        },
    )
    rust_update = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        json={
            "id": "ignored",
            "name": "also-ignored",
            "metadata": update_metadata_payload,
            "configuration": update_configuration_payload,
        },
        timeout=5,
    )

    assert rust_update.status_code == python_update.status_code == 200
    assert python_update.json()["id"] == python_session
    assert rust_update.json()["id"] == rust_session
    assert _workspace_contract_fields(rust_update.json()) == _workspace_contract_fields(
        python_update.json()
    ) == {"metadata": update_metadata, "configuration": update_configuration}

    python_null = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
        json={"metadata": None, "configuration": None},
    )
    rust_null = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        json={"metadata": None, "configuration": None},
        timeout=5,
    )
    assert rust_null.status_code == python_null.status_code == 200
    assert _workspace_contract_fields(rust_null.json()) == _workspace_contract_fields(
        python_null.json()
    ) == {"metadata": update_metadata, "configuration": update_configuration}

    python_clear = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
        json={"metadata": {}, "configuration": {}},
    )
    rust_clear = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        json={"metadata": {}, "configuration": {}},
        timeout=5,
    )
    assert rust_clear.status_code == python_clear.status_code == 200
    assert _workspace_contract_fields(rust_clear.json()) == _workspace_contract_fields(
        python_clear.json()
    ) == {"metadata": {}, "configuration": update_configuration}

    python_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == python_workspace,
            models.Session.name == python_session,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == rust_workspace,
            models.Session.name == rust_session,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _session_row_contract_fields(rust_row) == _session_row_contract_fields(
        python_row
    ) == {
        "is_active": True,
        "metadata": {},
        "configuration": update_configuration,
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_session_update_creates_missing_session_like_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-put-create-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-put-create-ws-{marker}"
    python_session = f"rust-contract-python-session-put-create-{marker}"
    rust_session = f"rust-contract-rust-session-put-create-{marker}"
    metadata = {"created_by": "put", "marker": marker}
    configuration_payload = {"peer_card": {"use": "true"}}
    configuration = {"peer_card": {"use": True}}

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
        json={"metadata": metadata, "configuration": configuration_payload},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        json={"metadata": metadata, "configuration": configuration_payload},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_session
    assert rust_response.json()["id"] == rust_session
    assert _workspace_contract_fields(rust_response.json()) == _workspace_contract_fields(
        python_response.json()
    ) == {"metadata": metadata, "configuration": configuration}

    python_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == python_workspace,
            models.Session.name == python_session,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == rust_workspace,
            models.Session.name == rust_session,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _session_row_contract_fields(rust_row) == _session_row_contract_fields(
        python_row
    ) == {
        "is_active": True,
        "metadata": metadata,
        "configuration": configuration,
        "internal_metadata": {},
    }


@pytest.mark.asyncio
async def test_session_add_peers_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-add-peer-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-add-peer-ws-{marker}"
    python_session = f"rust-contract-python-session-add-peer-{marker}"
    rust_session = f"rust-contract-rust-session-add-peer-{marker}"
    python_peer = f"rust-contract-python-session-added-peer-{marker}"
    rust_peer = f"rust-contract-rust-session-added-peer-{marker}"
    config_payload = {"observe_me": "true", "observe_others": "false"}
    expected_config = {"observe_me": True, "observe_others": False}

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_peer: config_payload},
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_peer: config_payload},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_session
    assert rust_response.json()["id"] == rust_session
    assert _workspace_contract_fields(rust_response.json()) == _workspace_contract_fields(
        python_response.json()
    ) == {"metadata": {}, "configuration": {}}

    python_peer_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == python_workspace,
            models.Peer.name == python_peer,
        )
    )
    rust_peer_row = await db_session.scalar(
        select(models.Peer).where(
            models.Peer.workspace_name == rust_workspace,
            models.Peer.name == rust_peer,
        )
    )
    assert python_peer_row is not None
    assert rust_peer_row is not None
    assert _peer_row_contract_fields(rust_peer_row) == _peer_row_contract_fields(
        python_peer_row
    ) == {
        "metadata": {},
        "configuration": {},
        "internal_metadata": {},
    }

    python_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_peer,
        )
    )
    rust_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_peer,
        )
    )
    assert python_session_peer is not None
    assert rust_session_peer is not None
    assert _session_peer_row_contract_fields(
        rust_session_peer
    ) == _session_peer_row_contract_fields(python_session_peer) == {
        "configuration": expected_config,
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_session_add_peers_preserves_active_config_and_rejoins_left_peer(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-readd-peer-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-readd-peer-ws-{marker}"
    python_session = f"rust-contract-python-session-readd-peer-{marker}"
    rust_session = f"rust-contract-rust-session-readd-peer-{marker}"
    python_active_peer = f"rust-contract-python-session-active-peer-{marker}"
    rust_active_peer = f"rust-contract-rust-session-active-peer-{marker}"
    python_rejoin_peer = f"rust-contract-python-session-rejoin-peer-{marker}"
    rust_rejoin_peer = f"rust-contract-rust-session-rejoin-peer-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201

    initial_active_config = {"observe_me": True, "observe_others": True}
    changed_active_config = {"observe_me": False, "observe_others": False}
    initial_rejoin_config = {"observe_me": False, "observe_others": True}
    changed_rejoin_config = {"observe_me": True, "observe_others": False}

    for workspace, session, active_peer, rejoin_peer, base_url in (
        (python_workspace, python_session, python_active_peer, python_rejoin_peer, ""),
        (
            rust_workspace,
            rust_session,
            rust_active_peer,
            rust_rejoin_peer,
            rust_api_writes_url,
        ),
    ):
        post = client.post if not base_url else httpx.post
        timeout = {} if not base_url else {"timeout": 5}
        assert post(
            f"{base_url}/v3/workspaces/{workspace}/sessions/{session}/peers",
            json={
                active_peer: initial_active_config,
                rejoin_peer: initial_rejoin_config,
            },
            **timeout,
        ).status_code == 200

    await db_session.execute(
        update(models.SessionPeer)
        .where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_rejoin_peer,
        )
        .values(left_at=datetime.now(UTC))
    )
    await db_session.execute(
        update(models.SessionPeer)
        .where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_rejoin_peer,
        )
        .values(left_at=datetime.now(UTC))
    )
    await db_session.commit()

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={
            python_active_peer: changed_active_config,
            python_rejoin_peer: changed_rejoin_config,
        },
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={
            rust_active_peer: changed_active_config,
            rust_rejoin_peer: changed_rejoin_config,
        },
        timeout=5,
    )
    assert rust_response.status_code == python_response.status_code == 200

    python_active = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_active_peer,
        )
    )
    rust_active = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_active_peer,
        )
    )
    python_rejoin = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_rejoin_peer,
        )
    )
    rust_rejoin = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_rejoin_peer,
        )
    )
    assert python_active is not None
    assert rust_active is not None
    assert python_rejoin is not None
    assert rust_rejoin is not None
    assert _session_peer_row_contract_fields(
        rust_active
    ) == _session_peer_row_contract_fields(python_active) == {
        "configuration": initial_active_config,
        "is_active": True,
    }
    assert _session_peer_row_contract_fields(
        rust_rejoin
    ) == _session_peer_row_contract_fields(python_rejoin) == {
        "configuration": changed_rejoin_config,
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_session_set_peers_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-set-peer-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-set-peer-ws-{marker}"
    python_session = f"rust-contract-python-session-set-peer-{marker}"
    rust_session = f"rust-contract-rust-session-set-peer-{marker}"
    python_old_peer = f"rust-contract-python-session-old-peer-{marker}"
    rust_old_peer = f"rust-contract-rust-session-old-peer-{marker}"
    python_new_peer = f"rust-contract-python-session-new-peer-{marker}"
    rust_new_peer = f"rust-contract-rust-session-new-peer-{marker}"
    initial_config = {"observe_me": True, "observe_others": True}
    replacement_config_payload = {"observe_me": "false", "observe_others": "true"}
    replacement_config = {"observe_me": False, "observe_others": True}

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_old_peer: initial_config},
    ).status_code == 200
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_old_peer: initial_config},
        timeout=5,
    ).status_code == 200

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_new_peer: replacement_config_payload},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_new_peer: replacement_config_payload},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_session
    assert rust_response.json()["id"] == rust_session
    assert _workspace_contract_fields(rust_response.json()) == _workspace_contract_fields(
        python_response.json()
    ) == {"metadata": {}, "configuration": {}}

    python_old = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_old_peer,
        )
    )
    rust_old = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_old_peer,
        )
    )
    python_new = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_new_peer,
        )
    )
    rust_new = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_new_peer,
        )
    )
    assert python_old is not None
    assert rust_old is not None
    assert python_new is not None
    assert rust_new is not None
    assert _session_peer_row_contract_fields(
        rust_old
    ) == _session_peer_row_contract_fields(python_old) == {
        "configuration": initial_config,
        "is_active": False,
    }
    assert _session_peer_row_contract_fields(
        rust_new
    ) == _session_peer_row_contract_fields(python_new) == {
        "configuration": replacement_config,
        "is_active": True,
    }


def test_session_set_peers_missing_session_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-set-missing-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-set-missing-ws-{marker}"
    python_session = f"rust-contract-python-session-set-missing-{marker}"
    rust_session = f"rust-contract-rust-session-set-missing-{marker}"

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={"peer-a": {}},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={"peer-a": {}},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 404
    assert python_response.json() == {
        "detail": f"Session {python_session} not found in workspace {python_workspace}"
    }
    assert rust_response.json() == {
        "detail": f"Session {rust_session} not found in workspace {rust_workspace}"
    }


@pytest.mark.asyncio
async def test_session_remove_peers_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-remove-peer-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-remove-peer-ws-{marker}"
    python_session = f"rust-contract-python-session-remove-peer-{marker}"
    rust_session = f"rust-contract-rust-session-remove-peer-{marker}"
    python_removed_peer = f"rust-contract-python-session-removed-peer-{marker}"
    rust_removed_peer = f"rust-contract-rust-session-removed-peer-{marker}"
    python_kept_peer = f"rust-contract-python-session-kept-peer-{marker}"
    rust_kept_peer = f"rust-contract-rust-session-kept-peer-{marker}"
    removed_config = {"observe_me": True, "observe_others": False}
    kept_config = {"observe_me": False, "observe_others": True}

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_removed_peer: removed_config, python_kept_peer: kept_config},
    ).status_code == 200
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_removed_peer: removed_config, rust_kept_peer: kept_config},
        timeout=5,
    ).status_code == 200

    python_response = client.request(
        "DELETE",
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json=[python_removed_peer, "missing-peer"],
    )
    rust_response = httpx.request(
        "DELETE",
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json=[rust_removed_peer, "missing-peer"],
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    assert python_response.json()["id"] == python_session
    assert rust_response.json()["id"] == rust_session
    assert _workspace_contract_fields(rust_response.json()) == _workspace_contract_fields(
        python_response.json()
    ) == {"metadata": {}, "configuration": {}}

    python_removed = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_removed_peer,
        )
    )
    rust_removed = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_removed_peer,
        )
    )
    python_kept = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_kept_peer,
        )
    )
    rust_kept = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_kept_peer,
        )
    )
    assert python_removed is not None
    assert rust_removed is not None
    assert python_kept is not None
    assert rust_kept is not None
    assert _session_peer_row_contract_fields(
        rust_removed
    ) == _session_peer_row_contract_fields(python_removed) == {
        "configuration": removed_config,
        "is_active": False,
    }
    assert _session_peer_row_contract_fields(
        rust_kept
    ) == _session_peer_row_contract_fields(python_kept) == {
        "configuration": kept_config,
        "is_active": True,
    }


def test_session_remove_peers_missing_session_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-remove-missing-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-remove-missing-ws-{marker}"
    python_session = f"rust-contract-python-session-remove-missing-{marker}"
    rust_session = f"rust-contract-rust-session-remove-missing-{marker}"

    python_response = client.request(
        "DELETE",
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json=["peer-a"],
    )
    rust_response = httpx.request(
        "DELETE",
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json=["peer-a"],
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 404
    assert python_response.json() == {
        "detail": f"Session {python_session} not found in workspace {python_workspace}"
    }
    assert rust_response.json() == {
        "detail": f"Session {rust_session} not found in workspace {rust_workspace}"
    }


@pytest.mark.asyncio
async def test_session_peer_config_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-config-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-config-ws-{marker}"
    python_session = f"rust-contract-python-peer-config-session-{marker}"
    rust_session = f"rust-contract-rust-peer-config-session-{marker}"
    python_peer = f"rust-contract-python-peer-config-peer-{marker}"
    rust_peer = f"rust-contract-rust-peer-config-peer-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201
    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_peer: {"observe_me": True}},
    ).status_code == 200
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_peer: {"observe_me": True}},
        timeout=5,
    ).status_code == 200

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers/{python_peer}/config",
        json={"observe_others": "true", "observe_me": None},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers/{rust_peer}/config",
        json={"observe_others": "true", "observe_me": None},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 204
    assert rust_response.content == python_response.content == b""

    python_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_peer,
        )
    )
    rust_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_peer,
        )
    )
    assert python_session_peer is not None
    assert rust_session_peer is not None
    assert _session_peer_row_contract_fields(
        rust_session_peer
    ) == _session_peer_row_contract_fields(python_session_peer) == {
        "configuration": {"observe_me": True, "observe_others": True},
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_session_peer_config_creates_missing_membership_like_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-config-create-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-config-create-ws-{marker}"
    python_session = f"rust-contract-python-peer-config-create-session-{marker}"
    rust_session = f"rust-contract-rust-peer-config-create-session-{marker}"
    python_peer = f"rust-contract-python-peer-config-create-peer-{marker}"
    rust_peer = f"rust-contract-rust-peer-config-create-peer-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201
    assert client.post(
        f"/v3/workspaces/{python_workspace}/peers",
        json={"id": python_peer},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/peers",
        json={"id": rust_peer},
        timeout=5,
    ).status_code == 201

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers/{python_peer}/config",
        json={"observe_me": "true"},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers/{rust_peer}/config",
        json={"observe_me": "true"},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 204
    assert rust_response.content == python_response.content == b""

    python_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_peer,
        )
    )
    rust_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_peer,
        )
    )
    assert python_session_peer is not None
    assert rust_session_peer is not None
    assert _session_peer_row_contract_fields(
        rust_session_peer
    ) == _session_peer_row_contract_fields(python_session_peer) == {
        "configuration": {"observe_me": True},
        "is_active": True,
    }


@pytest.mark.asyncio
async def test_session_peer_config_updates_left_membership_without_rejoin(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-config-left-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-config-left-ws-{marker}"
    python_session = f"rust-contract-python-peer-config-left-session-{marker}"
    rust_session = f"rust-contract-rust-peer-config-left-session-{marker}"
    python_peer = f"rust-contract-python-peer-config-left-peer-{marker}"
    rust_peer = f"rust-contract-rust-peer-config-left-peer-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201
    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json={python_peer: {"observe_me": True}},
    ).status_code == 200
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json={rust_peer: {"observe_me": True}},
        timeout=5,
    ).status_code == 200
    assert client.request(
        "DELETE",
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers",
        json=[python_peer],
    ).status_code == 200
    assert httpx.request(
        "DELETE",
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers",
        json=[rust_peer],
        timeout=5,
    ).status_code == 200

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers/{python_peer}/config",
        json={"observe_others": True},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers/{rust_peer}/config",
        json={"observe_others": True},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 204
    assert rust_response.content == python_response.content == b""

    python_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == python_workspace,
            models.SessionPeer.session_name == python_session,
            models.SessionPeer.peer_name == python_peer,
        )
    )
    rust_session_peer = await db_session.scalar(
        select(models.SessionPeer).where(
            models.SessionPeer.workspace_name == rust_workspace,
            models.SessionPeer.session_name == rust_session,
            models.SessionPeer.peer_name == rust_peer,
        )
    )
    assert python_session_peer is not None
    assert rust_session_peer is not None
    assert _session_peer_row_contract_fields(
        rust_session_peer
    ) == _session_peer_row_contract_fields(python_session_peer) == {
        "configuration": {"observe_me": True, "observe_others": True},
        "is_active": False,
    }


def test_session_peer_config_missing_peer_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-peer-config-missing-ws-{marker}"
    rust_workspace = f"rust-contract-rust-peer-config-missing-ws-{marker}"
    python_session = f"rust-contract-python-peer-config-missing-session-{marker}"
    rust_session = f"rust-contract-rust-peer-config-missing-session-{marker}"
    python_peer = f"rust-contract-python-peer-config-missing-peer-{marker}"
    rust_peer = f"rust-contract-rust-peer-config-missing-peer-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session},
        timeout=5,
    ).status_code == 201

    python_response = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/peers/{python_peer}/config",
        json={"observe_me": True},
    )
    rust_response = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/peers/{rust_peer}/config",
        json={"observe_me": True},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 404
    assert python_response.json() == {
        "detail": f"Peer {python_peer} not found in workspace {python_workspace}"
    }
    assert rust_response.json() == {
        "detail": f"Peer {rust_peer} not found in workspace {rust_workspace}"
    }


@pytest.mark.asyncio
async def test_session_delete_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-delete-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-delete-ws-{marker}"
    python_session = f"rust-contract-python-session-delete-{marker}"
    rust_session = f"rust-contract-rust-session-delete-{marker}"

    assert client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session, "metadata": {"delete": "me"}},
    ).status_code == 201
    assert httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session, "metadata": {"delete": "me"}},
        timeout=5,
    ).status_code == 201

    python_response = client.delete(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
    )
    rust_response = httpx.delete(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 202
    assert rust_response.json() == python_response.json() == {
        "message": "Session deleted successfully"
    }

    python_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == python_workspace,
            models.Session.name == python_session,
        )
    )
    rust_row = await db_session.scalar(
        select(models.Session).where(
            models.Session.workspace_name == rust_workspace,
            models.Session.name == rust_session,
        )
    )
    assert python_row is not None
    assert rust_row is not None
    assert _session_row_contract_fields(rust_row) == _session_row_contract_fields(
        python_row
    ) == {
        "is_active": False,
        "metadata": {"delete": "me"},
        "configuration": {},
        "internal_metadata": {},
    }

    python_queue = await db_session.scalar(
        select(models.QueueItem).where(
            models.QueueItem.workspace_name == python_workspace,
            models.QueueItem.task_type == "deletion",
            models.QueueItem.work_unit_key
            == f"deletion:{python_workspace}:session:{python_session}",
        )
    )
    rust_queue = await db_session.scalar(
        select(models.QueueItem).where(
            models.QueueItem.workspace_name == rust_workspace,
            models.QueueItem.task_type == "deletion",
            models.QueueItem.work_unit_key
            == f"deletion:{rust_workspace}:session:{rust_session}",
        )
    )
    assert python_queue is not None
    assert rust_queue is not None
    assert {
        "payload": rust_queue.payload,
        "processed": rust_queue.processed,
        "session_id": rust_queue.session_id,
        "message_id": rust_queue.message_id,
    } == {
        "payload": {
            "task_type": "deletion",
            "deletion_type": "session",
            "resource_id": rust_session,
        },
        "processed": False,
        "session_id": None,
        "message_id": None,
    }
    assert {
        "payload": python_queue.payload,
        "processed": python_queue.processed,
        "session_id": python_queue.session_id,
        "message_id": python_queue.message_id,
    } == {
        "payload": {
            "task_type": "deletion",
            "deletion_type": "session",
            "resource_id": python_session,
        },
        "processed": False,
        "session_id": None,
        "message_id": None,
    }


def test_session_delete_missing_session_matches_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-delete-missing-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-delete-missing-ws-{marker}"
    python_session = f"rust-contract-python-session-delete-missing-{marker}"
    rust_session = f"rust-contract-rust-session-delete-missing-{marker}"

    python_response = client.delete(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
    )
    rust_response = httpx.delete(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 404
    assert python_response.json() == {
        "detail": f"Session {python_session} not found in workspace {python_workspace}"
    }
    assert rust_response.json() == {
        "detail": f"Session {rust_session} not found in workspace {rust_workspace}"
    }


def _clone_message_contract_fields(message: models.Message) -> dict[str, Any]:
    return {
        "content": message.content,
        "peer_name": message.peer_name,
        "seq_in_session": message.seq_in_session,
        "token_count": message.token_count,
        "metadata": message.h_metadata,
        "internal_metadata": message.internal_metadata,
    }


def _clone_response_contract_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in body.items()
        if key not in ("id", "created_at", "workspace_id")
    }


def _seed_clone_source(
    client: TestClient, workspace: str, session: str, peer: str
) -> list[str]:
    assert client.post("/v3/workspaces", json={"name": workspace}).status_code in (
        200,
        201,
    )
    assert client.post(
        f"/v3/workspaces/{workspace}/peers", json={"name": peer}
    ).status_code in (200, 201)
    assert client.post(
        f"/v3/workspaces/{workspace}/sessions",
        json={
            "name": session,
            "metadata": {"src": "clone"},
            "peer_names": {peer: {"observe_me": True}},
        },
    ).status_code in (200, 201)
    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/{session}/messages",
        json={
            "messages": [
                {"content": "clone message one", "peer_id": peer, "metadata": {"order": 1}},
                {"content": "clone message two", "peer_id": peer, "metadata": {"order": 2}},
            ]
        },
    )
    assert response.status_code == 201
    return [message["id"] for message in response.json()]


@pytest.mark.asyncio
async def test_session_clone_write_shadow_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-clone-ws-{marker}"
    rust_workspace = f"rust-contract-rust-clone-ws-{marker}"
    python_session = f"rust-contract-python-clone-{marker}"
    rust_session = f"rust-contract-rust-clone-{marker}"
    peer = f"rust-contract-clone-peer-{marker}"

    _seed_clone_source(client, python_workspace, python_session, peer)
    _seed_clone_source(client, rust_workspace, rust_session, peer)

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/clone",
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/clone",
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    python_body = python_response.json()
    rust_body = rust_response.json()
    assert python_body["workspace_id"] == python_workspace
    assert rust_body["workspace_id"] == rust_workspace
    assert _clone_response_contract_fields(
        python_body
    ) == _clone_response_contract_fields(rust_body) == {
        "is_active": True,
        "metadata": {"src": "clone"},
        "configuration": {},
    }
    # The clone gets a fresh nanoid name, distinct from the source session.
    assert python_body["id"] != python_session
    assert rust_body["id"] != rust_session
    python_clone = python_body["id"]
    rust_clone = rust_body["id"]

    python_messages = (
        await db_session.scalars(
            select(models.Message)
            .where(
                models.Message.workspace_name == python_workspace,
                models.Message.session_name == python_clone,
            )
            .order_by(models.Message.seq_in_session)
        )
    ).all()
    rust_messages = (
        await db_session.scalars(
            select(models.Message)
            .where(
                models.Message.workspace_name == rust_workspace,
                models.Message.session_name == rust_clone,
            )
            .order_by(models.Message.seq_in_session)
        )
    ).all()
    assert len(rust_messages) == 2
    assert [_clone_message_contract_fields(m) for m in python_messages] == [
        _clone_message_contract_fields(m) for m in rust_messages
    ]

    python_peers = (
        await db_session.scalars(
            select(models.SessionPeer)
            .where(
                models.SessionPeer.workspace_name == python_workspace,
                models.SessionPeer.session_name == python_clone,
            )
            .order_by(models.SessionPeer.peer_name)
        )
    ).all()
    rust_peers = (
        await db_session.scalars(
            select(models.SessionPeer)
            .where(
                models.SessionPeer.workspace_name == rust_workspace,
                models.SessionPeer.session_name == rust_clone,
            )
            .order_by(models.SessionPeer.peer_name)
        )
    ).all()
    assert len(rust_peers) == 1
    assert [_session_peer_row_contract_fields(p) for p in python_peers] == [
        _session_peer_row_contract_fields(p) for p in rust_peers
    ]


@pytest.mark.asyncio
async def test_session_clone_cutoff_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-clonecut-ws-{marker}"
    rust_workspace = f"rust-contract-rust-clonecut-ws-{marker}"
    python_session = f"rust-contract-python-clonecut-{marker}"
    rust_session = f"rust-contract-rust-clonecut-{marker}"
    peer = f"rust-contract-clonecut-peer-{marker}"

    python_message_ids = _seed_clone_source(
        client, python_workspace, python_session, peer
    )
    rust_message_ids = _seed_clone_source(client, rust_workspace, rust_session, peer)

    python_response = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/clone",
        params={"message_id": python_message_ids[0]},
    )
    rust_response = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/clone",
        params={"message_id": rust_message_ids[0]},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 201
    python_clone = python_response.json()["id"]
    rust_clone = rust_response.json()["id"]

    python_messages = (
        await db_session.scalars(
            select(models.Message)
            .where(
                models.Message.workspace_name == python_workspace,
                models.Message.session_name == python_clone,
            )
            .order_by(models.Message.seq_in_session)
        )
    ).all()
    rust_messages = (
        await db_session.scalars(
            select(models.Message)
            .where(
                models.Message.workspace_name == rust_workspace,
                models.Message.session_name == rust_clone,
            )
            .order_by(models.Message.seq_in_session)
        )
    ).all()
    # Only the first message (the cutoff) should be cloned.
    assert len(python_messages) == len(rust_messages) == 1
    assert [_clone_message_contract_fields(m) for m in python_messages] == [
        _clone_message_contract_fields(m) for m in rust_messages
    ]


def test_session_clone_missing_and_bad_cutoff_match_fastapi(
    client: TestClient,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-clonemiss-ws-{marker}"
    rust_workspace = f"rust-contract-rust-clonemiss-ws-{marker}"
    python_session = f"rust-contract-python-clonemiss-{marker}"
    rust_session = f"rust-contract-rust-clonemiss-{marker}"
    peer = f"rust-contract-clonemiss-peer-{marker}"

    # Missing original session -> "Original session not found".
    python_missing = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/clone",
    )
    rust_missing = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/clone",
        timeout=5,
    )
    assert rust_missing.status_code == python_missing.status_code == 404
    assert python_missing.json() == {"detail": "Original session not found"}
    assert rust_missing.json() == {"detail": "Original session not found"}

    # Existing session but unknown cutoff message -> "Session not found".
    _seed_clone_source(client, python_workspace, python_session, peer)
    _seed_clone_source(client, rust_workspace, rust_session, peer)
    bad_cutoff = generate_nanoid()
    python_bad = client.post(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}/clone",
        params={"message_id": bad_cutoff},
    )
    rust_bad = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}/clone",
        params={"message_id": bad_cutoff},
        timeout=5,
    )
    assert rust_bad.status_code == python_bad.status_code == 404
    assert python_bad.json() == {"detail": "Session not found"}
    assert rust_bad.json() == {"detail": "Session not found"}


@pytest.mark.asyncio
async def test_inactive_session_writes_match_fastapi_not_found(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_writes_url: str,
):
    marker = generate_nanoid()
    python_workspace = f"rust-contract-python-session-inactive-ws-{marker}"
    rust_workspace = f"rust-contract-rust-session-inactive-ws-{marker}"
    python_session = f"rust-contract-python-session-inactive-{marker}"
    rust_session = f"rust-contract-rust-session-inactive-{marker}"
    db_session.add_all(
        [
            models.Workspace(name=python_workspace),
            models.Workspace(name=rust_workspace),
            models.Session(
                workspace_name=python_workspace,
                name=python_session,
                is_active=False,
            ),
            models.Session(
                workspace_name=rust_workspace,
                name=rust_session,
                is_active=False,
            ),
        ]
    )
    await db_session.commit()

    python_post = client.post(
        f"/v3/workspaces/{python_workspace}/sessions",
        json={"id": python_session, "metadata": {"ignored": True}},
    )
    rust_post = httpx.post(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions",
        json={"id": rust_session, "metadata": {"ignored": True}},
        timeout=5,
    )
    assert rust_post.status_code == python_post.status_code == 404
    assert python_post.json() == {
        "detail": f"Session {python_session} not found in workspace {python_workspace}"
    }
    assert rust_post.json() == {
        "detail": f"Session {rust_session} not found in workspace {rust_workspace}"
    }

    python_put = client.put(
        f"/v3/workspaces/{python_workspace}/sessions/{python_session}",
        json={"metadata": {"ignored": True}},
    )
    rust_put = httpx.put(
        f"{rust_api_writes_url}/v3/workspaces/{rust_workspace}/sessions/{rust_session}",
        json={"metadata": {"ignored": True}},
        timeout=5,
    )
    assert rust_put.status_code == python_put.status_code == 404
    assert python_put.json() == {
        "detail": f"Session {python_session} not found in workspace {python_workspace}"
    }
    assert rust_put.json() == {
        "detail": f"Session {rust_session} not found in workspace {rust_workspace}"
    }


def test_key_creation_matches_fastapi_claims(
    rust_api_auth_url: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    admin_token = create_admin_jwt()
    key_app = FastAPI()
    key_app.include_router(keys_router, prefix="/v3")

    @key_app.exception_handler(HonchoException)
    async def honcho_exception_handler(_request, exc: HonchoException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    client = TestClient(key_app)
    client.headers["Authorization"] = f"Bearer {admin_token}"
    path = (
        "/v3/keys?workspace_id=workspace-a&peer_id=peer-a"
        "&session_id=session-a&expires_at=2030-06-15T10%3A20%3A30.456%2B03%3A00"
    )

    python_response = client.post(path)
    rust_response = httpx.post(
        f"{rust_api_auth_url}{path}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )

    assert rust_response.status_code == python_response.status_code == 200
    python_claims = jwt.decode(
        python_response.json()["key"],
        "test-secret",
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    rust_claims = jwt.decode(
        rust_response.json()["key"],
        "test-secret",
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    assert rust_claims["t"]
    assert rust_claims["exp"] == python_claims["exp"] == "2030-06-15T07:20:30Z"
    assert rust_claims["w"] == python_claims["w"] == "workspace-a"
    assert rust_claims["p"] == python_claims["p"] == "peer-a"
    assert rust_claims["s"] == python_claims["s"] == "session-a"
    assert "ad" not in rust_claims

    no_scope_path = "/v3/keys"
    python_no_scope = client.post(no_scope_path)
    rust_no_scope = httpx.post(
        f"{rust_api_auth_url}{no_scope_path}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    assert rust_no_scope.status_code == python_no_scope.status_code == 422
    assert rust_no_scope.json() == python_no_scope.json()

    empty_scope_path = "/v3/keys?workspace_id=&peer_id=&session_id="
    python_empty_scope = client.post(empty_scope_path)
    rust_empty_scope = httpx.post(
        f"{rust_api_auth_url}{empty_scope_path}",
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=5,
    )
    assert rust_empty_scope.status_code == python_empty_scope.status_code == 422
    assert rust_empty_scope.json() == python_empty_scope.json()


def test_workspace_list_matches_fastapi(client: TestClient, rust_api_url: str):
    marker = f"rust-contract-{generate_nanoid()}"
    first = f"{marker}-a"
    second = f"{marker}-b"
    for name in (first, second):
        response = client.post(
            "/v3/workspaces",
            json={"name": name, "metadata": {"contract": marker}},
        )
        assert response.status_code in (200, 201)

    _compare(
        client,
        rust_api_url,
        "POST",
        "/v3/workspaces/list?page=1&size=10",
        json_body={"filters": {"metadata": {"contract": marker}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        "/v3/workspaces/list?page=1&size=1&reverse=true",
        json_body={"filters": {"metadata": {"contract": marker}}},
    )


def test_peer_and_session_lists_match_fastapi(client: TestClient, rust_api_url: str):
    workspace = f"rust-contract-ws-{generate_nanoid()}"
    peer = f"rust-contract-peer-{generate_nanoid()}"
    session = f"rust-contract-session-{generate_nanoid()}"

    assert client.post("/v3/workspaces", json={"name": workspace}).status_code in (
        200,
        201,
    )
    assert client.post(
        f"/v3/workspaces/{workspace}/peers",
        json={"name": peer, "metadata": {"role": "contract"}},
    ).status_code in (200, 201)
    assert client.post(
        f"/v3/workspaces/{workspace}/sessions",
        json={
            "name": session,
            "metadata": {"contract_session": True},
            "peer_names": {peer: {}},
        },
    ).status_code in (200, 201)

    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/peers/list?page=1&size=10",
        json_body={"filters": {"metadata": {"role": "contract"}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/peers/list?page=1&size=10",
        json_body={"filters": {}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/peers/list?page=1&size=10",
        json_body={"filters": None},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/list?page=1&size=10",
        json_body={"filters": {"metadata": {"contract_session": True}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/peers/{peer}/sessions?page=1&size=10",
        json_body={"filters": {"metadata": {"contract_session": True}}},
    )


def test_session_peer_read_endpoints_match_fastapi(
    client: TestClient, rust_api_url: str
):
    workspace = f"rust-contract-ws-{generate_nanoid()}"
    peer = f"rust-contract-peer-{generate_nanoid()}"
    configured_peer = f"rust-contract-peer-{generate_nanoid()}"
    missing_peer = f"rust-contract-peer-{generate_nanoid()}"
    session = f"rust-contract-session-{generate_nanoid()}"

    assert client.post("/v3/workspaces", json={"name": workspace}).status_code in (
        200,
        201,
    )
    assert client.post(
        f"/v3/workspaces/{workspace}/peers",
        json={"name": peer, "metadata": {"contract_peer": "default"}},
    ).status_code in (200, 201)
    assert client.post(
        f"/v3/workspaces/{workspace}/peers",
        json={"name": configured_peer, "metadata": {"contract_peer": "configured"}},
    ).status_code in (200, 201)
    assert client.post(
        f"/v3/workspaces/{workspace}/sessions",
        json={
            "name": session,
            "peer_names": {
                peer: {},
                configured_peer: {"observe_others": True, "observe_me": False},
            },
        },
    ).status_code in (200, 201)

    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/peers?page=1&size=10",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/peers/{peer}/config",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/peers/{configured_peer}/config",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/peers/{missing_peer}/config",
    )

    delete_response = client.request(
        "DELETE",
        f"/v3/workspaces/{workspace}/sessions/{session}/peers",
        json=[peer],
    )
    assert delete_response.status_code == 200

    session_peers_path = (
        f"/v3/workspaces/{workspace}/sessions/{session}/peers?page=1&size=10"
    )
    _compare(client, rust_api_url, "GET", session_peers_path)

    response = client.get(session_peers_path)
    assert response.status_code == 200
    peer_ids = {item["id"] for item in response.json()["items"]}
    assert peer not in peer_ids
    assert configured_peer in peer_ids


@pytest.mark.asyncio
async def test_peer_card_read_endpoint_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_url: str,
):
    workspace = f"rust-contract-card-{generate_nanoid()}"
    observer = f"rust-contract-observer-{generate_nanoid()}"
    seeded_observer = f"rust-contract-observer-{generate_nanoid()}"
    target = f"rust-contract-target-{generate_nanoid()}"
    missing_observer = f"rust-contract-observer-{generate_nanoid()}"
    missing_target = f"rust-contract-target-{generate_nanoid()}"

    db_session.add(models.Workspace(name=workspace))
    db_session.add_all(
        [
            models.Peer(name=observer, workspace_name=workspace),
            models.Peer(
                name=seeded_observer,
                workspace_name=workspace,
                internal_metadata={
                    "peer_card": ["self fact", "self preference"],
                    f"{target}_peer_card": ["target fact", "target preference"],
                },
            ),
        ]
    )
    await db_session.commit()

    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/peers/{observer}/card",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/peers/{observer}/card?target={missing_target}",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/peers/{seeded_observer}/card",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/peers/{seeded_observer}/card?target={target}",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/peers/{missing_observer}/card",
    )


def test_message_read_endpoints_match_fastapi(client: TestClient, rust_api_url: str):
    workspace = f"rust-contract-ws-{generate_nanoid()}"
    peer = f"rust-contract-peer-{generate_nanoid()}"
    session = f"rust-contract-session-{generate_nanoid()}"
    marker = f"rust-contract-message-{generate_nanoid()}"

    assert client.post("/v3/workspaces", json={"name": workspace}).status_code in (
        200,
        201,
    )
    assert client.post(f"/v3/workspaces/{workspace}/peers", json={"name": peer}).status_code in (
        200,
        201,
    )
    assert client.post(
        f"/v3/workspaces/{workspace}/sessions",
        json={"name": session, "peer_names": {peer: {}}},
    ).status_code in (200, 201)

    response = client.post(
        f"/v3/workspaces/{workspace}/sessions/{session}/messages",
        json={
            "messages": [
                {
                    "content": "First Rust contract message",
                    "peer_id": peer,
                    "metadata": {"marker": marker, "order": 1},
                },
                {
                    "content": "Second Rust contract message",
                    "peer_id": peer,
                    "metadata": {"marker": marker, "order": 2},
                },
            ]
        },
    )
    assert response.status_code == 201
    message_id = response.json()[0]["id"]

    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": None},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10&reverse=true",
        json_body={},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"metadata": {"marker": marker}}},
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/{message_id}",
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"id": message_id}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"public_id": message_id}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"content": "First Rust contract message"}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"peer_name": peer}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"session_name": session}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/list?page=1&size=10",
        json_body={"filters": {"workspace_name": workspace}},
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace}/sessions/{session}/messages/{generate_nanoid()}",
    )


@pytest.mark.asyncio
async def test_conclusion_list_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_url: str,
):
    workspace = f"rust-contract-conclusions-{generate_nanoid()}"
    observer = f"rust-contract-observer-{generate_nanoid()}"
    observed = f"rust-contract-observed-{generate_nanoid()}"
    other_observed = f"rust-contract-observed-{generate_nanoid()}"
    session = f"rust-contract-session-{generate_nanoid()}"
    marker = f"rust-contract-conclusion-{generate_nanoid()}"
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    db_session.add(models.Workspace(name=workspace))
    db_session.add_all(
        [
            models.Peer(name=observer, workspace_name=workspace),
            models.Peer(name=observed, workspace_name=workspace),
            models.Peer(name=other_observed, workspace_name=workspace),
        ]
    )
    db_session.add(models.Session(name=session, workspace_name=workspace))
    await db_session.flush()

    db_session.add_all(
        [
            models.Collection(
                observer=observer,
                observed=observed,
                workspace_name=workspace,
            ),
            models.Collection(
                observer=observer,
                observed=other_observed,
                workspace_name=workspace,
            ),
        ]
    )
    await db_session.flush()

    first_id = generate_nanoid()
    null_session_id = generate_nanoid()
    deleted_id = generate_nanoid()
    db_session.add_all(
        [
            models.Document(
                id=first_id,
                workspace_name=workspace,
                observer=observer,
                observed=observed,
                content="First Rust contract conclusion",
                session_name=session,
                internal_metadata={"marker": marker, "order": 1, "label": "plain-path"},
                created_at=base_time,
            ),
            models.Document(
                workspace_name=workspace,
                observer=observer,
                observed=observed,
                content="Second Rust contract conclusion",
                session_name=session,
                internal_metadata={
                    "marker": marker,
                    "order": 2,
                    "label": "literal 100%_path\\segment",
                },
                created_at=base_time + timedelta(minutes=1),
            ),
            models.Document(
                id=null_session_id,
                workspace_name=workspace,
                observer=observer,
                observed=other_observed,
                content="Rust contract conclusion without session",
                session_name=None,
                internal_metadata={"marker": marker, "order": 3, "label": "another-path"},
                created_at=base_time + timedelta(minutes=2),
            ),
            models.Document(
                id=deleted_id,
                workspace_name=workspace,
                observer=observer,
                observed=observed,
                content="Deleted Rust contract conclusion",
                session_name=session,
                internal_metadata={"marker": marker, "order": 4},
                created_at=base_time + timedelta(minutes=3),
                deleted_at=base_time + timedelta(minutes=4),
            ),
        ]
    )
    await db_session.commit()

    path = f"/v3/workspaces/{workspace}/conclusions/list?page=1&size=10"
    _compare(client, rust_api_url, "POST", path)
    _compare(client, rust_api_url, "POST", path, json_body={})
    _compare(client, rust_api_url, "POST", path, json_body={"filters": {}})
    _compare(client, rust_api_url, "POST", path, json_body={"filters": None})
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"session_id": session}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"observer_id": observer, "observed_id": observed}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"metadata": {"marker": marker}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"metadata": {"order": {"gte": 2}}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"metadata": {"order": {"in": [2, 3]}}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"metadata": {"order": {"in": []}}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"metadata": {"label": {"icontains": "100%_path"}}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={
            "filters": {
                "created_at": {"gte": (base_time + timedelta(minutes=1)).isoformat()}
            }
        },
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={"filters": {"created_at": {"in": []}}},
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        path,
        json_body={
            "filters": {
                "created_at": {
                    "in": [
                        (base_time + timedelta(minutes=1)).isoformat(),
                        (base_time + timedelta(minutes=2)).isoformat(),
                    ]
                }
            }
        },
    )
    _compare(
        client,
        rust_api_url,
        "POST",
        f"{path}&reverse=true",
        json_body={"filters": {"metadata": {"marker": marker}}},
    )

    response = client.post(path, json={"filters": {"metadata": {"marker": marker}}})
    assert response.status_code == 200
    items = response.json()["items"]
    ids = {item["id"] for item in items}
    assert deleted_id not in ids
    assert any(
        item["id"] == null_session_id and item["session_id"] is None for item in items
    )
    assert "internal_metadata" not in items[0]
    assert items[0]["id"] != first_id


@pytest.mark.asyncio
async def test_queue_status_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_url: str,
):
    workspace_name = f"rust-contract-queue-{generate_nanoid()}"
    session_name = f"rust-contract-session-{generate_nanoid()}"
    observer = f"rust-contract-observer-{generate_nanoid()}"
    observed = f"rust-contract-observed-{generate_nanoid()}"
    work_unit_in_progress = f"representation:{generate_nanoid()}"

    workspace = models.Workspace(name=workspace_name)
    session = models.Session(name=session_name, workspace_name=workspace_name)
    observer_peer = models.Peer(name=observer, workspace_name=workspace_name)
    observed_peer = models.Peer(name=observed, workspace_name=workspace_name)
    db_session.add_all([workspace, session, observer_peer, observed_peer])
    await db_session.flush()

    db_session.add_all(
        [
            models.QueueItem(
                session_id=session.id,
                work_unit_key=f"summary:{generate_nanoid()}",
                task_type="summary",
                payload={"observer": observer, "observed": observed},
                processed=True,
                workspace_name=workspace_name,
            ),
            models.QueueItem(
                session_id=session.id,
                work_unit_key=work_unit_in_progress,
                task_type="representation",
                payload={"observer": observer, "observed": observed},
                processed=False,
                workspace_name=workspace_name,
            ),
            models.QueueItem(
                session_id=session.id,
                work_unit_key=f"dream:{generate_nanoid()}",
                task_type="dream",
                payload={"observer": observer, "observed": observed},
                processed=False,
                workspace_name=workspace_name,
            ),
            models.ActiveQueueSession(work_unit_key=work_unit_in_progress),
        ]
    )
    await db_session.commit()

    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/queue/status",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/queue/status?session_id={session_name}",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/queue/status?observer_id={observer}",
    )


@pytest.mark.asyncio
async def test_webhook_list_matches_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_url: str,
):
    workspace_name = f"rust-contract-webhooks-{generate_nanoid()}"
    other_workspace = f"rust-contract-webhooks-{generate_nanoid()}"
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    db_session.add_all(
        [
            models.Workspace(name=workspace_name),
            models.Workspace(name=other_workspace),
        ]
    )
    await db_session.flush()

    db_session.add_all(
        [
            models.WebhookEndpoint(
                id=generate_nanoid(),
                workspace_name=workspace_name,
                url="https://example.com/webhooks/first",
                created_at=base_time,
            ),
            models.WebhookEndpoint(
                id=generate_nanoid(),
                workspace_name=other_workspace,
                url="https://example.com/webhooks/other",
                created_at=base_time + timedelta(minutes=1),
            ),
        ]
    )
    await db_session.commit()

    path = f"/v3/workspaces/{workspace_name}/webhooks?page=1&size=1"
    _compare(client, rust_api_url, "GET", path)

    response = client.get(path)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["workspace_id"] == workspace_name
    assert "workspace_name" not in body["items"][0]


@pytest.mark.asyncio
async def test_session_summaries_match_fastapi(
    client: TestClient,
    db_session: AsyncSession,
    rust_api_url: str,
):
    workspace_name = f"rust-contract-summary-{generate_nanoid()}"
    empty_session_name = f"rust-contract-session-{generate_nanoid()}"
    summary_session_name = f"rust-contract-session-{generate_nanoid()}"
    missing_session_name = f"rust-contract-session-{generate_nanoid()}"

    workspace = models.Workspace(name=workspace_name)
    empty_session = models.Session(
        name=empty_session_name,
        workspace_name=workspace_name,
    )
    summary_session = models.Session(
        name=summary_session_name,
        workspace_name=workspace_name,
        internal_metadata={
            "summaries": {
                "honcho_chat_summary_short": {
                    "content": "Short summary content",
                    "message_id": 101,
                    "message_public_id": "msg_public_short",
                    "summary_type": "short",
                    "created_at": "2026-06-15T10:00:00+00:00",
                    "token_count": 11,
                },
                "honcho_chat_summary_long": {
                    "content": "Long summary content",
                    "message_id": 202,
                    "message_public_id": "msg_public_long",
                    "summary_type": "long",
                    "created_at": "2026-06-15T11:00:00+00:00",
                    "token_count": 22,
                },
            }
        },
    )
    db_session.add_all([workspace, empty_session, summary_session])
    await db_session.commit()

    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/sessions/{empty_session_name}/summaries",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/sessions/{missing_session_name}/summaries",
    )
    _compare(
        client,
        rust_api_url,
        "GET",
        f"/v3/workspaces/{workspace_name}/sessions/{summary_session_name}/summaries",
    )

    response = client.get(
        f"/v3/workspaces/{workspace_name}/sessions/{summary_session_name}/summaries"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["short_summary"]["message_id"] == "msg_public_short"
    assert body["long_summary"]["message_id"] == "msg_public_long"
    assert "message_public_id" not in body["short_summary"]
