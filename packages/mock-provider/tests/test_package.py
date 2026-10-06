"""Standalone provider behavior without importing the server or its fixtures."""

from __future__ import annotations

import subprocess
import sys

from fastapi.testclient import TestClient

from honcho_mock_provider.main import create_app


def test_factory_and_deterministic_wire_responses():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok", "provider": "mock"}
        body = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        first = client.post("/v1/chat/completions", json=body)
        assert first.status_code == 200
        assert first.content == client.post("/chat/completions", json=body).content
        embeddings = client.post(
            "/v1/embeddings",
            json={"input": "hello", "dimensions": 8, "encoding_format": "float"},
        ).json()
        assert len(embeddings["data"][0]["embedding"]) == 8
        assert client.post("/unsupported", json={}).status_code == 405


def test_no_server_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from honcho_mock_provider.main import create_app; create_app(); assert not any(m == 'src' or m.startswith(('src.', 'sqlalchemy', 'honcho.', 'tests.')) for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
