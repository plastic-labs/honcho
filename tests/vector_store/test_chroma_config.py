"""Deployment mode validation does not require the optional Chroma SDK."""

import pytest
from pydantic import ValidationError

from src.config import VectorStoreSettings


def test_chroma_defaults_to_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTOR_STORE_CHROMA_CLIENT_MODE", raising=False)
    assert VectorStoreSettings(TYPE="chromadb").CHROMA_CLIENT_MODE == "http"


def test_chroma_rejects_persistent_mode_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE_CHROMA_CLIENT_MODE", "persistent")
    with pytest.raises(
        ValidationError, match="unsafe across Honcho's multiple processes"
    ):
        VectorStoreSettings(TYPE="chromadb")


def test_chroma_cloud_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECTOR_STORE_CHROMA_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="CHROMA_API_KEY must be set"):
        VectorStoreSettings(TYPE="chromadb", CHROMA_CLIENT_MODE="cloud")


def test_chroma_cloud_accepts_api_key() -> None:
    config = VectorStoreSettings(
        TYPE="chromadb", CHROMA_CLIENT_MODE="cloud", CHROMA_API_KEY="test-key"
    )
    assert config.CHROMA_CLIENT_MODE == "cloud"
