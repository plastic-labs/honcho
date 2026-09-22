"""Shared fixtures for vector-store tests."""

from pathlib import Path

import pytest


@pytest.fixture
def embedded_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use single-process embedded storage without exposing it in Honcho settings."""
    chromadb = pytest.importorskip("chromadb")
    from src.vector_store.chroma import ChromaVectorStore

    monkeypatch.setattr(
        ChromaVectorStore,
        "_create_client",
        staticmethod(lambda: chromadb.PersistentClient(path=str(tmp_path))),
    )
