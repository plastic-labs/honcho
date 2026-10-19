"""The optional Chroma SDK must be checked when selecting its backend."""

import sys

import pytest

from src.config import settings
from src.vector_store import (
    _create_store_by_type,  # pyright: ignore[reportPrivateUsage]
    get_external_vector_store,
)


def test_chroma_factory_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "src.vector_store.chroma", raising=False)
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with pytest.raises(ValueError, match="uv sync --extra chromadb") as error:
        _create_store_by_type("chromadb")
    assert isinstance(error.value.__cause__, ImportError)


def test_pgvector_does_not_require_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "pgvector")
    get_external_vector_store.cache_clear()
    try:
        assert get_external_vector_store() is None
    finally:
        get_external_vector_store.cache_clear()
