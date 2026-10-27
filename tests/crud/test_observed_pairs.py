"""Document queries over several (observer, observed) pairs."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.crud import document as document_module
from src.vector_store import VectorQueryResult


@pytest.fixture
def external_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """An external store whose namespaces each hold three scored hits."""
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

    def namespace(_kind: str, _workspace: str, observer: str, observed: str) -> str:
        return f"{observer}->{observed}"

    hits = {
        "a->a": [("a1", 0.10), ("a2", 0.40), ("a3", 0.70)],
        "b->b": [("b1", 0.05), ("b2", 0.30), ("b3", 0.90)],
    }

    async def query(ns: str, _embedding: list[float], **kwargs: Any):
        return [
            VectorQueryResult(id=doc_id, score=score)
            for doc_id, score in hits.get(ns, [])[: kwargs["top_k"]]
        ]

    store = MagicMock()
    store.get_vector_namespace = MagicMock(side_effect=namespace)
    store.query = AsyncMock(side_effect=query)
    monkeypatch.setattr(document_module, "get_external_vector_store", lambda: store)
    return store


@pytest.mark.asyncio
async def test_several_pairs_query_each_namespace_and_merge_by_distance(
    external_store: MagicMock,
):
    ids = await document_module.query_external_vector_document_ids(
        "ws",
        "a",
        "a",
        [0.0],
        top_k=4,
        pairs=[("a", "a"), ("b", "b")],
    )

    assert ids == ["b1", "a1", "b2", "a2"]
    queried = sorted(call.args[0] for call in external_store.query.await_args_list)
    assert queried == ["a->a", "b->b"]


@pytest.mark.asyncio
async def test_one_pair_keeps_the_store_order(external_store: MagicMock):
    ids = await document_module.query_external_vector_document_ids(
        "ws", "a", "a", [0.0], top_k=2
    )

    assert ids == ["a1", "a2"]
    assert external_store.query.await_count == 1


def test_one_pair_clause_is_two_equalities():
    [observer, observed] = document_module.observed_pairs_clause([("a", "b")])
    assert str(observer.compile(compile_kwargs={"literal_binds": True})).endswith(
        "observer = 'a'"
    )
    assert str(observed.compile(compile_kwargs={"literal_binds": True})).endswith(
        "observed = 'b'"
    )


def test_pairs_clause_needs_a_pair():
    with pytest.raises(ValueError):
        document_module.observed_pairs_clause([])
