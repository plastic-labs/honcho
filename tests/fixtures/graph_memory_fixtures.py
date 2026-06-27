"""Shared fixtures for graph-memory backend tests.

Tests exercise the real pipeline against a per-test PostgreSQL database and a
fakeredis cache.  Embeddings are controlled deterministically so that topics
form tight cosine clusters, making semantic-similarity behaviour observable
without calling a live embedding provider.
"""

from __future__ import annotations

import math
import random
from collections.abc import AsyncGenerator, Callable
from typing import Any

import pytest
import pytest_asyncio
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _rewrite_db_host_for_graph_memory_tests() -> AsyncGenerator[None, None]:
    """Use the host-forwarded database if the compose service name is unreachable.

    The project's .env points at the ``database`` service name, which is only
    resolvable inside the compose network.  When tests run directly on the host,
    PostgreSQL is forwarded to localhost:5432.  This fixture makes the suite
    runnable in both contexts without editing .env.
    """
    import socket

    original_uri = settings.DB.CONNECTION_URI
    try:
        socket.gethostbyname("database")
    except OSError:
        if "database:" in original_uri:
            settings.DB.CONNECTION_URI = original_uri.replace("database:", "localhost:")
    try:
        yield
    finally:
        settings.DB.CONNECTION_URI = original_uri


# Topic index used to build controlled embeddings.  Each topic gets a unique
# sparse coordinate block so cosine similarity is high within a topic and
# near-zero across topics.
TOPIC_INDICES = {
    "llminal": 0,
    "honcho": 1,
    "user_profile": 2,
    "agentc_process": 3,
}

VECTOR_DIMENSIONS: int = settings.EMBEDDING.VECTOR_DIMENSIONS
# Coordinate block per topic.  Kept small so within-topic cosine is very high
# and cross-topic cosine is near zero.
BLOCK_SIZE = 24


def topic_vector(topic: str, seed: int = 0, dim: int = VECTOR_DIMENSIONS) -> list[float]:
    """Return a unit-length embedding that clusters by topic.

    Each topic is assigned a distinct block of coordinates so same-topic
    vectors have high cosine similarity and cross-topic vectors are nearly
    orthogonal.  Per-seed noise within the block differentiates observations
    within a topic while keeping the cluster tight.
    """
    idx = TOPIC_INDICES[topic]
    rng = random.Random(seed)
    vec = [0.0] * dim

    start = idx * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, dim)
    for i in range(start, end):
        vec[i] = 1.0 + rng.uniform(-0.05, 0.05)

    # Add tiny noise off the topic block so observations are not identical.
    for i in range(dim):
        if i < start or i >= end:
            vec[i] = rng.uniform(-0.05, 0.05)

    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def query_vector_for_topic(topic: str, dim: int = VECTOR_DIMENSIONS) -> list[float]:
    """Return a query embedding near the centroid of a topic cluster.

    The query vector points into the same coordinate block as the document
    vectors for the topic, so cosine distance cleanly separates topics.
    """
    idx = TOPIC_INDICES[topic]
    vec = [0.0] * dim
    start = idx * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, dim)
    for i in range(start, end):
        vec[i] = 1.0
    return vec


# A set of observations covering four distinct topics.  The promotion worker
# should create edges within topics (because their embeddings are similar) and
# avoid edges across topics.
TOPIC_OBSERVATIONS: dict[str, list[str]] = {
    "llminal": [
        "We decided the LLMinal protocol uses L1 encoding for clarity-critical messages.",
        "Dogfooding results show LLMinal saves 33 percent on clarity-critical messages.",
        "The LLMinal wire format prefers brevity over human readability.",
        "LLMinal dogfooding revealed token-efficiency wins in multi-turn sessions.",
    ],
    "honcho": [
        "The Honcho graph memory backend builds edges between semantically related observations.",
        "Honcho ngram bridge converts short observations into durable long-term memory.",
        "The promotion worker moves L1 observations into L2 graph memory.",
        "Graph memory confidence decays over a 30-day half-life.",
    ],
    "user_profile": [
        "The user is located in Lyons, Colorado, near Boulder.",
        "The user is seeking a remote software leadership role paying at least 220k per year.",
        "The user prefers Slack for continuous monitoring over Signal or email.",
        "The user's background spans networking infrastructure, RF systems, and AI.",
    ],
    "agentc_process": [
        "AgentC requires adversarial review before merging security-sensitive changes.",
        "The AgentC definition of done is verified running, not just merged.",
        "Simulation-first is mandatory for AgentC concurrency and recovery work.",
        "AgentC uses per-commit anti-pattern scanning on new Python files.",
    ],
}


def _make_mock_embedding_client() -> Any:
    """Build a tiny async embedding client that returns deterministic vectors."""

    class MockEmbeddingClient:
        async def embed(self, query: str) -> list[float]:
            topic = "llminal"
            q = query.lower()
            if "honcho" in q or "graph memory" in q or "promotion" in q:
                topic = "honcho"
            elif "lyons" in q or "job" in q or "user" in q or "slacks" in q:
                topic = "user_profile"
            elif "agentc" in q or "adversarial" in q or "anti-pattern" in q:
                topic = "agentc_process"
            return query_vector_for_topic(topic)

        async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
            return [self._vector_for_text(t) for t in texts]

        def _vector_for_text(self, text: str) -> list[float]:
            text_lower = text.lower()
            for topic, obs_list in TOPIC_OBSERVATIONS.items():
                for o in obs_list:
                    if o.lower() in text_lower or text_lower in o.lower():
                        return topic_vector(topic, seed=hash(text) % 10000)
            return topic_vector("llminal", seed=hash(text) % 10000)

    return MockEmbeddingClient()


@pytest_asyncio.fixture(scope="function")
async def graph_memory_setup(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> AsyncGenerator[dict[str, Any], None]:
    """Create a workspace, peers, collection, session, and topic observations.

    Returns a dict with:
      - workspace, observer, observed, session
      - collection_name (observer/observed pair)
      - all_docs: list of created Document rows
      - docs_by_topic: topic -> list of Document rows
      - ids_by_topic: topic -> list of ids
    """
    workspace, observer_peer = sample_data

    observed_peer = models.Peer(
        name=str(generate_nanoid()),
        workspace_name=workspace.name,
    )
    db_session.add(observed_peer)
    await db_session.flush()

    collection = models.Collection(
        workspace_name=workspace.name,
        observer=observer_peer.name,
        observed=observed_peer.name,
    )
    db_session.add(collection)
    await db_session.flush()

    session = models.Session(
        name=str(generate_nanoid()),
        workspace_name=workspace.name,
    )
    db_session.add(session)
    await db_session.flush()

    all_docs: list[models.Document] = []
    docs_by_topic: dict[str, list[models.Document]] = {}

    for topic, contents in TOPIC_OBSERVATIONS.items():
        topic_docs: list[models.Document] = []
        for i, content in enumerate(contents):
            doc = models.Document(
                workspace_name=workspace.name,
                observer=observer_peer.name,
                observed=observed_peer.name,
                content=content,
                level="explicit",
                times_derived=1,
                internal_metadata={"topic": topic},
                session_name=session.name,
                embedding=topic_vector(topic, seed=i),
            )
            db_session.add(doc)
            topic_docs.append(doc)
            all_docs.append(doc)
        docs_by_topic[topic] = topic_docs

    await db_session.commit()
    for doc in all_docs:
        await db_session.refresh(doc)

    yield {
        "workspace": workspace,
        "observer": observer_peer,
        "observed": observed_peer,
        "collection_name": f"{observer_peer.name}/{observed_peer.name}",
        "session": session,
        "all_docs": all_docs,
        "docs_by_topic": docs_by_topic,
        "ids_by_topic": {t: [d.id for d in docs] for t, docs in docs_by_topic.items()},
    }


@pytest.fixture
def controlled_embedding_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the graph-memory router's embedding client to deterministic vectors."""
    mock_client = _make_mock_embedding_client()
    monkeypatch.setattr(
        "src.routers.graph_memory.embedding_client",
        mock_client,
        raising=False,
    )
    return mock_client


@pytest.fixture
def force_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the promotion worker so every observation passes the LLM test."""

    async def _always_promote(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True

    monkeypatch.setattr(
        "src.deriver.promotion._llm_promotion_test",
        _always_promote,
    )


@pytest_asyncio.fixture(scope="function", autouse=True)
async def clean_graph_memory_queue_tables(db_session: AsyncSession) -> AsyncGenerator[None, None]:
    """Remove queue items and active queue sessions before each graph-memory test."""
    from sqlalchemy import delete
    await db_session.execute(delete(models.ActiveQueueSession))
    await db_session.execute(delete(models.QueueItem))
    await db_session.commit()
    yield


@pytest.fixture
def patch_embedding_client_for_topic(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Any]:
    """Return a helper that patches the router embedding client for a query topic."""

    def _patch(topic: str) -> Any:
        class TopicClient:
            async def embed(self, query: str) -> list[float]:
                return query_vector_for_topic(topic)

        monkeypatch.setattr(
            "src.routers.graph_memory.embedding_client",
            TopicClient(),
            raising=False,
        )
        return TopicClient()

    return _patch
