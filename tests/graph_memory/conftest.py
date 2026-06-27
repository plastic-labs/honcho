"""Fixtures for graph-memory backend tests.

Re-exports shared fixtures from ``tests/fixtures/graph_memory_fixtures`` so
that recall, CRUD, and integration tests in this package can use them.
"""

from __future__ import annotations

from tests.fixtures.graph_memory_fixtures import (  # noqa: F401
    TOPIC_INDICES,
    TOPIC_OBSERVATIONS,
    VECTOR_DIMENSIONS,
    _make_mock_embedding_client,
    _rewrite_db_host_for_graph_memory_tests,
    clean_graph_memory_queue_tables,
    controlled_embedding_client,
    force_promote,
    graph_memory_setup,
    patch_embedding_client_for_topic,
    query_vector_for_topic,
    topic_vector,
)
