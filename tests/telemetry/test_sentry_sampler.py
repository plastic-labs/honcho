"""Tests for the Sentry traces sampler.

The sampler must drop high-volume infra/scrape transactions (health checks,
Prometheus scrapes, OpenAPI schema, docs) while sampling real traffic at the
configured rate. These endpoints otherwise dominate transaction + profiling
volume and drown out useful traces.
"""

import pytest

from src.config import settings
from src.telemetry.sentry import traces_sampler


@pytest.mark.parametrize(
    "path",
    ["/metrics", "/health", "/openapi.json", "/docs", "/redoc"],
)
def test_infra_paths_are_dropped(path: str) -> None:
    """ASGI requests to infra/scrape paths get a 0.0 sample rate."""
    assert traces_sampler({"asgi_scope": {"path": path}}) == 0.0


@pytest.mark.parametrize(
    "name",
    [
        "src.telemetry.prometheus.metrics.metrics_endpoint",
        "src.prometheus.metrics",
        "fastapi.applications.FastAPI.setup.<locals>.openapi",
    ],
)
def test_infra_transaction_names_are_dropped(name: str) -> None:
    """Transactions without an ASGI path still drop by their endpoint name."""
    assert traces_sampler({"transaction_context": {"name": name}}) == 0.0


def test_real_route_uses_default_rate() -> None:
    """A normal API route is sampled at the configured default rate."""
    ctx = {
        "asgi_scope": {"path": "/v3/peers/alice/chat"},
        "transaction_context": {"name": "src.routers.peers.chat"},
    }
    assert traces_sampler(ctx) == settings.SENTRY.TRACES_SAMPLE_RATE


def test_parent_sampling_decision_is_respected() -> None:
    """When continuing a distributed trace, inherit the upstream decision."""
    assert traces_sampler({"parent_sampled": True}) == 1.0
    assert traces_sampler({"parent_sampled": False}) == 0.0


def test_infra_path_overrides_parent_decision() -> None:
    """Infra paths are dropped even if an upstream trace was sampled in."""
    ctx = {"asgi_scope": {"path": "/metrics"}, "parent_sampled": True}
    assert traces_sampler(ctx) == 0.0


def test_empty_context_falls_back_to_default_rate() -> None:
    """A context with no scope, name, or parent uses the default rate."""
    assert traces_sampler({}) == settings.SENTRY.TRACES_SAMPLE_RATE
