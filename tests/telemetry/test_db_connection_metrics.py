"""Tests for the physical-DB-connection metrics.

``db_connections_open`` (gauge) and ``db_connections_established`` (counter) are
driven by SQLAlchemy connection-lifecycle events rather than the pool object, so
they report real connections under EVERY pool class — including ``NullPool``, whose
pool keeps no records for the scrape-time ``db_pool_connections`` collector to read.

Asserts two properties:
- zero-init — resolving a labeled child materializes it at 0, so an absent series
  means a broken scrape rather than "no connections";
- ``DBConnectionTracker`` semantics — increment once per connect, decrement at most
  once per connection (marker-guarded, so it can't leak upward or go negative), and
  the establishment counter is monotonic (closes never decrement it).
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY

from src.db import DBConnectionTracker
from src.telemetry.prometheus.metrics import (
    db_connections_established_counter,
    db_connections_open_gauge,
)


@pytest.fixture
def ns(monkeypatch: pytest.MonkeyPatch) -> str:
    """Enable metrics under a namespace unique to this test.

    The process-global REGISTRY keeps a materialized child for the rest of the
    session, so a shared namespace would let one test satisfy another's
    presence/absence assertions independently of the code under test.
    """
    namespace = f"test_db_conn_{uuid4().hex[:8]}"
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", True)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", namespace)
    return namespace


def sample(name: str, namespace: str, **labels: str) -> float | None:
    """Value of a series if it exists, else None. Never materializes it."""
    return REGISTRY.get_sample_value(name, {"namespace": namespace, **labels})


def test_connection_children_zero_init(ns: str) -> None:
    """Resolving the labeled children materializes both series at 0."""
    db_connections_open_gauge.labels(instance_type="api")
    db_connections_established_counter.labels(instance_type="api")

    assert sample("db_connections_open", ns, instance_type="api") == 0.0
    # prometheus_client appends _total to counter names
    assert sample("db_connections_established_total", ns, instance_type="api") == 0.0


def test_tracker_inc_dec_and_counter_monotonic(ns: str) -> None:
    """connect increments both metrics; close decrements only the gauge."""
    open_child = db_connections_open_gauge.labels(instance_type="api")
    established_child = db_connections_established_counter.labels(instance_type="api")
    tracker = DBConnectionTracker(open_child, established_child)

    rec1, rec2 = SimpleNamespace(info={}), SimpleNamespace(info={})
    tracker.on_connect(None, rec1)
    tracker.on_connect(None, rec2)
    assert sample("db_connections_open", ns, instance_type="api") == 2.0
    assert sample("db_connections_established_total", ns, instance_type="api") == 2.0

    tracker.on_close(None, rec1)
    tracker.on_close(None, rec2)
    assert sample("db_connections_open", ns, instance_type="api") == 0.0
    # the counter is monotonic: closes never decrement it
    assert sample("db_connections_established_total", ns, instance_type="api") == 2.0


def test_marker_prevents_double_dec_and_negative(ns: str) -> None:
    """The ConnectionRecord marker bounds each connection to one dec."""
    open_child = db_connections_open_gauge.labels(instance_type="api")
    established_child = db_connections_established_counter.labels(instance_type="api")
    tracker = DBConnectionTracker(open_child, established_child)

    # a close with no matching connect must not drive the gauge negative
    tracker.on_close(None, SimpleNamespace(info={}))
    assert sample("db_connections_open", ns, instance_type="api") == 0.0

    # connect, then close AND invalidate on the same record (both fire during
    # invalidation cleanup): the marker ensures exactly one decrement. The third
    # positional arg is invalidate's exception, absorbed by on_close's *_.
    rec = SimpleNamespace(info={})
    tracker.on_connect(None, rec)
    tracker.on_close(None, rec)
    tracker.on_close(None, rec, ValueError("invalidated"))
    assert sample("db_connections_open", ns, instance_type="api") == 0.0
