"""Tests for startup zero-initialization of bounded-label metrics.

Asserts that:
- bounded-label counter children are materialized at 0 before any event,
- high-cardinality / impossible label combinations are deliberately NOT,
- per-process init doesn't materialize the other process's counters,
- the explicit registries stay in sync with the source of truth (drift guards).
"""
# region ai
# Reads use ``REGISTRY.get_sample_value`` (returns the value if a series exists,
# ``None`` if it does not) rather than ``counter.labels(...)``, because ``.labels``
# would itself materialize the child and destroy the presence/absence signal.
# endregion

from collections.abc import Iterator
from typing import cast
from uuid import uuid4

import pytest
from prometheus_client import REGISTRY

from src.config import REASONING_LEVELS, settings
from src.dreamer.specialists import BaseSpecialist
from src.telemetry.events import ALL_EVENT_TYPES, HIGH_VOLUME_EVENT_TYPES
from src.telemetry.events.base import BaseEvent
from src.telemetry.prometheus.metrics import (
    _DERIVER_TOKEN_COMBOS_BY_TASK,  # pyright: ignore[reportPrivateUsage]
    DeriverComponents,
    DeriverTaskTypes,
    DialecticComponents,
    TokenTypes,
    prometheus_metrics,
)
from src.utils.types import walk_subclasses


def unique_ns(tag: str) -> str:
    """A ``namespace`` label value no other test can have materialized under."""
    # region ai
    # Every assertion here reads the process-global ``REGISTRY``, which keeps a child
    # series for the rest of the session once anything materializes it. A shared
    # namespace (several other suites pin ``"test"``) would let another test's children
    # satisfy a presence assertion, or break an absence assertion, independently of
    # what the initializer under test actually did.
    # endregion
    return f"test_metric_zero_init_{tag}_{uuid4().hex[:8]}"


@pytest.fixture
def metrics_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Enable metrics under a namespace unique to the requesting test."""
    ns = unique_ns("enabled")
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", True)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", ns)
    yield ns


def sample(name: str, **labels: str) -> float | None:
    """Value of a series if it exists, else None. Never materializes it.

    Resolves the namespace from settings, so it always reads the unique one the
    active test pinned.
    """
    ns = cast(str, settings.METRICS.NAMESPACE)
    return REGISTRY.get_sample_value(name, {"namespace": ns, **labels})


# ---------------------------------------------------------------------------
# Drift guards (pure logic — no registry). Adding an event type / token component
# without updating the registry fails here, with a pointer to what to fix.
# ---------------------------------------------------------------------------


def test_all_event_types_registry_matches_subclasses():
    """ALL_EVENT_TYPES must equal every BaseEvent subclass's _event_type.

    If this fails you added/removed a BaseEvent subclass without updating
    ALL_EVENT_TYPES in src/telemetry/events/__init__.py — its Prometheus counter
    would not be zero-initialized. Update the registry.
    """
    discovered = {
        event_type
        for cls in walk_subclasses(BaseEvent)
        if (event_type := getattr(cls, "_event_type", None)) is not None
    }
    assert set(ALL_EVENT_TYPES) == discovered
    assert len(ALL_EVENT_TYPES) == len(set(ALL_EVENT_TYPES)), "duplicate event types"


def test_high_volume_registry_matches_subclasses():
    """HIGH_VOLUME_EVENT_TYPES must equal the high_volume-classed subclasses."""
    discovered = {
        event_type
        for cls in walk_subclasses(BaseEvent)
        if (event_type := getattr(cls, "_event_type", None)) is not None
        and getattr(cls, "_volume_class", None) == "high_volume"
    }
    assert set(HIGH_VOLUME_EVENT_TYPES) == discovered
    assert set(HIGH_VOLUME_EVENT_TYPES) <= set(ALL_EVENT_TYPES)


def test_deriver_token_combos_are_valid_and_complete():
    """Every combo uses real enum values; the union across tasks covers every
    DeriverComponent; and no task enumerates an impossible pair.

    Fails if a DeriverComponent/DeriverTaskType is added without deciding which
    task_type + token_type it pairs with in _DERIVER_TOKEN_COMBOS_BY_TASK.
    """
    valid_token_types = {t.value for t in TokenTypes}
    valid_components = {c.value for c in DeriverComponents}
    valid_task_types = {t.value for t in DeriverTaskTypes}

    assert set(_DERIVER_TOKEN_COMBOS_BY_TASK) == valid_task_types
    all_components: set[str] = set()
    for task_type, combos in _DERIVER_TOKEN_COMBOS_BY_TASK.items():
        assert task_type in valid_task_types
        for token_type, component in combos:
            assert token_type in valid_token_types
            assert component in valid_components
        # each task enumerates fewer than its cartesian product (no impossible pairs)
        assert len(combos) < len(valid_token_types) * len(valid_components)
        all_components.update(comp for _, comp in combos)

    # every component is reachable via some task
    assert all_components == valid_components
    # previous_summary is summary-only: ingestion must NOT enumerate it
    ingestion = _DERIVER_TOKEN_COMBOS_BY_TASK[DeriverTaskTypes.INGESTION.value]
    assert (
        TokenTypes.INPUT.value,
        DeriverComponents.PREVIOUS_SUMMARY.value,
    ) not in ingestion


_API_DERIVER_METRIC_GAUGES = (
    "deriver_outstanding_work_seconds",
    "deriver_queue_work_units_eligible",
    "deriver_queue_work_units_claimed",
    "deriver_queue_items_pending",
    "deriver_queue_oldest_pending_age_seconds",
    "dreams_due",
    "message_embeddings_pending_due",
)

_SHARED_DERIVER_METRIC_GAUGES = ("message_embeddings_pending",)


# ---------------------------------------------------------------------------
# API-process zero-init
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("metrics_enabled")
def test_api_init_materializes_event_type_children():
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    for event_type in ALL_EVENT_TYPES:
        assert sample("telemetry_events_emitted_total", type=event_type) is not None
    for event_type in HIGH_VOLUME_EVENT_TYPES:
        assert sample("telemetry_events_sampled_out_total", type=event_type) is not None


@pytest.mark.usefixtures("metrics_enabled")
def test_api_init_materializes_dialectic_and_embed():
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    for token_type in TokenTypes:
        for level in REASONING_LEVELS:
            assert (
                sample(
                    "dialectic_tokens_processed_total",
                    token_type=token_type.value,
                    component=DialecticComponents.TOTAL.value,
                    reasoning_level=level,
                )
                is not None
            )
    assert sample("embed_now_tasks_shed_total") is not None
    assert sample("embed_now_tasks_in_flight") == 0.0  # gauge, explicit .set(0)
    for gauge in (*_API_DERIVER_METRIC_GAUGES, *_SHARED_DERIVER_METRIC_GAUGES):
        assert sample(gauge) == 0.0, f"{gauge} was not zero-initialized"


@pytest.mark.usefixtures("metrics_enabled")
def test_sampled_out_excludes_ground_truth_event_types():
    """Ground-truth events can never be sampled out, so their sampled_out series
    must NOT be pre-created (they'd be permanently misleading zeros)."""
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    ground_truth = set(ALL_EVENT_TYPES) - set(HIGH_VOLUME_EVENT_TYPES)
    for event_type in ground_truth:
        assert sample("telemetry_events_sampled_out_total", type=event_type) is None


# ---------------------------------------------------------------------------
# Deriver-process zero-init
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("metrics_enabled")
def test_deriver_init_materializes_token_and_backlog():
    prometheus_metrics.initialize_bounded_metrics(instance_type="deriver")
    for task_type, combos in _DERIVER_TOKEN_COMBOS_BY_TASK.items():
        for token_type, component in combos:
            assert (
                sample(
                    "deriver_tokens_processed_total",
                    task_type=task_type,
                    token_type=token_type,
                    component=component,
                )
                is not None
            )
    # region ai
    # Specialist names are derived from the concrete BaseSpecialist subclasses here
    # too, rather than hardcoded: a hardcoded list would keep passing when a new
    # specialist is added (it only asserts presence), silently leaving it uncovered.
    # endregion
    specialist_names = {
        name
        for cls in walk_subclasses(BaseSpecialist)
        if (name := getattr(cls, "name", None)) is not None
    }
    assert {"deduction", "induction", "card_refresh"} <= specialist_names
    for specialist_name in specialist_names:
        assert (
            sample(
                "dreamer_tokens_processed_total",
                specialist_name=specialist_name,
                token_type=TokenTypes.INPUT.value,
            )
            is not None
        ), f"specialist {specialist_name!r} was not zero-initialized"
    assert sample("message_embeddings_pending") == 0.0  # gauge zero-init


@pytest.mark.usefixtures("metrics_enabled")
def test_deriver_init_omits_impossible_token_combos():
    """The cartesian product includes combos that never occur (e.g. output tokens
    with an input component). Those must not be materialized."""
    prometheus_metrics.initialize_bounded_metrics(instance_type="deriver")
    # output tokens never pair with an input component
    assert (
        sample(
            "deriver_tokens_processed_total",
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.OUTPUT.value,
            component=DeriverComponents.PROMPT.value,
        )
        is None
    )
    # previous_summary is summary-only — ingestion must not materialize it
    assert (
        sample(
            "deriver_tokens_processed_total",
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.INPUT.value,
            component=DeriverComponents.PREVIOUS_SUMMARY.value,
        )
        is None
    )
    # base specialist is abstract and never emits — must not be materialized
    assert (
        sample(
            "dreamer_tokens_processed_total",
            specialist_name="base",
            token_type=TokenTypes.INPUT.value,
        )
        is None
    )


# ---------------------------------------------------------------------------
# High-cardinality counters are left open, and per-process isolation holds
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("metrics_enabled")
def test_high_cardinality_counters_not_materialized():
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    prometheus_metrics.initialize_bounded_metrics(instance_type="deriver")
    # no endpoint/workspace_name series fabricated
    assert (
        sample(
            "api_requests_total",
            method="GET",
            endpoint="/v3/does-not-exist",
            status_code="200",
        )
        is None
    )
    assert sample("messages_created_total", workspace_name="nope_ws") is None


@pytest.mark.usefixtures("metrics_enabled")
def test_api_init_does_not_touch_deriver_counters():
    """api-only init must not materialize or change a deriver-only counter.

    Delta-based (before == after) so it's robust to prior tests having
    materialized the series.
    """
    labels = dict(
        task_type=DeriverTaskTypes.INGESTION.value,
        token_type=TokenTypes.INPUT.value,
        component=DeriverComponents.PROMPT.value,
    )
    before = sample("deriver_tokens_processed_total", **labels)
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    after = sample("deriver_tokens_processed_total", **labels)
    assert before == after


@pytest.mark.usefixtures("metrics_enabled")
def test_deriver_init_does_not_touch_api_counters():
    """The inverse: deriver-only init must not materialize an API-only counter.

    Without this, a deriver-startup regression could silently fabricate API
    series (permanently-0 dialectic tokens on a process that never serves chat).
    """
    labels = dict(
        token_type=TokenTypes.INPUT.value,
        component=DialecticComponents.TOTAL.value,
        reasoning_level=REASONING_LEVELS[0],
    )
    before = sample("dialectic_tokens_processed_total", **labels)
    prometheus_metrics.initialize_bounded_metrics(instance_type="deriver")
    after = sample("dialectic_tokens_processed_total", **labels)
    assert before == after
    # the API-process embed_now counters are equally off-limits
    assert sample("embed_now_tasks_shed_total") is None
    assert sample("embed_now_tasks_in_flight") is None
    # so are the deriver-work gauges: the deriver never measures its own backlog
    for gauge in _API_DERIVER_METRIC_GAUGES:
        assert sample(gauge) is None, f"{gauge} must be API-only"


# ---------------------------------------------------------------------------
# telemetry_events_dropped: per-emitter child materialization
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("metrics_enabled")
def test_dropped_counter_children_materialized():
    prometheus_metrics.initialize_telemetry_dropped_metrics(
        reasons=["buffer_full", "send_failed"]
    )
    assert sample("telemetry_events_dropped_total", reason="buffer_full") is not None
    assert sample("telemetry_events_dropped_total", reason="send_failed") is not None


def test_dropped_counter_init_noop_when_metrics_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """The per-emitter initializer must no-op when metrics are disabled."""
    # region ai
    # The enabled/disabled pair above and below this line exists for
    # ``initialize_bounded_metrics`` (see ``test_init_noop_when_metrics_disabled``);
    # without this test the sibling initializer had only the enabled half, so its
    # ``METRICS.ENABLED`` guard could be deleted with the suite staying green. The
    # unique namespace is what makes the absence assertion mean anything — the enabled
    # test above materializes these same two reason values under a different one.
    # endregion
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", False)
    monkeypatch.setattr(
        "src.config.settings.METRICS.NAMESPACE", unique_ns("dropped_disabled")
    )
    prometheus_metrics.initialize_telemetry_dropped_metrics(
        reasons=["buffer_full", "send_failed"]
    )
    assert sample("telemetry_events_dropped_total", reason="buffer_full") is None
    assert sample("telemetry_events_dropped_total", reason="send_failed") is None


def test_init_noop_when_metrics_disabled(monkeypatch: pytest.MonkeyPatch):
    """With metrics disabled, init must not fabricate series for a fresh label."""
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", False)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", unique_ns("disabled"))
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    assert sample("telemetry_events_emitted_total", type="message.created") is None
