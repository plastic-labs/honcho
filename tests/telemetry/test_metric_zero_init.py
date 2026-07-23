"""Tests for startup zero-initialization of bounded-label metrics.

Backfills the coverage PR #927 shipped without, and covers the generalization:
- bounded-label counter children are materialized at 0 before any event,
- high-cardinality / impossible label combinations are deliberately NOT,
- per-process init doesn't materialize the other process's counters,
- the explicit registries stay in sync with the source of truth (drift guards).

Reads use ``REGISTRY.get_sample_value`` (returns the value if a series exists,
``None`` if it does not) rather than ``counter.labels(...)``, because ``.labels``
would itself materialize the child and destroy the presence/absence signal.
"""

from collections.abc import Iterator

import pytest
from prometheus_client import REGISTRY

from src.telemetry.events import ALL_EVENT_TYPES, HIGH_VOLUME_EVENT_TYPES
from src.telemetry.events.base import BaseEvent
from src.telemetry.prometheus.metrics import (
    _DERIVER_TOKEN_COMBOS_BY_TASK,  # pyright: ignore[reportPrivateUsage]
    REASONING_LEVELS,
    DeriverComponents,
    DeriverTaskTypes,
    DialecticComponents,
    TokenTypes,
    prometheus_metrics,
)

NS = "test"


@pytest.fixture
def metrics_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", True)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", NS)
    yield


def sample(name: str, **labels: str) -> float | None:
    """Value of a series if it exists, else None. Never materializes it."""
    return REGISTRY.get_sample_value(name, {"namespace": NS, **labels})


# ---------------------------------------------------------------------------
# Drift guards (pure logic — no registry). These are the repo-visible collaborator
# notes: adding an event type / token component without updating the registry
# fails here with a pointer to what to fix.
# ---------------------------------------------------------------------------


def _walk_event_subclasses(cls: type[BaseEvent]) -> Iterator[type[BaseEvent]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_event_subclasses(sub)


def test_all_event_types_registry_matches_subclasses():
    """ALL_EVENT_TYPES must equal every BaseEvent subclass's _event_type.

    If this fails you added/removed a BaseEvent subclass without updating
    ALL_EVENT_TYPES in src/telemetry/events/__init__.py — its Prometheus counter
    would not be zero-initialized. Update the registry.
    """
    discovered = {
        event_type
        for cls in _walk_event_subclasses(BaseEvent)
        if (event_type := getattr(cls, "_event_type", None)) is not None
    }
    assert set(ALL_EVENT_TYPES) == discovered
    assert len(ALL_EVENT_TYPES) == len(set(ALL_EVENT_TYPES)), "duplicate event types"


def test_high_volume_registry_matches_subclasses():
    """HIGH_VOLUME_EVENT_TYPES must equal the high_volume-classed subclasses."""
    discovered = {
        event_type
        for cls in _walk_event_subclasses(BaseEvent)
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
    # dreamer specialists are derived from the concrete BaseSpecialist subclasses
    for specialist_name in ("deduction", "induction"):
        assert (
            sample(
                "dreamer_tokens_processed_total",
                specialist_name=specialist_name,
                token_type=TokenTypes.INPUT.value,
            )
            is not None
        )
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


# ---------------------------------------------------------------------------
# Dropped-counter backfill (#927 shipped without a test)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("metrics_enabled")
def test_dropped_counter_children_materialized():
    prometheus_metrics.initialize_telemetry_dropped_metrics(
        reasons=["buffer_full", "send_failed"]
    )
    assert sample("telemetry_events_dropped_total", reason="buffer_full") is not None
    assert sample("telemetry_events_dropped_total", reason="send_failed") is not None


def test_init_noop_when_metrics_disabled(monkeypatch: pytest.MonkeyPatch):
    """With metrics disabled, init must not fabricate series for a fresh label."""
    monkeypatch.setattr("src.config.settings.METRICS.ENABLED", False)
    monkeypatch.setattr("src.config.settings.METRICS.NAMESPACE", "disabled_ns")
    prometheus_metrics.initialize_bounded_metrics(instance_type="api")
    assert (
        REGISTRY.get_sample_value(
            "telemetry_events_emitted_total",
            {"namespace": "disabled_ns", "type": "message.created"},
        )
        is None
    )
