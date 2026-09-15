# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests pinning tenant identity at honcho's telemetry chokepoints.

Covers the CloudEvents `tenantid` extension attribute (the emitter), the
`TenantScopedCounter` Prometheus label wrapper, and the per-request Sentry
`tenant_id` tag -- each read from the ambient `tenant_context` only under
`MULTI_TENANT`, never threaded through an event class or a call site.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
import sentry_sdk
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from prometheus_client import REGISTRY

from src.config import settings
from src.db import tenant_context
from src.security import JWTParams, create_jwt, require_auth
from src.telemetry.emitter import TelemetryEmitter
from src.telemetry.events.api import MessageCreatedEvent
from src.telemetry.events.reconciliation import SyncVectorsCompletedEvent
from src.telemetry.prometheus.metrics import prometheus_metrics
from src.telemetry.tenant import TENANTLESS_CATEGORIES, current_tenant_id


def unique_ns(tag: str) -> str:
    """A `namespace` label value no other test can have materialized under.

    Mirrors `tests/telemetry/test_metric_zero_init.py`'s convention: the
    prometheus_client REGISTRY is process-global and keeps every child series for
    the rest of the session, so presence/absence assertions need a namespace no
    other test's writes can satisfy or contaminate.
    """
    return f"test_tenant_identity_{tag}_{uuid4().hex[:8]}"


@pytest.fixture
def bound_tenant() -> Iterator[None]:
    """Bind `tenant_context` to "acme" for the test body; always reset."""
    token = tenant_context.set("acme")
    try:
        yield
    finally:
        tenant_context.reset(token)


def _message_created_event(**overrides: object) -> MessageCreatedEvent:
    """A minimal, non-reconciliation ("api" category) event."""
    fields: dict[str, object] = {
        "workspace_name": "ws",
        "session_name": "sess",
        "message_count": 1,
        "total_tokens": 10,
        "last_message_id": "msg_1",
    }
    fields.update(overrides)
    return MessageCreatedEvent(**fields)  # pyright: ignore[reportArgumentType]


def _sync_vectors_event(**overrides: object) -> SyncVectorsCompletedEvent:
    """A tenant-less-by-construction ("reconciliation" category) event."""
    fields: dict[str, object] = {"total_duration_ms": 1.0}
    fields.update(overrides)
    return SyncVectorsCompletedEvent(**fields)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# CloudEvents `tenantid` extension attribute (src/telemetry/emitter.py)
# ---------------------------------------------------------------------------


class TestEmitterTenantIdentity:
    @pytest.mark.usefixtures("bound_tenant")
    def test_emit_stamps_tenantid_when_flag_on_and_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-on with a bound tenant: the CloudEvent carries `tenantid` equal to
        the bound value; `source` keeps naming the instance/category (never the
        tenant); the body is the ordinary event payload -- tenant identity rides
        the envelope only, never `data`."""
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        monkeypatch.setattr(settings.TELEMETRY, "NAMESPACE", "test_ns")
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")

        emitter.emit(_message_created_event())

        ce = emitter._buffer[-1]
        attrs = ce.get_attributes()
        assert attrs["tenantid"] == "acme"
        assert attrs["source"] == "/honcho/test_ns/api"
        assert "tenantid" not in ce.data
        assert ce.data["workspace_name"] == "ws"
        assert ce.data["last_message_id"] == "msg_1"

    def test_emit_records_untenanted_counter_when_flag_on_and_unbound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-on with nothing bound, for a non-reconciliation event: no
        `tenantid` attribute, and `telemetry_events_untenanted_total{type=...}`
        increments by exactly 1 -- an emit site running outside its bind scope
        must be visible, not silently swallowed."""
        assert tenant_context.get() is None
        ns = unique_ns("untenanted")
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        monkeypatch.setattr(settings.TELEMETRY, "NAMESPACE", ns)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", ns)
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")
        labels = {"namespace": ns, "type": "message.created"}
        before = REGISTRY.get_sample_value("telemetry_events_untenanted_total", labels)

        emitter.emit(_message_created_event())

        after = REGISTRY.get_sample_value("telemetry_events_untenanted_total", labels)
        ce = emitter._buffer[-1]
        assert "tenantid" not in ce.get_attributes()
        assert (before or 0.0) + 1.0 == after

    def test_emit_reconciliation_event_exempt_from_untenanted_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-on with nothing bound, for a reconciliation event: still no
        `tenantid` attribute, but the untenanted counter is untouched -- these
        events are tenant-less by construction, not an emit-site bug, and must
        never be counted as one."""
        ns = unique_ns("reconciliation")
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        monkeypatch.setattr(settings.TELEMETRY, "NAMESPACE", ns)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", ns)
        event = _sync_vectors_event()
        assert event.category() in TENANTLESS_CATEGORIES
        emitter = TelemetryEmitter(endpoint="http://test:8001/events")
        labels = {"namespace": ns, "type": event.event_type()}
        before = REGISTRY.get_sample_value("telemetry_events_untenanted_total", labels)

        emitter.emit(event)

        after = REGISTRY.get_sample_value("telemetry_events_untenanted_total", labels)
        ce = emitter._buffer[-1]
        assert "tenantid" not in ce.get_attributes()
        assert before == after  # untouched, not merely un-incremented (both None)

    def test_flag_off_attribute_set_is_identical_bound_or_unbound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag-off byte-identity receipt: a tenant bound on the ContextVar (as a
        control plane that mints claims for every tenant might do, even against a
        dedicated instance) changes nothing. The CloudEvent attribute key set and
        `source` are identical whether or not something is bound, and equal
        exactly today's five CloudEvents-required keys plus the library's own
        `specversion`."""
        monkeypatch.setattr(settings, "MULTI_TENANT", False)
        monkeypatch.setattr(settings.TELEMETRY, "NAMESPACE", "test_ns")
        expected_keys = {"specversion", "id", "source", "type", "time", "dataschema"}

        unbound_emitter = TelemetryEmitter(endpoint="http://test:8001/events")
        unbound_emitter.emit(_message_created_event())
        unbound_attrs = unbound_emitter._buffer[-1].get_attributes()

        token = tenant_context.set("acme")
        try:
            bound_emitter = TelemetryEmitter(endpoint="http://test:8001/events")
            bound_emitter.emit(_message_created_event())
            bound_attrs = bound_emitter._buffer[-1].get_attributes()
        finally:
            tenant_context.reset(token)

        assert set(unbound_attrs.keys()) == expected_keys
        assert set(bound_attrs.keys()) == expected_keys
        assert "tenantid" not in bound_attrs
        assert bound_attrs["source"] == unbound_attrs["source"] == "/honcho/test_ns/api"


# ---------------------------------------------------------------------------
# TenantScopedCounter (src/telemetry/prometheus/metrics.py)
# ---------------------------------------------------------------------------


class TestTenantScopedCounter:
    @pytest.mark.usefixtures("bound_tenant")
    def test_labels_carry_bound_tenant_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-on with a bound tenant: the recorded series carries tenant_id
        equal to the bound value, beside namespace and the metric's own labels."""
        ns = unique_ns("counter_on")
        monkeypatch.setattr(settings.METRICS, "ENABLED", True)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", ns)
        monkeypatch.setattr(settings, "MULTI_TENANT", True)

        prometheus_metrics.record_messages_created(count=1, workspace_name="w")

        value = REGISTRY.get_sample_value(
            "messages_created_total",
            {"namespace": ns, "tenant_id": "acme", "workspace_name": "w"},
        )
        assert value == 1.0

    @pytest.mark.usefixtures("bound_tenant")
    def test_labels_are_empty_when_flag_off_even_if_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-off with a (stray) bound tenant: the series carries tenant_id=""
        (Prometheus/VictoriaMetrics treat that as absent), and no tenant_id="acme"
        series is created -- the bound value never leaks into the label flag-off."""
        ns = unique_ns("counter_off")
        monkeypatch.setattr(settings.METRICS, "ENABLED", True)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", ns)
        monkeypatch.setattr(settings, "MULTI_TENANT", False)

        prometheus_metrics.record_messages_created(count=1, workspace_name="w")

        empty_series = REGISTRY.get_sample_value(
            "messages_created_total",
            {"namespace": ns, "tenant_id": "", "workspace_name": "w"},
        )
        leaked_series = REGISTRY.get_sample_value(
            "messages_created_total",
            {"namespace": ns, "tenant_id": "acme", "workspace_name": "w"},
        )
        assert empty_series == 1.0
        assert leaked_series is None

    @pytest.mark.usefixtures("bound_tenant")
    def test_deriver_tokens_recorder_is_also_tenant_scoped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One representative non-API-only recorder (the deriver's token counter)
        confirms the wrapper applies uniformly across the six counters, not just
        `messages_created`."""
        ns = unique_ns("deriver_tokens")
        monkeypatch.setattr(settings.METRICS, "ENABLED", True)
        monkeypatch.setattr(settings.METRICS, "NAMESPACE", ns)
        monkeypatch.setattr(settings, "MULTI_TENANT", True)

        prometheus_metrics.record_deriver_tokens(
            count=5, task_type="ingestion", token_type="input", component="prompt"
        )

        value = REGISTRY.get_sample_value(
            "deriver_tokens_processed_total",
            {
                "namespace": ns,
                "tenant_id": "acme",
                "task_type": "ingestion",
                "token_type": "input",
                "component": "prompt",
            },
        )
        assert value == 5.0


# ---------------------------------------------------------------------------
# Sentry `tenant_id` tag on the API request scope (src/security.py)
# ---------------------------------------------------------------------------


def _acme_credentials() -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=create_jwt(JWTParams(tn="acme", w="ws-a"))
    )


def _bare_request() -> Request:
    return Request(
        {"type": "http", "query_string": b"", "path_params": {}, "headers": []}
    )


class TestRequireAuthSentryTenantTag:
    @pytest.mark.asyncio
    async def test_tag_set_for_the_bind_and_removed_on_teardown_when_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-on: the isolation scope's `tenant_id` tag is set to the bound
        tenant for exactly the lifetime of the yield-dependency's bind, mirroring
        the `tenant_context` ContextVar it lives and dies with."""
        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        dependency = require_auth()(
            request=_bare_request(), credentials=_acme_credentials()
        )
        params = await dependency.__anext__()
        try:
            assert params.tn == "acme"
            assert sentry_sdk.get_isolation_scope()._tags.get("tenant_id") == "acme"
        finally:
            await dependency.aclose()
        assert "tenant_id" not in sentry_sdk.get_isolation_scope()._tags

    @pytest.mark.asyncio
    async def test_tag_never_set_when_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag-off: even a `tn`-bearing token never sets the Sentry tag, matching
        the ContextVar bind it rides alongside (gated on MULTI_TENANT, not on `tn`
        presence)."""
        monkeypatch.setattr(settings, "MULTI_TENANT", False)
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        dependency = require_auth()(
            request=_bare_request(), credentials=_acme_credentials()
        )
        params = await dependency.__anext__()
        try:
            assert params.tn == "acme"
            assert "tenant_id" not in sentry_sdk.get_isolation_scope()._tags
        finally:
            await dependency.aclose()


# ---------------------------------------------------------------------------
# current_tenant_id() itself
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("bound_tenant")
def test_current_tenant_id_is_none_flag_off_even_when_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single source every chokepoint above reads: flag-off it is None no
    matter what is bound on the ContextVar."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    assert current_tenant_id() is None


@pytest.mark.usefixtures("bound_tenant")
def test_current_tenant_id_returns_the_bound_value_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert current_tenant_id() == "acme"
