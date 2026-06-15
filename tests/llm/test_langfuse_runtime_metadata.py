import inspect
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.config import AppSettings, settings
from src.llm.runtime import update_current_langfuse_observation


class FakeLangfuseClient:
    raise_on_update: bool

    def __init__(self, *, raise_on_update: bool = False) -> None:
        self.raise_on_update = raise_on_update
        self.span_calls: list[dict[str, Any]] = []
        self.propagate_calls: list[dict[str, Any]] = []

    def update_current_span(self, **kwargs: Any) -> None:
        if self.raise_on_update:
            raise RuntimeError("span boom")
        self.span_calls.append(kwargs)

    @contextmanager
    def propagate_attributes(self, **kwargs: Any):
        if self.raise_on_update:
            raise RuntimeError("trace boom")
        self.propagate_calls.append(kwargs)
        yield


def _fake_langfuse_module(fake_client: FakeLangfuseClient) -> SimpleNamespace:
    return SimpleNamespace(
        get_client=lambda: fake_client,
        propagate_attributes=fake_client.propagate_attributes,
    )


def test_langfuse_sdk_supports_required_span_and_trace_attrs() -> None:
    import langfuse
    from langfuse import get_client

    span_signature = inspect.signature(get_client().update_current_span)
    for parameter in ("name", "metadata"):
        assert parameter in span_signature.parameters

    trace_signature = inspect.signature(langfuse.propagate_attributes)
    for parameter in ("user_id", "session_id", "metadata", "tags"):
        assert parameter in trace_signature.parameters


def test_langfuse_propagate_attributes_sets_current_span_attrs() -> None:
    from langfuse import propagate_attributes
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    with (
        tracer.start_as_current_span("langfuse-propagation-test"),
        propagate_attributes(
            user_id="user-123",
            session_id="chat-abc",
            tags=["honcho", "memory"],
            metadata={"honcho_operation": "dialectic_chat"},
        ),
    ):
        pass

    attributes = cast(Mapping[str, object], exporter.get_finished_spans()[0].attributes)
    assert attributes["user.id"] == "user-123"
    assert attributes["session.id"] == "chat-abc"
    assert attributes["langfuse.trace.tags"] == ("honcho", "memory")
    assert attributes["langfuse.trace.metadata.honcho_operation"] == "dialectic_chat"


def test_update_current_langfuse_observation_noops_without_public_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        _fake_langfuse_module(fake_client),
    )

    update_current_langfuse_observation(
        "openai",
        "openai/gpt-4o-mini",
        metadata={"honcho_operation": "dialectic_chat"},
        trace_user_id="user-123",
        trace_session_id="chat-abc",
        trace_tags=["honcho"],
    )

    assert fake_client.span_calls == []
    assert fake_client.propagate_calls == []


def test_update_current_langfuse_observation_merges_span_metadata_and_updates_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "NAMESPACE", "honcho-test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        _fake_langfuse_module(fake_client),
    )

    metadata = {
        "component": "honcho",
        "honcho_operation": "dialectic_chat",
        "honcho_workspace_id": "acme-user-123",
    }

    update_current_langfuse_observation(
        "openai",
        "openai/gpt-4o-mini",
        name="Dialectic Agent",
        metadata=metadata,
        trace_user_id="user-123",
        trace_session_id="chat-abc",
        trace_tags=["honcho", "memory", "dialectic_chat"],
    )

    assert fake_client.span_calls == [
        {
            "name": "Dialectic Agent",
            "metadata": {
                "namespace": "honcho-test",
                "provider": "openai",
                "model": "openai/gpt-4o-mini",
                "component": "honcho",
                "honcho_operation": "dialectic_chat",
                "honcho_workspace_id": "acme-user-123",
            },
        }
    ]
    assert fake_client.propagate_calls == [
        {
            "user_id": "user-123",
            "session_id": "chat-abc",
            "tags": ["honcho", "memory", "dialectic_chat"],
            "metadata": metadata,
        }
    ]


def test_update_current_langfuse_observation_keeps_existing_behavior_without_new_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "NAMESPACE", "honcho-test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        _fake_langfuse_module(fake_client),
    )

    update_current_langfuse_observation(
        "openai",
        "openai/gpt-4o-mini",
    )

    assert fake_client.span_calls == [
        {
            "metadata": {
                "namespace": "honcho-test",
                "provider": "openai",
                "model": "openai/gpt-4o-mini",
            }
        }
    ]
    assert fake_client.propagate_calls == []


def test_update_current_langfuse_observation_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="src.llm.runtime")
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        _fake_langfuse_module(FakeLangfuseClient(raise_on_update=True)),
    )

    update_current_langfuse_observation(
        "openai",
        "openai/gpt-4o-mini",
        metadata={"honcho_operation": "dialectic_chat"},
        trace_user_id="user-123",
    )

    assert "Failed to update Langfuse" in caplog.text


def test_langfuse_tenant_settings_parse_with_defaults_and_env() -> None:
    defaults = AppSettings()
    configured = AppSettings(
        LANGFUSE_TENANT_WORKSPACE_PREFIX="acme-",
        LANGFUSE_TENANT_PLATFORM="acme",
    )

    assert defaults.LANGFUSE_TENANT_WORKSPACE_PREFIX is None
    assert defaults.LANGFUSE_TENANT_PLATFORM is None
    assert configured.LANGFUSE_TENANT_WORKSPACE_PREFIX == "acme-"
    assert configured.LANGFUSE_TENANT_PLATFORM == "acme"
