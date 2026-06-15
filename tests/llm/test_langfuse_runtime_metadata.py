import inspect
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import AppSettings, settings
from src.llm.runtime import update_current_langfuse_observation


class FakeLangfuseClient:
    raise_on_update: bool

    def __init__(self, *, raise_on_update: bool = False) -> None:
        self.raise_on_update = raise_on_update
        self.span_calls: list[dict[str, Any]] = []
        self.trace_calls: list[dict[str, Any]] = []

    def update_current_span(self, **kwargs: Any) -> None:
        if self.raise_on_update:
            raise RuntimeError("span boom")
        self.span_calls.append(kwargs)

    def update_current_trace(self, **kwargs: Any) -> None:
        if self.raise_on_update:
            raise RuntimeError("trace boom")
        self.trace_calls.append(kwargs)


def test_langfuse_sdk_update_current_trace_supports_required_attrs() -> None:
    from langfuse import get_client

    signature = inspect.signature(get_client().update_current_trace)

    for parameter in ("user_id", "session_id", "metadata", "tags"):
        assert parameter in signature.parameters


def test_update_current_langfuse_observation_noops_without_public_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        SimpleNamespace(get_client=lambda: fake_client),
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
    assert fake_client.trace_calls == []


def test_update_current_langfuse_observation_merges_span_metadata_and_updates_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "NAMESPACE", "honcho-test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        SimpleNamespace(get_client=lambda: fake_client),
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
    assert fake_client.trace_calls == [
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
        SimpleNamespace(get_client=lambda: fake_client),
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
    assert fake_client.trace_calls == []


def test_update_current_langfuse_observation_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="src.llm.runtime")
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setitem(
        __import__("sys").modules,
        "langfuse",
        SimpleNamespace(get_client=lambda: FakeLangfuseClient(raise_on_update=True)),
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
