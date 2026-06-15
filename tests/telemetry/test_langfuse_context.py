import inspect

import pytest

from src.telemetry.langfuse_context import (
    build_honcho_langfuse_metadata,
    build_honcho_langfuse_trace_attrs,
    derive_tenant_user_id,
)


def test_derive_tenant_user_id_requires_explicit_prefix() -> None:
    assert derive_tenant_user_id("acme-user-123", None) is None
    assert derive_tenant_user_id("acme-user-123", "acme-") == "user-123"
    assert derive_tenant_user_id("other-user-123", "acme-") is None
    assert derive_tenant_user_id(None, "acme-") is None
    assert derive_tenant_user_id("acme-", "acme-") is None


def test_build_langfuse_metadata_uses_allowlist_and_generic_tenant_prefix() -> None:
    metadata = build_honcho_langfuse_metadata(
        operation="dialectic_chat",
        workspace_name="acme-user-123",
        session_name="chat-abc",
        observer="acme",
        observed="user-123",
        reasoning_level="low",
        message_count=2,
        tenant_workspace_prefix="acme-",
        tenant_platform="acme",
    )

    assert metadata == {
        "honcho_metadata_schema_version": "phase2.1",
        "component": "honcho",
        "subsystem": "honcho-memory",
        "honcho_operation": "dialectic_chat",
        "honcho_workspace_id": "acme-user-123",
        "honcho_session_id": "chat-abc",
        "honcho_observer_peer": "acme",
        "honcho_observed_peer": "user-123",
        "honcho_reasoning_level": "low",
        "honcho_message_count": 2,
        "tenant_user_id": "user-123",
        "tenant_platform": "acme",
    }


def test_build_langfuse_metadata_ignores_secret_shaped_list_values_and_has_no_passthrough() -> None:
    metadata = build_honcho_langfuse_metadata(
        operation="minimal_deriver",
        workspace_name="hermes",
        session_name="session",
        message_public_ids=["msg-public-1", "Bearer secret-token"],
    )

    assert metadata["honcho_workspace_id"] == "hermes"
    assert metadata["honcho_message_public_ids"] == ["msg-public-1"]
    assert "tenant_user_id" not in metadata
    assert "api_key" not in metadata
    assert "prompt" not in metadata
    assert "Bearer secret-token" not in str(metadata)

    sig = inspect.signature(build_honcho_langfuse_metadata)
    assert "extra" not in sig.parameters
    assert "unknown_fields" not in sig.parameters
    assert not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


@pytest.mark.parametrize(
    "secret_value",
    [
        "sk-or-v1-abcdef",
        "pk-live-abcdef",
        "lf_sk_abcdef",
        "lf_pk_abcdef",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature",
        "postgresql://user:password@localhost/db",
    ],
)
def test_build_langfuse_metadata_omits_secret_shaped_scalar_values(secret_value: str) -> None:
    metadata = build_honcho_langfuse_metadata(
        operation="dialectic_chat",
        workspace_name=secret_value,
        session_name="chat-abc",
        observer="acme",
        observed="user-123",
        run_id=secret_value,
    )

    assert secret_value not in str(metadata)
    assert "honcho_workspace_id" not in metadata
    assert "honcho_run_id" not in metadata


def test_build_langfuse_metadata_bounds_public_id_lists() -> None:
    message_ids = [f"msg-{i}" for i in range(40)]

    metadata = build_honcho_langfuse_metadata(
        operation="minimal_deriver",
        workspace_name="workspace",
        message_public_ids=message_ids,
    )

    assert metadata["honcho_message_public_ids"] == message_ids[:25]


def test_build_langfuse_trace_attrs_uses_safe_metadata() -> None:
    metadata = build_honcho_langfuse_metadata(
        operation="dialectic_chat",
        workspace_name="acme-user-123",
        session_name="chat-abc",
        tenant_workspace_prefix="acme-",
        tenant_platform="acme",
    )

    attrs = build_honcho_langfuse_trace_attrs(metadata)

    assert attrs == {
        "user_id": "user-123",
        "session_id": "chat-abc",
        "tags": ["honcho", "memory", "dialectic_chat", "acme"],
    }


def test_build_langfuse_trace_attrs_omits_absent_optional_values() -> None:
    metadata = build_honcho_langfuse_metadata(
        operation="short_summary",
        workspace_name="workspace",
    )

    attrs = build_honcho_langfuse_trace_attrs(metadata)

    assert attrs == {"tags": ["honcho", "memory", "short_summary"]}
