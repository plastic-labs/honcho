"""Safe Langfuse correlation metadata helpers for Honcho traces."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

HonchoLangfuseOperation = Literal[
    "dialectic_chat",
    "minimal_deriver",
    "short_summary",
    "long_summary",
]

_METADATA_SCHEMA_VERSION = "phase2.1"
_MAX_LIST_ITEMS = 25
_SECRET_PATTERNS = (
    re.compile(r"\bbearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9][A-Za-z0-9_-]*\b"),
    re.compile(r"\blf_(?:sk|pk)_[A-Za-z0-9][A-Za-z0-9_-]*\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s/]+:[^\s/@]+@"),
)


def _is_secret_shaped(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped == "***":
        return True
    return any(pattern.search(stripped) for pattern in _SECRET_PATTERNS)


def _safe_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or _is_secret_shaped(stripped):
        return None
    return stripped


def _safe_int(value: int | None) -> int | None:
    if value is None or isinstance(value, bool) or value < 0:
        return None
    return value


def _safe_string_list(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None

    safe_values: list[str] = []
    for value in values:
        safe = _safe_string(value)
        if safe is None:
            continue
        safe_values.append(safe)
        if len(safe_values) >= _MAX_LIST_ITEMS:
            break

    return safe_values or None


def derive_tenant_user_id(
    workspace_name: str | None,
    prefix: str | None,
) -> str | None:
    """Derive a generic tenant user id from an explicitly configured prefix."""
    safe_workspace = _safe_string(workspace_name)
    safe_prefix = _safe_string(prefix)
    if safe_workspace is None or safe_prefix is None:
        return None
    if not safe_workspace.startswith(safe_prefix):
        return None

    tenant_user_id = safe_workspace.removeprefix(safe_prefix).strip()
    return _safe_string(tenant_user_id)


def build_honcho_langfuse_metadata(
    *,
    operation: HonchoLangfuseOperation,
    workspace_name: str | None = None,
    session_name: str | None = None,
    observer: str | None = None,
    observed: str | None = None,
    observers: Iterable[str] | None = None,
    reasoning_level: str | None = None,
    run_id: str | None = None,
    message_count: int | None = None,
    message_public_ids: Iterable[str] | None = None,
    latest_message_public_id: str | None = None,
    queue_item_count: int | None = None,
    tenant_workspace_prefix: str | None = None,
    tenant_platform: str | None = None,
) -> dict[str, Any]:
    """Build allowlist-only metadata for Honcho Langfuse traces.

    This helper intentionally accepts explicit named fields only. It does not
    accept arbitrary request bodies or passthrough dictionaries because Langfuse
    metadata must never gain prompts, memory contents, credentials, or headers by
    accident.
    """
    metadata: dict[str, Any] = {
        "honcho_metadata_schema_version": _METADATA_SCHEMA_VERSION,
        "component": "honcho",
        "subsystem": "honcho-memory",
        "honcho_operation": operation,
    }

    string_fields = {
        "honcho_workspace_id": workspace_name,
        "honcho_session_id": session_name,
        "honcho_observer_peer": observer,
        "honcho_observed_peer": observed,
        "honcho_reasoning_level": reasoning_level,
        "honcho_run_id": run_id,
        "honcho_latest_message_public_id": latest_message_public_id,
        "tenant_platform": tenant_platform,
    }
    for key, value in string_fields.items():
        safe = _safe_string(value)
        if safe is not None:
            metadata[key] = safe

    observer_peers = _safe_string_list(observers)
    if observer_peers is not None:
        metadata["honcho_observer_peers"] = observer_peers

    public_ids = _safe_string_list(message_public_ids)
    if public_ids is not None:
        metadata["honcho_message_public_ids"] = public_ids

    int_fields = {
        "honcho_message_count": message_count,
        "honcho_queue_item_count": queue_item_count,
    }
    for key, value in int_fields.items():
        safe = _safe_int(value)
        if safe is not None:
            metadata[key] = safe

    tenant_user_id = derive_tenant_user_id(workspace_name, tenant_workspace_prefix)
    if tenant_user_id is not None:
        metadata["tenant_user_id"] = tenant_user_id

    return metadata


def build_honcho_langfuse_trace_attrs(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build top-level Langfuse trace attrs from safe Honcho metadata."""
    attrs: dict[str, Any] = {}

    user_id = _safe_string(metadata.get("tenant_user_id"))
    if user_id is not None:
        attrs["user_id"] = user_id

    session_id = _safe_string(metadata.get("honcho_session_id"))
    if session_id is not None:
        attrs["session_id"] = session_id

    tags = ["honcho", "memory"]
    operation = _safe_string(metadata.get("honcho_operation"))
    if operation is not None:
        tags.append(operation)
    tenant_platform = _safe_string(metadata.get("tenant_platform"))
    if tenant_platform is not None:
        tags.append(tenant_platform)
    attrs["tags"] = tags

    return attrs
