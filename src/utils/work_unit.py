"""Work unit utility functions for generating and parsing work unit keys."""

from typing import Any

from pydantic import BaseModel

from src.config import settings
from src.db import tenant_context

# The task types that lead a base (non-tenant-namespaced) work_unit_key. Used to
# tell whether a key carries a leading tenant_id prefix: the prefix is present
# (added under MULTI_TENANT) iff the first segment is not one of these.
_TASK_TYPES = frozenset(
    {
        "representation",
        "summary",
        "dream",
        "webhook",
        "deletion",
        "reconciler",
        "scope_backfill",
        "scope_removal",
    }
)


class ParsedWorkUnit(BaseModel):
    """Parsed work unit components."""

    task_type: str
    workspace_name: str | None
    session_name: str | None
    observer: str | None
    observed: str | None
    dream_type: str | None = None
    # Set when the key was tenant-namespaced (MULTI_TENANT); None otherwise.
    tenant_id: str | None = None


def construct_work_unit_key(
    workspace_name: str, payload: dict[str, Any] | ParsedWorkUnit
) -> str:
    """Generate a work unit key, tenant-namespaced when MULTI_TENANT is on.

    When MULTI_TENANT is set and a tenant is in scope, the current tenant_id is
    prepended so work units — and the batches drained from them — never span
    tenants (workspace_name alone is not globally unique). Off, or with no tenant
    in scope (e.g. the tenant-less reconciler task), the key is unchanged.
    """
    base_key = _construct_base_work_unit_key(workspace_name, payload)
    if settings.MULTI_TENANT and (tenant := tenant_context.get()):
        return f"{tenant}:{base_key}"
    return base_key


def _construct_base_work_unit_key(
    workspace_name: str, payload: dict[str, Any] | ParsedWorkUnit
) -> str:
    """
    Generate a work unit key for a given task type, workspace name, and event type.

    Args:
        workspace_name: The name of the workspace the work unit belongs to
        payload: Dictionary containing work unit information

    Returns:
        Formatted work unit key string

    Raises:
        ValueError: If required fields are missing or task type is invalid
    """
    if isinstance(payload, ParsedWorkUnit):
        payload = payload.model_dump()

    task_type: str | None = payload.get("task_type")
    if not workspace_name or not task_type:
        raise ValueError(
            "workspace_name and task_type are required to generate a work_unit_key"
        )

    if task_type in ["representation", "summary", "dream"]:
        observer = payload.get("observer", "None")
        observed = payload.get("observed", "None")
        session_name = payload.get("session_name", "None")
        if task_type == "dream":
            dream_type = payload.get("dream_type")
            if not dream_type:
                raise ValueError("dream_type is required for dream tasks")
            return f"{task_type}:{dream_type}:{workspace_name}:{observer}:{observed}"
        if task_type == "representation":
            # Representation tasks don't include observer in the key since
            # we process once and save to multiple collections
            return f"{task_type}:{workspace_name}:{session_name}:{observed}"
        return f"{task_type}:{workspace_name}:{session_name}:{observer}:{observed}"

    if task_type == "webhook":
        return f"webhook:{workspace_name}"

    if task_type == "deletion":
        deletion_type = payload.get("deletion_type")
        resource_id = payload.get("resource_id")
        if not deletion_type or not resource_id:
            raise ValueError(
                "deletion_type and resource_id are required for deletion tasks"
            )
        return f"deletion:{workspace_name}:{deletion_type}:{resource_id}"

    if task_type == "reconciler":
        reconciler_type = payload.get("reconciler_type")
        if not reconciler_type:
            raise ValueError("reconciler_type is required for reconciler tasks")
        return f"reconciler:{reconciler_type}"

    if task_type in ("scope_backfill", "scope_removal"):
        scope_peer = payload.get("scope_peer")
        session_name = payload.get("session_name")
        if not scope_peer or not session_name:
            raise ValueError(
                f"scope_peer and session_name are required for {task_type} tasks"
            )
        return f"{task_type}:{workspace_name}:{scope_peer}:{session_name}"

    raise ValueError(f"Invalid task type: {task_type}")


def parse_work_unit_key(work_unit_key: str) -> ParsedWorkUnit:
    """Parse a work unit key, transparently handling a tenant_id prefix.

    A key produced under MULTI_TENANT is `{tenant_id}:{base_key}`; otherwise it is
    just `{base_key}`. The two are told apart by the leading segment: a known task
    type means no prefix; anything else is a tenant_id to strip off and record.
    """
    head, _, rest = work_unit_key.partition(":")
    if head and head not in _TASK_TYPES and rest:
        parsed = _parse_base_work_unit_key(rest)
        parsed.tenant_id = head
        return parsed
    return _parse_base_work_unit_key(work_unit_key)


def _parse_base_work_unit_key(work_unit_key: str) -> ParsedWorkUnit:
    """
    Parse a work unit key to extract its components.

    Args:
        work_unit_key: The work unit key string to parse

    Returns:
        ParsedWorkUnit with extracted components

    Raises:
        ValueError: If the work unit key format is invalid
    """
    parts = work_unit_key.split(":")
    task_type = parts[0]

    if task_type == "representation":
        if len(parts) == 4:
            # New format: representation:{workspace}:{session}:{observed}
            return ParsedWorkUnit(
                task_type=task_type,
                workspace_name=parts[1],
                session_name=parts[2],
                observer=None,
                observed=parts[3],
            )
        elif len(parts) == 5:
            # Legacy format: representation:{workspace}:{session}:{observer}:{observed}
            return ParsedWorkUnit(
                task_type=task_type,
                workspace_name=parts[1],
                session_name=parts[2],
                observer=parts[3],
                observed=parts[4],
            )
        else:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )

    if task_type == "summary":
        if len(parts) != 5:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=parts[1],
            session_name=parts[2],
            observer=parts[3],
            observed=parts[4],
        )

    if task_type == "dream":
        if len(parts) != 5:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=parts[2],
            session_name=None,
            observer=parts[3],
            observed=parts[4],
            dream_type=parts[1],
        )

    if task_type == "webhook":
        if len(parts) != 2:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=parts[1],
            session_name=None,
            observer=None,
            observed=None,
        )

    if task_type == "deletion":
        if len(parts) != 4:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=parts[1],
            session_name=None,
            observer=None,
            observed=None,
        )

    if task_type == "reconciler":
        if len(parts) != 2:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=None,
            session_name=None,
            observer=None,
            observed=None,
        )

    if task_type in ("scope_backfill", "scope_removal"):
        # {task_type}:{workspace}:{scope_peer}:{session}
        if len(parts) != 4:
            raise ValueError(
                f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}"
            )
        return ParsedWorkUnit(
            task_type=task_type,
            workspace_name=parts[1],
            session_name=parts[3],
            # The scope peer is the observer of every collection the task touches.
            observer=parts[2],
            observed=None,
        )

    raise ValueError(f"Invalid task type in work_unit_key: {task_type}")
