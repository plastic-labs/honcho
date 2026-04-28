"""Deterministic validation and provenance for observation candidates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.utils.types import DocumentLevel

ObservationOrigin = Literal["minimal_deriver", "agent_tool", "api", "dreamer", "dialectic"]
ValidationStatus = Literal["accepted", "rejected"]


class ObservationProvenance(BaseModel):
    """Machine-readable provenance attached to persisted observations."""

    origin: ObservationOrigin
    validation_status: ValidationStatus
    validation_errors: list[str] = Field(default_factory=list)
    source_message_ids: list[int] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)


class ObservationValidationResult(BaseModel):
    """Result of deterministic validation for one observation candidate."""

    accepted: bool
    errors: list[str] = Field(default_factory=list)
    provenance: ObservationProvenance


def validate_observation_candidate(
    *,
    content: str,
    level: DocumentLevel,
    origin: ObservationOrigin,
    source_message_ids: list[int] | None = None,
    source_ids: list[str] | None = None,
    available_source_ids: set[str] | None = None,
) -> ObservationValidationResult:
    """Validate an observation candidate without calling an LLM or embedding model."""

    errors: list[str] = []
    normalized_source_ids = list(source_ids or [])
    distinct_source_ids = set(normalized_source_ids)

    if not content.strip():
        errors.append("content must not be empty or whitespace-only")
    if "\x00" in content:
        errors.append("content must not contain NUL bytes")

    if level in ("deductive", "inductive") and not normalized_source_ids:
        errors.append(f"{level} observations require source_ids")
    if level == "contradiction" and len(distinct_source_ids) < 2:
        errors.append("contradiction observations require at least 2 distinct source_ids")

    if available_source_ids is not None:
        missing_source_ids = sorted(set(normalized_source_ids) - available_source_ids)
        if missing_source_ids:
            errors.append(
                f"source_ids not found in available context: {', '.join(missing_source_ids)}"
            )

    accepted = not errors
    provenance = ObservationProvenance(
        origin=origin,
        validation_status="accepted" if accepted else "rejected",
        validation_errors=errors,
        source_message_ids=list(source_message_ids or []),
        source_document_ids=normalized_source_ids,
    )
    return ObservationValidationResult(
        accepted=accepted,
        errors=errors,
        provenance=provenance,
    )
