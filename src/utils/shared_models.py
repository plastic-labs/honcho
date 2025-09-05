"""
Shared Pydantic models used by both dialectic and deriver modules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PeerCardQuery(BaseModel):
    """
    Model for peer card query generation responses.

    Contains the new peer card, or None if there are no new key observations.
    The notes field is just a place for stupid models to dump useless info.
    """

    card: list[str] | None
    notes: str | None


class SemanticQueries(BaseModel):
    """Model for semantic query generation responses."""

    queries: list[str] = Field(
        description="List of semantic search queries to retrieve relevant observations"
    )
