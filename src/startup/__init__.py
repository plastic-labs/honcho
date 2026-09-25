"""Startup-time validators that gate API/deriver boot."""

from src.startup.embedding_validator import (
    StartupValidationError,
    validate_embedding_schema,
)

__all__ = ("StartupValidationError", "validate_embedding_schema")
