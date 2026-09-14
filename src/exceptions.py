"""
Custom exceptions for the Honcho application.
"""

import json
from typing import Any, final

from src.config import settings


class HonchoException(Exception):
    """Base exception for all Honcho-specific errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: str | None = None, status_code: int | None = None):
        self.detail = detail or self.detail
        self.status_code = status_code or self.status_code
        super().__init__(self.detail)


@final
class ResourceNotFoundException(HonchoException):
    """Exception raised when a requested resource is not found."""

    status_code = 404
    detail = "Resource not found"


@final
class ObserverException(HonchoException):
    """Exception raised when a request tries to add too many observers to a session"""

    status_code = 400

    def __init__(self, session_name: str, extra_count: int):
        self.detail = (
            f"Cannot create session {session_name} with {extra_count} observers. "
            + f"Maximum allowed is {settings.SESSION_OBSERVERS_LIMIT} observers per session. "
            + "Observers are peers with 'observe_others' set to true."
        )
        super().__init__(self.detail)


@final
class ValidationException(HonchoException):
    """Exception raised when validation fails."""

    status_code = 422
    detail = "Validation error"


@final
class ConflictException(HonchoException):
    """Exception raised when there's a resource conflict."""

    status_code = 409
    detail = "Resource conflict"


@final
class AuthenticationException(HonchoException):
    """Exception raised when authentication fails."""

    status_code = 401
    detail = "Authentication failed"


@final
class AuthorizationException(HonchoException):
    """Exception raised when authorization fails."""

    status_code = 403
    detail = "Not authorized to access this resource"


@final
class DisabledException(HonchoException):
    """Exception raised when a feature is disabled."""

    status_code = 405
    detail = "Feature is disabled"


@final
class FilterError(HonchoException):
    """Exception raised when a filter is misconfigured or invalid."""

    status_code = 422
    detail = "Invalid filter configuration"


@final
class UnsupportedFileTypeError(HonchoException):
    status_code = 415
    detail = "Unsupported file type"


@final
class FileTooLargeError(HonchoException):
    status_code = 413
    detail = "File too large"


@final
class FileProcessingError(HonchoException):
    status_code = 500
    detail = "File processing error"


@final
class SurprisalError(HonchoException):
    """Exception raised when surprisal sampling fails during a dream cycle."""

    status_code = 500
    detail = "Surprisal sampling failed"


@final
class SpecialistExecutionError(HonchoException):
    """Exception raised when a specialist fails during dream orchestration."""

    status_code = 500
    detail = "Specialist execution failed"


@final
class VectorStoreError(HonchoException):
    """Exception raised when a vector store operation fails."""

    status_code = 500
    detail = "Vector store operation failed"


@final
class RepresentationSaveError(HonchoException):
    """Raised when every observer's representation save fails in a batch."""

    status_code: int = 500
    detail: str = "Representation save failed for all observers"


@final
class EmptyRepresentationError(HonchoException):
    """Raised when a batch's structured response is empty in a way that looks
    degraded (truncated output, provider content filter, or nothing returned at
    all) rather than a model that read the messages and asserted nothing.

    Raising this is what keeps the queue item unprocessed: the deriver calls it
    *before* the batch is marked processed, and the queue worker treats it as a
    retryable failure, so the work unit is re-claimed instead of being consumed
    with zero observations derived from it.

    The constructor carries the classification fields an operator needs to
    triage the failure without logging the prompt itself (upstream issue #993's
    observability contract): parse class, provider/model, attempt count, and the
    prompt's byte length + digest.
    """

    status_code: int = 500
    detail: str = "Deriver generated an empty representation"

    #: Triage fields; all optional so existing callers keep working.
    parse_class: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    attempts: int = 1
    prompt_bytes: int = 0
    prompt_digest: str = ""

    def __init__(
        self,
        detail: str,
        *,
        parse_class: str = "unknown",
        provider: str = "unknown",
        model: str = "unknown",
        attempts: int = 1,
        prompt_bytes: int = 0,
        prompt_digest: str = "",
    ) -> None:
        super().__init__(detail)
        self.parse_class = parse_class
        self.provider = provider
        self.model = model
        self.attempts = attempts
        self.prompt_bytes = prompt_bytes
        self.prompt_digest = prompt_digest


@final
class EmptySummaryError(HonchoException):
    """Raised when a summary's structured response is empty in a way that looks
    degraded rather than a genuinely empty conversation.

    The summarizer persists nothing when it falls back to a placeholder, so
    raising (and thereby requeueing the summary item) is the only path that can
    still produce a real summary for that boundary. Carries the same triage
    fields as :class:`EmptyRepresentationError`.
    """

    status_code: int = 500
    detail: str = "Summarizer generated an empty summary"

    parse_class: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    attempts: int = 1
    prompt_bytes: int = 0
    prompt_digest: str = ""

    def __init__(
        self,
        detail: str,
        *,
        parse_class: str = "unknown",
        provider: str = "unknown",
        model: str = "unknown",
        attempts: int = 1,
        prompt_bytes: int = 0,
        prompt_digest: str = "",
    ) -> None:
        super().__init__(detail)
        self.parse_class = parse_class
        self.provider = provider
        self.model = model
        self.attempts = attempts
        self.prompt_bytes = prompt_bytes
        self.prompt_digest = prompt_digest


class LLMError(Exception):
    """Exception raised when an LLM call fails.

    Accepts arbitrary positional and keyword inputs, normalizes them into a
    JSON-serializable object, and uses the resulting JSON string as the
    exception message. The normalized object is available via ``to_dict()``
    and the ``data`` attribute.

    Positional and keyword inputs are represented as a JSON object. If a
    single positional argument is a mapping and there are no keyword
    arguments, that mapping is used as the root object; otherwise the shape is
    ``{"args": [...], "kwargs": {...}}``. Values that are not natively
    serializable are converted using ``repr``.
    """

    data: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        normalized = {"args": list(args), "kwargs": kwargs}
        message = json.dumps(
            normalized, default=self._json_fallback, ensure_ascii=False
        )
        self.data = normalized
        super().__init__(message)

    @staticmethod
    def _json_fallback(value: Any) -> str:
        """Fallback serializer that returns ``repr(value)`` for unsupported types."""
        return repr(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized JSON object for programmatic access."""
        return self.data
