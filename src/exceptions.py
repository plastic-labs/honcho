"""
Custom exceptions for the Honcho application.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, final

from src.config import settings

# Mirrors sentry_sdk's LogLevelStr. Redeclared rather than imported so this
# module stays free of telemetry imports -- src.telemetry.sentry imports from
# here, and the reverse direction would be a cycle.
SentryLevel = Literal["fatal", "critical", "error", "warning", "info", "debug"]


@final
@dataclass(frozen=True, slots=True)
class SentryPolicy:
    """How ``src.telemetry.sentry.default_before_send`` should treat an exception.

    Attached to an exception class rather than matched on in the filter, so the
    decision lives next to the error it describes and adding a new exception
    never means editing the filter.

    Args:
        level: Event level to report at. Deliberately has no default: whether a
            kept exception is a page or a line on a graph is the whole decision,
            so each policy states it outright.
        fingerprint: Collapse every occurrence into this one Sentry issue.
            ``None`` leaves Sentry's default grouping alone.
    """

    level: SentryLevel
    fingerprint: str | None = None


class HonchoException(Exception):
    """Base exception for all Honcho-specific errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred"

    # Honcho exceptions are client-facing by default: the caller already sees
    # them in the HTTP response, so Sentry drops them (``None``). Override with
    # a SentryPolicy on subclasses that describe a symptom worth watching.
    sentry_policy: ClassVar[SentryPolicy | None] = None

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


class DegradedLLMResponseError(HonchoException):
    """Raised when a structured response came back empty in a way that looks
    degraded -- truncated or withheld by the provider -- rather than a model
    that read the input and had nothing to say.

    Raised from a ``validate_response`` hook inside the retried LLM call, so it
    is retried and reaches the fallback model like a provider error. The message
    carries the prompt's length and digest, never its text, so a repeat can be
    correlated across workers without logging user content.
    """

    status_code: int = 500
    detail: str = "Structured LLM response was empty and degraded"

    # Raised only after the retry budget and the fallback are spent, so it is
    # memory or a summary that did not get written: an error, not a client
    # response. Each subclass gets its own fingerprint so the per-issue spike
    # alert counts that failure's real rate instead of splitting it by stack.
    sentry_policy: ClassVar[SentryPolicy | None] = SentryPolicy(
        level="error", fingerprint="honcho-llm-degraded-empty-response"
    )

    def __init__(
        self,
        detail: str,
        *,
        parse_class: str,
        prompt_bytes: int,
        prompt_digest: str,
    ) -> None:
        super().__init__(detail)
        self.parse_class: str = parse_class
        self.prompt_bytes: int = prompt_bytes
        self.prompt_digest: str = prompt_digest

    @classmethod
    def for_prompt(
        cls,
        prompt: str,
        *,
        parse_class: str,
        finish_reasons: list[str],
        output_tokens: int,
    ) -> Self:
        """Build the error from the prompt that was sent, keeping only its
        length and a digest."""
        encoded = prompt.encode("utf-8")
        prompt_digest = hashlib.sha256(encoded).hexdigest()[:16]
        return cls(
            f"{cls.detail} (parse_class={parse_class}, "
            + f"finish_reasons={finish_reasons}, output_tokens={output_tokens}, "
            + f"prompt_bytes={len(encoded)}, prompt_sha256={prompt_digest})",
            parse_class=parse_class,
            prompt_bytes=len(encoded),
            prompt_digest=prompt_digest,
        )


@final
class EmptyRepresentationError(DegradedLLMResponseError):
    """The deriver's structured response for a batch was empty and degraded."""

    detail: str = "Deriver generated an empty representation"
    sentry_policy: ClassVar[SentryPolicy | None] = SentryPolicy(
        level="error", fingerprint="honcho-deriver-degraded-empty-representation"
    )


@final
class EmptySummaryError(DegradedLLMResponseError):
    """The summarizer's response was empty and degraded."""

    detail: str = "Summarizer generated an empty summary"
    sentry_policy: ClassVar[SentryPolicy | None] = SentryPolicy(
        level="error", fingerprint="honcho-summary-degraded-empty-summary"
    )


@final
class UpstreamLLMError(HonchoException):
    """Raised when the upstream model provider is unreachable or failing.

    Distinct from a bug in our own request: the provider (or the proxy in front
    of it) returned 5xx or refused the connection, so the caller should retry
    rather than treat the request as malformed.
    """

    status_code = 503
    detail = "Upstream language model provider is unavailable"

    # Kept by Sentry, unlike most Honcho errors: an outage is ours to notice,
    # and the deriver has no status-code metric to notice it in the way the
    # API does.
    # Reported at error level because the retry budget and the fallback chain
    # have both been spent by the time this escapes, so it is a sustained
    # outage rather than a blip -- and the fingerprint already keeps that to a
    # single issue rather than one per call site.
    sentry_policy: ClassVar[SentryPolicy | None] = SentryPolicy(
        level="error",
        fingerprint="honcho-upstream-llm-unavailable",
    )


@final
class StructuredResponseError(HonchoException):
    """Raised when a chat answer still fails its ``response_format`` schema after
    the retry budget and the fallback chain are spent."""

    status_code = 422
    detail = "The model's answer did not match the requested response_format"

    sentry_policy: ClassVar[SentryPolicy | None] = SentryPolicy(
        level="warning",
        fingerprint="honcho-dialectic-structured-response-invalid",
    )


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
