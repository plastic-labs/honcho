"""
Custom exceptions for the Honcho application.
"""

import json
from typing import Any, final


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
