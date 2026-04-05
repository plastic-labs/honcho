"""Exception classes for Honcho SDK errors."""

from __future__ import annotations

from typing import Any


class HonchoError(Exception):
    """Base error class for all Honcho SDK errors."""

    message: str
    status: int
    code: str | None
    body: Any

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        code: str | None = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.body = body

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, status={self.status})"
        )


class APIError(HonchoError):
    """Error from an API response with HTTP status code."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: Any = None,
    ) -> None:
        super().__init__(message, status=status, code="api_error", body=body)


class BadRequestError(APIError):
    """Error thrown when request validation fails (400)."""

    def __init__(self, message: str = "Bad request", body: Any = None) -> None:
        super().__init__(message, status=400, body=body)
        self.code = "bad_request"  # pyright: ignore[reportUnannotatedClassAttribute]


class AuthenticationError(APIError):
    """Error thrown when authentication fails (401)."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status=401)
        self.code = "authentication_error"  # pyright: ignore[reportUnannotatedClassAttribute]


class PermissionDeniedError(APIError):
    """Error thrown when the user lacks permission (403)."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(message, status=403)
        self.code = "permission_denied"  # pyright: ignore[reportUnannotatedClassAttribute]


class NotFoundError(APIError):
    """Error thrown when a resource is not found (404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status=404)
        self.code = "not_found"  # pyright: ignore[reportUnannotatedClassAttribute]


class ConflictError(APIError):
    """Error thrown on resource conflict (409)."""

    def __init__(self, message: str = "Resource conflict", body: Any = None) -> None:
        super().__init__(message, status=409, body=body)
        self.code = "conflict"  # pyright: ignore[reportUnannotatedClassAttribute]


class UnprocessableEntityError(APIError):
    """Error thrown when entity cannot be processed (422)."""

    def __init__(self, message: str = "Unprocessable entity", body: Any = None) -> None:
        super().__init__(message, status=422, body=body)
        self.code = "unprocessable_entity"  # pyright: ignore[reportUnannotatedClassAttribute]


class RateLimitError(APIError):
    """Error thrown when rate limited (429)."""

    retry_after: float | None = None

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status=429)
        self.code = "rate_limit_exceeded"  # pyright: ignore[reportUnannotatedClassAttribute]
        self.retry_after = retry_after


class ServerError(APIError):
    """Error thrown on server errors (5xx)."""

    def __init__(self, message: str = "Server error", status: int = 500) -> None:
        super().__init__(message, status=status)
        self.code = "server_error"  # pyright: ignore[reportUnannotatedClassAttribute]


class TimeoutError(HonchoError):
    """Error thrown when a request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message, code="timeout")


class ConnectionError(HonchoError):
    """Error thrown when a connection fails."""

    def __init__(self, message: str = "Connection failed") -> None:
        super().__init__(message, code="connection_error")


def create_error_from_response(
    status: int,
    message: str,
    body: Any = None,
    retry_after: float | None = None,
) -> HonchoError:
    """Create the appropriate error type based on HTTP status code."""
    if status == 400:
        return BadRequestError(message, body=body)
    elif status == 401:
        return AuthenticationError(message)
    elif status == 403:
        return PermissionDeniedError(message)
    elif status == 404:
        return NotFoundError(message)
    elif status == 409:
        return ConflictError(message, body=body)
    elif status == 422:
        return UnprocessableEntityError(message, body=body)
    elif status == 429:
        return RateLimitError(message, retry_after=retry_after)
    elif status >= 500:
        return ServerError(message, status=status)
    else:
        return APIError(message, status=status, body=body)
