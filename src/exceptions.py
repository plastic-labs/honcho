"""
Custom exceptions for the Honcho application.
"""

from typing import Optional


class HonchoException(Exception):
    """Base exception for all Honcho-specific errors."""

    status_code = 500
    detail = "An unexpected error occurred"

    def __init__(self, detail: Optional[str] = None, status_code: Optional[int] = None):
        self.detail = detail or self.detail
        self.status_code = status_code or self.status_code
        super().__init__(self.detail)


class ResourceNotFoundException(HonchoException):
    """Exception raised when a requested resource is not found."""

    status_code = 404
    detail = "Resource not found"


class ValidationException(HonchoException):
    """Exception raised when validation fails."""

    status_code = 422
    detail = "Validation error"


class ConflictException(HonchoException):
    """Exception raised when there's a resource conflict."""

    status_code = 409
    detail = "Resource conflict"


class AuthenticationException(HonchoException):
    """Exception raised when authentication fails."""

    status_code = 401
    detail = "Authentication failed"


class AuthorizationException(HonchoException):
    """Exception raised when authorization fails."""

    status_code = 403
    detail = "Not authorized to access this resource"


class DisabledException(HonchoException):
    """Exception raised when a feature is disabled."""

    status_code = 405
    detail = "Feature is disabled"
