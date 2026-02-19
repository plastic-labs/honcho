"""HTTP client module for Honcho SDK."""

from .async_client import AsyncHonchoHTTPClient
from .client import HonchoHTTPClient
from .exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ConnectionError,
    HonchoError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    TimeoutError,
    UnprocessableEntityError,
)

__all__ = [
    # Errors
    "HonchoError",
    "APIError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ConnectionError",
    # Clients
    "HonchoHTTPClient",
    "AsyncHonchoHTTPClient",
]
