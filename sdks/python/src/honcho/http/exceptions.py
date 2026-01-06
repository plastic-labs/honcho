"""Exception classes for API errors."""


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code:
            return f"APIError({self.status_code}): {self.message}"
        return f"APIError: {self.message}"


class AuthenticationError(APIError):
    """Raised for 401 authentication errors."""

    def __init__(self, message: str):
        super().__init__(message, 401)


class NotFoundError(APIError):
    """Raised for 404 not found errors."""

    def __init__(self, message: str):
        super().__init__(message, 404)


class RateLimitError(APIError):
    """Raised for 429 rate limit errors."""

    def __init__(self, message: str):
        super().__init__(message, 429)


class ServerError(APIError):
    """Raised for 5xx server errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)
