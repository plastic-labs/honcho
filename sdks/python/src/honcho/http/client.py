"""Synchronous HTTP client for the Honcho API."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx

from .exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
)


class HttpClient:
    """Synchronous HTTP client with retry logic and authentication."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
    ):
        self._base_url: str = base_url.rstrip("/")
        self._api_key: str | None = api_key
        self._timeout: float = timeout
        self._max_retries: int = max_retries
        self._default_headers: dict[str, str] = default_headers or {}
        self._client: httpx.Client = httpx.Client(timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str | None:
        return self._api_key

    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self._default_headers,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _map_error(self, response: httpx.Response) -> APIError:
        status = response.status_code
        try:
            body = response.json()
            message = body.get("detail", str(body))
        except Exception:
            message = response.text or f"HTTP error {status}"

        if status == 401:
            return AuthenticationError(message)
        if status == 404:
            return NotFoundError(message)
        if status == 429:
            return RateLimitError(message)
        if status >= 500:
            return ServerError(message, status)
        return APIError(message, status)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: URL path (will be joined with base_url)
            json: JSON body to send
            params: Query parameters
            headers: Additional headers
            files: Files to upload (multipart/form-data)

        Returns:
            Parsed JSON response

        Raises:
            APIError: On API errors
        """
        url = f"{self._base_url}{path}"

        # Filter out None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                request_headers = self._build_headers(headers)

                # Remove Content-Type for file uploads (httpx sets it with boundary)
                if files:
                    request_headers.pop("Content-Type", None)

                response = self._client.request(
                    method,
                    url,
                    json=json if not files else None,
                    params=params,
                    headers=request_headers,
                    files=files,
                    data=json if files else None,  # Use data for multipart form
                )

                if response.is_success:
                    # Handle 204 No Content
                    if response.status_code == 204:
                        return None
                    text = response.text
                    if not text:
                        return None
                    return response.json()

                # Retry on 5xx errors
                if response.status_code >= 500 and attempt < self._max_retries:
                    last_error = self._map_error(response)
                    time.sleep(2**attempt)
                    continue

                raise self._map_error(response)

            except httpx.TimeoutException:
                last_error = APIError("Request timed out")
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
                    continue
                raise last_error

            except APIError:
                raise

            except Exception as e:
                last_error = APIError(str(e))
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
                    continue
                raise last_error

        raise last_error or APIError("Max retries exceeded")

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Stream HTTP response for SSE.

        Args:
            method: HTTP method
            path: URL path
            json: JSON body to send
            headers: Additional headers

        Yields:
            Lines from the SSE stream

        Raises:
            APIError: On API errors
        """
        url = f"{self._base_url}{path}"
        request_headers = self._build_headers(headers)
        request_headers["Accept"] = "text/event-stream"

        with self._client.stream(
            method,
            url,
            json=json,
            headers=request_headers,
        ) as response:
            if not response.is_success:
                # Read the response body for error details
                response.read()
                raise self._map_error(response)

            for line in response.iter_lines():
                if line:
                    yield line

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
