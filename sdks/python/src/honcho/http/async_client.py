"""Async HTTP client for Honcho SDK."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from .exceptions import (
    ConnectionError,
    RateLimitError,
    ServerError,
    TimeoutError,
    create_error_from_response,
)

DEFAULT_TIMEOUT = 60.0  # 60 seconds
DEFAULT_MAX_RETRIES = 2
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
INITIAL_RETRY_DELAY = 0.5  # 500ms


class AsyncHonchoHTTPClient:
    """Async HTTP client for the Honcho API with retry logic and timeout support."""

    base_url: str
    api_key: str | None
    timeout: float
    max_retries: int
    default_headers: dict[str, str]
    default_query: dict[str, Any] | None
    _owns_client: bool
    _client: httpx.AsyncClient

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Remove trailing slash from base_url
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = {
            "Content-Type": "application/json",
            **(default_headers or {}),
        }
        self.default_query = default_query
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(  # nosec B113
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncHonchoHTTPClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make an HTTP request with automatic retries and timeout handling."""
        url = self._build_url(path)
        request_headers = self._build_headers(headers)
        request_timeout = timeout if timeout is not None else self.timeout
        merged_query = {**(self.default_query or {}), **(query or {})} or None

        last_error: Exception | None = None
        attempt = 0

        while attempt <= self.max_retries:
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=body if body is not None else None,
                    params=self._clean_query_params(merged_query),
                    headers=request_headers,
                    timeout=request_timeout,
                )

                if response.is_success:
                    # Handle empty responses
                    text = response.text
                    if not text:
                        return None
                    return response.json()

                # Handle error responses
                error_body = self._parse_error_body(response)
                retry_after = self._parse_retry_after(response)
                error = create_error_from_response(
                    response.status_code,
                    error_body.get("message") or f"HTTP {response.status_code}",
                    body=error_body,
                    retry_after=retry_after,
                )

                # Only retry on specific status codes
                if (
                    response.status_code in RETRY_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    last_error = error
                    await asyncio.sleep(self._get_retry_delay(attempt, retry_after))
                    attempt += 1
                    continue

                raise error

            except httpx.TimeoutException as e:
                error = TimeoutError(f"Request timed out after {request_timeout}s")
                if attempt < self.max_retries:
                    last_error = error
                    await asyncio.sleep(self._get_retry_delay(attempt))
                    attempt += 1
                    continue
                raise error from e

            except httpx.ConnectError as e:
                error = ConnectionError(f"Connection failed: {e}")
                if attempt < self.max_retries:
                    last_error = error
                    await asyncio.sleep(self._get_retry_delay(attempt))
                    attempt += 1
                    continue
                raise error from e

            except (TimeoutError, ConnectionError, RateLimitError, ServerError):
                raise

            except Exception as e:
                # Re-raise API errors
                if hasattr(e, "status"):
                    raise
                raise ConnectionError(str(e)) from e

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error
        raise ConnectionError("Request failed after retries")

    async def get(
        self,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a GET request."""
        return await self.request(
            "GET", path, query=query, headers=headers, timeout=timeout
        )

    async def post(
        self,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a POST request."""
        return await self.request(
            "POST", path, body=body, query=query, headers=headers, timeout=timeout
        )

    async def put(
        self,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a PUT request."""
        return await self.request(
            "PUT", path, body=body, query=query, headers=headers, timeout=timeout
        )

    async def patch(
        self,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a PATCH request."""
        return await self.request(
            "PATCH", path, body=body, query=query, headers=headers, timeout=timeout
        )

    async def delete(
        self,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a DELETE request."""
        return await self.request(
            "DELETE", path, body=body, query=query, headers=headers, timeout=timeout
        )

    async def stream(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[bytes]:
        """Make a streaming request that yields raw bytes for SSE parsing."""
        url = self._build_url(path)
        request_headers = {
            **self._build_headers(headers),
            "Accept": "text/event-stream",
        }
        request_timeout = timeout if timeout is not None else self.timeout
        merged_query = {**(self.default_query or {}), **(query or {})} or None

        async with self._client.stream(
            method,
            url,
            json=body if body is not None else None,
            params=self._clean_query_params(merged_query),
            headers=request_headers,
            timeout=request_timeout,
        ) as response:
            if not response.is_success:
                # Read error body
                await response.aread()
                error_body = self._parse_error_body(response)
                raise create_error_from_response(
                    response.status_code,
                    error_body.get("message") or f"HTTP {response.status_code}",
                    body=error_body,
                )

            async for chunk in response.aiter_bytes():
                yield chunk

    async def upload(
        self,
        path: str,
        *,
        files: dict[str, Any],
        data: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Make a multipart form data request (for file uploads)."""
        url = self._build_url(path)
        # Don't set Content-Type for multipart - httpx will set it with boundary
        request_headers = self._build_headers(headers)
        request_headers.pop("Content-Type", None)
        request_timeout = timeout if timeout is not None else self.timeout
        merged_query = {**(self.default_query or {}), **(query or {})} or None

        response = await self._client.post(
            url,
            files=files,
            data=data,
            params=self._clean_query_params(merged_query),
            headers=request_headers,
            timeout=request_timeout,
        )

        if not response.is_success:
            error_body = self._parse_error_body(response)
            raise create_error_from_response(
                response.status_code,
                error_body.get("message") or f"HTTP {response.status_code}",
                body=error_body,
            )

        text = response.text
        if not text:
            return None
        return response.json()

    def _build_url(self, path: str) -> str:
        """Build the full URL from path."""
        if path.startswith("/"):
            return f"{self.base_url}{path}"
        return f"{self.base_url}/{path}"

    def _build_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build request headers including auth."""
        headers = {**self.default_headers}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if extra:
            headers.update(extra)

        return headers

    def _clean_query_params(
        self, params: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Remove None values from query params."""
        if params is None:
            return None
        return {k: v for k, v in params.items() if v is not None}

    def _parse_error_body(self, response: httpx.Response) -> dict[str, Any]:
        """Parse error body from response."""
        try:
            body: Any = response.json()
            if isinstance(body, dict):
                body_dict: dict[str, Any] = cast(dict[str, Any], body)
                return {
                    "message": body_dict.get("detail")
                    or body_dict.get("message")
                    or body_dict.get("error"),
                    **body_dict,
                }
            return {"message": str(body)}
        except Exception:
            return {"message": f"HTTP {response.status_code}"}

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Parse Retry-After header."""
        header = response.headers.get("Retry-After")
        if not header:
            return None

        try:
            # Try parsing as seconds
            return float(header)
        except ValueError:
            pass

        try:
            # Try parsing as HTTP date
            import time
            from datetime import datetime
            from email.utils import parsedate_to_datetime

            dt: datetime = cast(datetime, parsedate_to_datetime(header))
            timestamp: float = dt.timestamp()
            return max(0.0, timestamp - time.time())
        except Exception:
            return None

    def _get_retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay before next retry."""
        if retry_after is not None:
            return retry_after
        # Exponential backoff: 0.5s, 1s, 2s, etc.
        return INITIAL_RETRY_DELAY * (2**attempt)
