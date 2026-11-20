"""A small read-only client for GitHub's GraphQL API, and a REST GET for what
GraphQL can't do (comparing two commits).

It waits out rate limits, both the hourly budget and secondary limits, retries
server errors and network failures, and never echoes the token or a response
body in an error.
"""

import logging
import time
from collections.abc import Callable
from typing import Any, cast, final

import httpx

from honcho_github.mapping import canonical_host

logger = logging.getLogger(__name__)

USER_AGENT = "honcho-github-connector (https://github.com/plastic-labs/honcho)"
_RETRIES = 6
_MAX_RATE_LIMIT_WAITS = 50
# GitHub's documented wait after a secondary rate limit with no Retry-After.
_SECONDARY_WAIT = 60.0


def graphql_url(host: str) -> str:
    """The GraphQL endpoint: api.github.com, or `/api/graphql` on Enterprise."""
    name = canonical_host(host)
    if name == "github.com":
        return "https://api.github.com/graphql"
    return f"https://{name}/api/graphql"


@final
class GitHubError(Exception):
    """A GitHub request failed.

    `retryable` is False when GitHub refused it (bad token, no access, a query
    error), True when it kept failing (server errors, timeouts): running again
    later may succeed.
    """

    def __init__(self, status: int, reason: str, *, retryable: bool):
        super().__init__(f"GitHub {status or 'error'}: {reason}")
        self.status: int = status
        self.retryable: bool = retryable


def _error_types(errors: list[Any]) -> set[str]:
    return {
        str(cast(dict[str, Any], e).get("type")) for e in errors if isinstance(e, dict)
    }


def _error_summary(errors: list[Any]) -> str:
    """The first few errors' types and where they happened.

    Never GitHub's message text: it can quote what the query read, and this
    summary reaches logs and `export.json`.
    """
    parts: list[str] = []
    for e in errors[:3]:
        if isinstance(e, dict):
            error = cast(dict[str, Any], e)
            kind = str(error.get("type") or "ERROR")[:40]
            path = error.get("path")
            where = (
                ".".join(str(part) for part in cast(list[Any], path))[:120]
                if isinstance(path, list)
                else ""
            )
            parts.append(f"{kind} at {where}" if where else kind)
    return "; ".join(parts) or "error"


def _says_rate_limited(response: httpx.Response) -> bool:
    """A 403 whose message is about a rate limit (GitHub's secondary limits)."""
    try:
        message = cast(dict[str, Any], response.json()).get("message")
    except ValueError:
        return False
    return isinstance(message, str) and "rate limit" in message.lower()


def rest_url(host: str) -> str:
    """The REST API's root: api.github.com, or `/api/v3` on Enterprise."""
    name = canonical_host(host)
    if name == "github.com":
        return "https://api.github.com"
    return f"https://{name}/api/v3"


@final
class GitHubClient:
    """Runs GraphQL queries with a token that can read the repositories."""

    def __init__(
        self,
        token: str,
        *,
        host: str = "github.com",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
    ):
        self.host: str = host
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
            },
            transport=transport,
            timeout=60,
        )
        self._url = graphql_url(host)
        self._rest = rest_url(host)
        self._sleep = sleep
        self._now = now
        self.cost: int = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _wait_for_reset(self, reset: Any) -> None:
        """Wait until the hourly budget resets (at least a second)."""
        try:
            wait = float(reset) - self._now() + 1
        except (TypeError, ValueError):
            wait = _SECONDARY_WAIT
        wait = min(max(wait, 1.0), 3600.0)
        logger.info("GitHub rate limit reached; waiting %.0fs", wait)
        self._sleep(wait)

    def _wait_out_limit(self, response: httpx.Response, attempt: int) -> None:
        """Wait as GitHub asks: `Retry-After`, else the reset time if the hour's
        budget is spent and the reset is ahead, else a minute, doubling each time
        (GitHub's guidance for secondary limits)."""
        headers = response.headers
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                # GitHub's own wait, in full: retrying sooner is refused again.
                self._sleep(max(float(retry_after), 1.0))
                return
            except ValueError:
                pass
        reset = headers.get("X-RateLimit-Reset")
        if headers.get("X-RateLimit-Remaining") == "0" and reset is not None:
            try:
                if float(reset) > self._now():
                    self._wait_for_reset(reset)
                    return
            except ValueError:
                pass
        wait = min(_SECONDARY_WAIT * 2.0 ** (attempt - 1), 900.0)
        logger.info("GitHub secondary rate limit; waiting %.0fs", wait)
        self._sleep(wait)

    def rest(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        """GET a REST path and return its JSON; None when GitHub answers 404.

        Rate limits, server errors and network failures are handled as `query`
        handles them.
        """
        failures = 0
        rate_limited = 0
        headers = {"Accept": "application/vnd.github+json"}
        while True:
            try:
                response = self._http.get(
                    f"{self._rest}{path}", params=params, headers=headers
                )
            except httpx.TransportError as error:
                failures += 1
                if failures >= _RETRIES:
                    raise GitHubError(0, type(error).__name__, retryable=True) from None
                self._sleep(min(2.0**failures, 30.0))
                continue
            status = response.status_code
            if status == 429 or (
                status == 403
                and (
                    response.headers.get("Retry-After") is not None
                    or response.headers.get("X-RateLimit-Remaining") == "0"
                    or _says_rate_limited(response)
                )
            ):
                rate_limited += 1
                if rate_limited > _MAX_RATE_LIMIT_WAITS:
                    raise GitHubError(status, "still rate limited", retryable=True)
                self._wait_out_limit(response, rate_limited)
                continue
            if status >= 500:
                failures += 1
                if failures >= _RETRIES:
                    raise GitHubError(status, "server error", retryable=True)
                self._sleep(min(2.0**failures, 30.0))
                continue
            if status == 401:
                raise GitHubError(status, "the token was refused", retryable=False)
            if status == 404:
                return None
            if status >= 400:
                raise GitHubError(status, "request refused", retryable=False)
            try:
                return response.json()
            except ValueError:
                failures += 1
                if failures >= _RETRIES:
                    raise GitHubError(
                        status, "unreadable response", retryable=True
                    ) from None
                self._sleep(min(2.0**failures, 30.0))

    def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        allow_not_found: bool = False,
    ) -> dict[str, Any]:
        """Run a query and return its `data`.

        With `allow_not_found`, NOT_FOUND errors (a login that names no
        account) leave that field null instead of failing the query.
        """
        body = {"query": query, "variables": variables or {}}
        failures = 0
        rate_limited = 0
        while True:
            try:
                response = self._http.post(self._url, json=body)
            except httpx.TransportError as error:
                failures += 1
                if failures >= _RETRIES:
                    # The exception's text can include request headers.
                    raise GitHubError(0, type(error).__name__, retryable=True) from None
                self._sleep(min(2.0**failures, 30.0))
                continue

            status = response.status_code
            headers = response.headers
            if status == 429 or (
                status == 403
                and (
                    headers.get("Retry-After") is not None
                    or headers.get("X-RateLimit-Remaining") == "0"
                    or _says_rate_limited(response)
                )
            ):
                rate_limited += 1
                if rate_limited > _MAX_RATE_LIMIT_WAITS:
                    raise GitHubError(status, "still rate limited", retryable=True)
                self._wait_out_limit(response, rate_limited)
                continue
            if status >= 500:
                failures += 1
                if failures >= _RETRIES:
                    raise GitHubError(status, "server error", retryable=True)
                self._sleep(min(2.0**failures, 30.0))
                continue
            if status == 401:
                raise GitHubError(status, "the token was refused", retryable=False)
            if status >= 400:
                raise GitHubError(status, "request refused", retryable=False)

            try:
                payload = cast(dict[str, Any], response.json())
            except ValueError:
                failures += 1
                if failures >= _RETRIES:
                    raise GitHubError(
                        status, "unreadable response", retryable=True
                    ) from None
                self._sleep(min(2.0**failures, 30.0))
                continue
            errors = cast(list[Any], payload.get("errors") or [])
            if errors:
                types = _error_types(errors)
                if "RATE_LIMITED" in types:
                    rate_limited += 1
                    if rate_limited > _MAX_RATE_LIMIT_WAITS:
                        raise GitHubError(status, "still rate limited", retryable=True)
                    self._wait_out_limit(response, rate_limited)
                    continue
                if not (allow_not_found and types == {"NOT_FOUND"}):
                    raise GitHubError(status, _error_summary(errors), retryable=False)
            data = cast(dict[str, Any], payload.get("data") or {})
            limit = cast(dict[str, Any], data.pop("rateLimit", None) or {})
            self.cost += int(limit.get("cost") or 0)
            if limit.get("remaining") is not None and int(limit["remaining"]) < 50:
                # Spend no more of the hour's budget than the next query needs.
                self._wait_for_reset(headers.get("X-RateLimit-Reset"))
            return data
