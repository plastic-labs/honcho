"""A small read-only client for Discord's REST API.

HTTP only: it never opens a gateway connection, so it can run beside a bot that
is already connected with the same token. It waits out rate limits and retries
server errors, and reads every list Discord pages.
"""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, cast, final

import httpx

from honcho_discord.mapping import Json

logger = logging.getLogger(__name__)

API_BASE = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://github.com/plastic-labs/honcho, 0.1)"
PAGE = 100
_RETRIES = 6
_MAX_RATE_LIMIT_WAITS = 200


def _discord_message(response: httpx.Response) -> str:
    """Discord's error message, without echoing anything else from the response."""
    try:
        body = response.json()
    except ValueError:
        return "error"
    message: Any = (
        cast(dict[str, Any], body).get("message") if isinstance(body, dict) else None
    )
    return str(message)[:200] if message else "error"


@final
class DiscordError(Exception):
    """A Discord request failed.

    `retryable` is False when Discord refused it (no access, unknown channel),
    True when it kept failing (server errors, timeouts): running again later
    may succeed.
    """

    def __init__(self, status: int, path: str, reason: str, *, retryable: bool):
        super().__init__(f"Discord {status or 'error'} on {path}: {reason}")
        self.status: int = status
        self.path: str = path
        self.retryable: bool = retryable


@final
class DiscordClient:
    """Reads a server's channels, threads, members and message history."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = API_BASE,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bot {token}", "User-Agent": USER_AGENT},
            transport=transport,
            timeout=30,
        )
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DiscordClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a path and return its JSON.

        Rate limits are waited out and don't count as failures (up to
        `_MAX_RATE_LIMIT_WAITS` in a row; any other answer resets the count). Server errors and network failures
        are retried with backoff, `_RETRIES` times.
        """
        failures = 0
        rate_limited = 0
        while True:
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as error:
                failures += 1
                if failures >= _RETRIES:
                    # The exception's text can include request headers.
                    raise DiscordError(
                        0, path, type(error).__name__, retryable=True
                    ) from None
                self._sleep(min(2.0**failures, 30.0))
                continue
            if response.status_code == 429:
                rate_limited += 1
                if rate_limited > _MAX_RATE_LIMIT_WAITS:
                    raise DiscordError(429, path, "still rate limited", retryable=True)
                body = cast(dict[str, Any], response.json()) if response.content else {}
                wait = float(
                    body.get("retry_after")
                    or response.headers.get("Retry-After")
                    or 1.0
                )
                logger.info("Rate limited on %s; waiting %.2fs", path, wait)
                self._sleep(wait)
                continue
            rate_limited = 0
            if response.status_code >= 500:
                failures += 1
                if failures >= _RETRIES:
                    raise DiscordError(
                        response.status_code, path, "server error", retryable=True
                    )
                self._sleep(min(2.0**failures, 30.0))
                continue
            if response.status_code >= 400:
                raise DiscordError(
                    response.status_code,
                    path,
                    _discord_message(response),
                    retryable=False,
                )
            # Spend the bucket's last request, then wait for it to refill.
            if response.headers.get("X-RateLimit-Remaining") == "0":
                self._sleep(float(response.headers.get("X-RateLimit-Reset-After", 0)))
            return response.json()

    def channels(self, guild_id: str) -> list[Json]:
        return cast(list[Json], self.get(f"/guilds/{guild_id}/channels"))

    def active_threads(self, guild_id: str) -> list[Json]:
        body = cast(dict[str, Any], self.get(f"/guilds/{guild_id}/threads/active"))
        return cast(list[Json], body.get("threads", []))

    def archived_threads(self, channel_id: str) -> Iterator[Json]:
        """A channel's archived public threads, newest archived first."""
        before: str | None = None
        while True:
            params: dict[str, Any] = {"limit": PAGE}
            if before:
                params["before"] = before
            body = cast(
                dict[str, Any],
                self.get(f"/channels/{channel_id}/threads/archived/public", params),
            )
            threads = cast(list[Json], body.get("threads", []))
            yield from threads
            if not body.get("has_more") or not threads:
                return
            before = threads[-1]["thread_metadata"]["archive_timestamp"]

    def members(self, guild_id: str) -> Iterator[Json]:
        """Every member of the server. Needs the server members intent."""
        after = "0"
        while True:
            page = cast(
                list[Json],
                self.get(
                    f"/guilds/{guild_id}/members", {"limit": 1000, "after": after}
                ),
            )
            yield from page
            if len(page) < 1000:
                return
            after = str(max(int(m["user"]["id"]) for m in page))

    def channel(self, channel_id: str) -> Json | None:
        """A channel or thread, or None if it's gone or the bot can't see it."""
        try:
            return cast(Json, self.get(f"/channels/{channel_id}"))
        except DiscordError as error:
            if error.status in (403, 404):
                return None
            raise

    def message(self, channel_id: str, message_id: str) -> Json | None:
        """One message as it is now, or None if it's gone."""
        try:
            return cast(Json, self.get(f"/channels/{channel_id}/messages/{message_id}"))
        except DiscordError as error:
            if error.status == 404:
                return None
            raise

    def newest_message_id(self, channel_id: str) -> str | None:
        """The ID of a channel's or thread's newest message; None when it has none."""
        page = cast(
            list[Json], self.get(f"/channels/{channel_id}/messages", {"limit": 1})
        )
        return str(page[0]["id"]) if page else None

    def messages(self, channel_id: str, *, after: str = "0") -> Iterator[Json]:
        """A channel's or thread's messages after `after`, oldest first."""
        while True:
            page = cast(
                list[Json],
                self.get(
                    f"/channels/{channel_id}/messages", {"limit": PAGE, "after": after}
                ),
            )
            if not page:
                return
            # Discord sends the page closest to `after`, newest first. Read until
            # an empty page rather than trusting a short one to be the last.
            page = sorted(page, key=lambda m: int(m["id"]))
            yield from page
            after = str(page[-1]["id"])
