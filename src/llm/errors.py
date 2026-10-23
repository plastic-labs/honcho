"""Translate provider SDK transport failures into Honcho's own error type.

Every backend call funnels through `src.llm.request_builder`, so this is the one
place that has to know what a provider outage looks like across the three SDKs.
They do not share a base class, but they split failures the same two ways:

* the request never got a response -- each SDK has its own connection-error
  class (timeouts subclass it), and google-genai surfaces raw httpx errors, or
  a bare `TimeoutError` on its aiohttp transport;
* the request got a response carrying a server-side status -- anthropic and
  openai expose it as `status_code`, google-genai as `code`.

It also owns the retry condition every LLM retry site shares, since that
decision turns on the same per-SDK status reading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import anthropic
import httpx
import openai
from google.genai import errors as genai_errors
from tenacity import RetryCallState

from src.exceptions import UpstreamLLMError

logger = logging.getLogger(__name__)

# Anything at or above this is the provider (or a proxy in front of it) failing,
# not a malformed request from us. 4xx stays untranslated so genuine client
# errors -- and 429, which callers throttle on rather than retry blindly -- keep
# surfacing as themselves.
_SERVER_ERROR_FLOOR = 500

# "Never reached the provider." Neither anthropic's nor openai's class derives
# from httpx's, so each must be named; httpx and `TimeoutError` cover
# google-genai, which surfaces transport failures unwrapped.
_CONNECTION_ERRORS = (
    anthropic.APIConnectionError,
    openai.APIConnectionError,
    httpx.TransportError,
    TimeoutError,
)


def _status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a provider SDK exception.

    Set on the instance rather than the class, so this reads attributes instead
    of matching types: anthropic and openai use `status_code`, google `code`.
    """
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def as_upstream_error(exc: BaseException) -> UpstreamLLMError | None:
    """Return the 503 to raise for `exc`, or None to let it propagate as-is.

    Covers both halves of an outage: a request that never landed and one that
    landed on something returning 5xx -- a litellm proxy mid-restart produces
    each in turn.
    """
    if isinstance(exc, _CONNECTION_ERRORS):
        return UpstreamLLMError(f"Could not reach the model provider: {exc}")

    status = _status_of(exc)
    if status is not None and status >= _SERVER_ERROR_FLOOR:
        return UpstreamLLMError(f"Model provider returned HTTP {status}")

    return None


def raise_upstream_error(exc: BaseException) -> None:
    """Re-raise `exc` as an `UpstreamLLMError` when it looks like an outage.

    A no-op for anything else, so the caller's bare `raise` still applies.
    """
    upstream = as_upstream_error(exc)
    if upstream is not None:
        raise upstream from exc


# The provider's own status errors -- the only exceptions whose status says
# anything about the request. Honcho's exceptions carry a `status_code` too
# (ValidationException is 422), so status is read only after this type check.
_PROVIDER_STATUS_ERRORS = (
    anthropic.APIStatusError,
    openai.APIStatusError,
    genai_errors.ClientError,
)

# The 4xx a provider sends about its own state rather than the request: timed
# out, conflicting, too early, rate limited. The same request can succeed later.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 409, 425, 429})


def is_non_retryable_client_error(exc: BaseException) -> bool:
    """True when the provider rejected the request itself, so resending it
    unchanged gets the same answer (bad request, auth, not found, too large).

    Only provider SDK status errors qualify. Everything else -- outages already
    translated to `UpstreamLLMError`, connection errors, and the response
    checks that consume attempts by design -- stays retryable. The openai and
    anthropic SDKs still honour a server's `x-should-retry` hint in their own
    retries before raising.
    """
    if not isinstance(exc, _PROVIDER_STATUS_ERRORS):
        return False
    status = _status_of(exc)
    return (
        status is not None
        and 400 <= status < _SERVER_ERROR_FLOOR
        and status not in _RETRYABLE_CLIENT_STATUSES
    )


def llm_retry_condition(
    *, can_fall_back: bool, retry_attempts: int, model: str
) -> Callable[[RetryCallState], bool]:
    """Build the tenacity `retry=` condition for an LLM retry site.

    Retries any `Exception`, as tenacity's default does, except a
    non-retryable client error when no fallback can take a later attempt.
    With a fallback configured the error keeps today's path: it is retried
    and the final attempt goes to the fallback, which may accept the request.

    Args:
        can_fall_back: Whether a later attempt can reach a different model.
        retry_attempts: The site's attempt budget, for the log line.
        model: `transport/model` of the attempts this condition stops, for the
            log line.
    """

    def _should_retry(retry_state: RetryCallState) -> bool:
        """Decide whether tenacity retries the attempt recorded in `retry_state`."""
        outcome = retry_state.outcome
        if outcome is None or not outcome.failed:
            return False
        exc = outcome.exception()
        if not isinstance(exc, Exception):
            return False
        if can_fall_back or not is_non_retryable_client_error(exc):
            return True
        # tenacity asks this before checking the budget, so only an attempt
        # that had retries left is an early stop worth logging.
        if retry_state.attempt_number < retry_attempts:
            logger.warning(
                f"Not retrying {model} after attempt "
                + f"{retry_state.attempt_number}/{retry_attempts}: the provider "
                + f"rejected the request with HTTP {_status_of(exc)} and no "
                + f"other model can take a later attempt: {exc}"
            )
        return False

    return _should_retry


__all__ = [
    "as_upstream_error",
    "is_non_retryable_client_error",
    "llm_retry_condition",
    "raise_upstream_error",
]
