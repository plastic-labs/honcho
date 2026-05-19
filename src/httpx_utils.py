"""Utility for creating httpx transports that work in IPv4-only Docker networks.

When running in Docker bridge networks (IPv4-only), httpx's async transport
uses anyio's Happy Eyeballs (RFC 8305) algorithm which can attempt IPv6
resolution before IPv4, causing POST requests to hang or crash the uvicorn
worker. Setting ``FORCE_IPV4=true`` forces all httpx async connections to
bind to ``0.0.0.0``, skipping the IPv6 path entirely.

See: https://github.com/plastic-labs/honcho/issues/605
"""

import os

import httpx


def _force_ipv4_enabled() -> bool:
    """Return True when the FORCE_IPV4 env var is set to a truthy value."""
    return os.environ.get("FORCE_IPV4", "").lower() in ("1", "true", "yes")


def get_async_transport(**kwargs) -> httpx.AsyncHTTPTransport | None:
    """Return an IPv4-bound async transport when FORCE_IPV4 is enabled.

    Pass the result as ``transport=`` to :class:`httpx.AsyncClient`.
    Returns ``None`` when no override is needed (the default).
    """
    if _force_ipv4_enabled():
        return httpx.AsyncHTTPTransport(local_address="0.0.0.0", **kwargs)
    return None


def get_httpx_client(**kwargs) -> httpx.AsyncClient | None:
    """Return a pre-configured :class:`httpx.AsyncClient` for libraries that
    accept an ``http_client`` parameter (e.g. ``openai.AsyncOpenAI``).

    Returns ``None`` when no override is needed.
    """
    transport = get_async_transport()
    if transport is not None:
        return httpx.AsyncClient(transport=transport, **kwargs)
    return None
