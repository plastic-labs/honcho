"""Poll the local API health endpoint."""

from __future__ import annotations

import time

import httpx

from honcho_cli.local.docker import compose_ps, services_running, stack_containers_up
from honcho_cli.local.profile import LocalProfile


def api_healthy(base_url: str, *, timeout: float = 2.0) -> bool:
    """True when ``GET /health`` returns HTTP 200."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(base_url.rstrip("/") + "/health")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def stack_healthy(profile: LocalProfile) -> bool:
    """True when Compose services are up and the API answers /health."""
    if not profile.compose_file().exists():
        return False
    ps = compose_ps(profile)
    if not stack_containers_up(ps):
        return False
    return api_healthy(profile.base_url)


def wait_for_health(
    profile: LocalProfile,
    *,
    timeout: float,
    interval: float = 1.0,
) -> bool:
    """Poll until the API is healthy or ``timeout`` seconds elapse.

    Returns False on timeout. Fails fast if a required container has exited.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ps = compose_ps(profile)
        states = services_running(ps)
        for _name, state in states.items():
            if "exit" in state or state in {"dead"}:
                return False
        if api_healthy(profile.base_url) and stack_containers_up(ps):
            return True
        time.sleep(interval)
    return api_healthy(profile.base_url)
