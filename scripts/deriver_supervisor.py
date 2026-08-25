"""Run Deriver under a bounded queue-health supervision loop.

Docker Compose and Fly restart containers or Machines after a process exits; a
passive health status alone is not a restart action.  This supervisor owns the
probe, terminates the child after bounded consecutive failures, and exits
nonzero so the deployment's restart policy can create a fresh Deriver.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_CHILD_STOP_TIMEOUT_SECONDS = 4


@dataclass(frozen=True)
class SupervisorSettings:
    """Runtime settings for the Deriver health supervisor."""

    start_period_seconds: int
    interval_seconds: int
    failures_before_restart: int
    command_timeout_seconds: int

    @classmethod
    def from_app_settings(cls) -> SupervisorSettings:
        return cls(
            start_period_seconds=settings.DERIVER.HEALTHCHECK_START_PERIOD_SECONDS,
            interval_seconds=settings.DERIVER.HEALTHCHECK_INTERVAL_SECONDS,
            failures_before_restart=(
                settings.DERIVER.HEALTHCHECK_FAILURES_BEFORE_RESTART
            ),
            command_timeout_seconds=settings.DERIVER.HEALTHCHECK_COMMAND_TIMEOUT_SECONDS,
        )


def run_healthcheck(command_timeout_seconds: int) -> bool:
    """Run the probe once, bounding the entire child command rather than just DB connect."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.deriver_healthcheck"],
            check=False,
            capture_output=True,
            text=True,
            timeout=command_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Deriver health probe exceeded its %ss command timeout",
            command_timeout_seconds,
        )
        return False

    if result.returncode == 0:
        return True

    detail = (result.stdout or result.stderr).strip()
    logger.error(
        "Deriver health probe exited %s%s",
        result.returncode,
        f": {detail[:1000]}" if detail else "",
    )
    return False


def _stop_child(child: Any, signum: int) -> None:
    """Forward a signal and reap the child, escalating only after the grace period."""
    if child.poll() is not None:
        return

    child.send_signal(signum)
    try:
        child.wait(timeout=_CHILD_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Deriver did not stop after %ss; killing child process",
            _CHILD_STOP_TIMEOUT_SECONDS,
        )
        child.kill()
        child.wait()


def run_supervisor(
    supervisor_settings: SupervisorSettings,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    probe_runner: Callable[[int], bool] = run_healthcheck,
    wait_for_shutdown: Callable[[float], bool] | None = None,
    install_signal_handlers: bool = True,
) -> int:
    """Run a Deriver child and return the status intended for the deployment runtime."""
    shutdown_event = threading.Event()
    shutdown_signal: int | None = None

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal shutdown_signal
        shutdown_signal = signum
        shutdown_event.set()

    previous_handlers: dict[int, Any] = {}
    if install_signal_handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_shutdown)

    wait = wait_for_shutdown or shutdown_event.wait

    try:
        child = popen_factory([sys.executable, "-m", "src.deriver"])
        if wait(supervisor_settings.start_period_seconds):
            _stop_child(child, shutdown_signal or signal.SIGTERM)
            return 0

        consecutive_failures = 0
        while child.poll() is None:
            if shutdown_event.is_set():
                _stop_child(child, shutdown_signal or signal.SIGTERM)
                return 0

            if probe_runner(supervisor_settings.command_timeout_seconds):
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.error(
                    "Deriver health probe failure %d/%d",
                    consecutive_failures,
                    supervisor_settings.failures_before_restart,
                )
                if (
                    consecutive_failures
                    >= supervisor_settings.failures_before_restart
                ):
                    _stop_child(child, signal.SIGTERM)
                    return 1

            if wait(supervisor_settings.interval_seconds):
                _stop_child(child, shutdown_signal or signal.SIGTERM)
                return 0

        exit_code = child.wait()
        logger.error("Deriver child exited unexpectedly with status %s", exit_code)
        return exit_code if exit_code != 0 else 1
    finally:
        if install_signal_handlers:
            for signum, previous_handler in previous_handlers.items():
                signal.signal(signum, previous_handler)


def main() -> int:
    """Run the production supervisor with configuration resolved by Honcho."""
    return run_supervisor(SupervisorSettings.from_app_settings())


if __name__ == "__main__":
    raise SystemExit(main())
