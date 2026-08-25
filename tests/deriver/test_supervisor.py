import signal
import subprocess
from collections.abc import Callable
from typing import final

import pytest

from scripts import deriver_supervisor


@final
class FakeChild:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.signals: list[int] = []
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise subprocess.TimeoutExpired("deriver", 1)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -signal.SIGKILL


def _settings(*, failures_before_restart: int = 3) -> deriver_supervisor.SupervisorSettings:
    return deriver_supervisor.SupervisorSettings(
        start_period_seconds=0,
        interval_seconds=1,
        failures_before_restart=failures_before_restart,
        command_timeout_seconds=10,
    )


def _child_factory(child: FakeChild) -> Callable[[list[str]], FakeChild]:
    def start_child(_command: list[str]) -> FakeChild:
        return child

    return start_child


def test_supervisor_restarts_after_bounded_consecutive_probe_failures() -> None:
    child = FakeChild()
    failures = iter([False, False, False])

    result = deriver_supervisor.run_supervisor(
        _settings(),
        popen_factory=_child_factory(child),
        probe_runner=lambda _timeout: next(failures),
        wait_for_shutdown=lambda _seconds: False,
        install_signal_handlers=False,
    )

    assert result == 1
    assert child.signals == [signal.SIGTERM]


def test_supervisor_resets_failure_counter_after_a_healthy_probe() -> None:
    child = FakeChild()
    probes = iter([False, True, False])
    wait_count = 0

    def wait_for_shutdown(_seconds: float) -> bool:
        nonlocal wait_count
        wait_count += 1
        if wait_count == 4:
            child.returncode = 0
        return False

    result = deriver_supervisor.run_supervisor(
        _settings(failures_before_restart=2),
        popen_factory=_child_factory(child),
        probe_runner=lambda _timeout: next(probes),
        wait_for_shutdown=wait_for_shutdown,
        install_signal_handlers=False,
    )

    assert result == 1  # A clean child exit is still unexpected for a worker.
    assert child.signals == []


def test_supervisor_forwards_requested_shutdown_and_exits_cleanly() -> None:
    child = FakeChild()
    waits = iter([True])

    result = deriver_supervisor.run_supervisor(
        _settings(),
        popen_factory=_child_factory(child),
        probe_runner=lambda _timeout: True,
        wait_for_shutdown=lambda _seconds: next(waits),
        install_signal_handlers=False,
    )

    assert result == 0
    assert child.signals == [signal.SIGTERM]


def test_supervisor_converts_an_unexpected_clean_child_exit_to_failure() -> None:
    child = FakeChild(returncode=0)

    result = deriver_supervisor.run_supervisor(
        _settings(),
        popen_factory=_child_factory(child),
        probe_runner=lambda _timeout: True,
        wait_for_shutdown=lambda _seconds: False,
        install_signal_handlers=False,
    )

    assert result == 1
    assert child.signals == []


def test_healthcheck_timeout_is_an_unhealthy_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("healthcheck", 10)

    monkeypatch.setattr(subprocess, "run", timeout)

    assert deriver_supervisor.run_healthcheck(10) is False
