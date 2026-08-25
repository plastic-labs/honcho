import asyncio

import pytest
import uvloop

from src.config import settings
from src.deriver import __main__ as deriver_entrypoint
from src.deriver.queue_manager import QuarantinePersistenceError


def test_fatal_quarantine_error_exits_deriver_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process entrypoint must expose the fail-closed state to its supervisor."""
    fatal_error = QuarantinePersistenceError("forced failed quarantine")

    def raise_fatal(coroutine: object) -> None:
        close = getattr(coroutine, "close", None)
        if callable(close):
            close()
        raise fatal_error

    def ignore_event_loop_policy(_policy: asyncio.AbstractEventLoopPolicy) -> None:
        return None

    monkeypatch.setattr(deriver_entrypoint, "setup_logging", lambda: None)
    monkeypatch.setattr(asyncio, "set_event_loop_policy", ignore_event_loop_policy)
    monkeypatch.setattr(asyncio, "run", raise_fatal)
    monkeypatch.setattr(uvloop, "EventLoopPolicy", lambda: object())
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)

    assert deriver_entrypoint.run() == 1
