"""Shared test fixtures."""

from __future__ import annotations

import pytest

from honcho_cli import common
from honcho_cli.output import set_json_mode


@pytest.fixture(autouse=True)
def _reset_cli_globals():
    """Reset process-global CLI state between tests.

    ``_global_overrides`` (set by ``-w``/``-p``/``-s`` flags) and the JSON-mode
    flag are module globals that leak across tests otherwise — a workspace set
    by one test would silently satisfy the next test's workspace check.
    """
    yield
    common._global_overrides.update(workspace=None, peer=None, session=None)
    set_json_mode(False)
