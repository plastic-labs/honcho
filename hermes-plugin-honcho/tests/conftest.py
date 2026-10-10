"""Load the external plugin with real Hermes host APIs, not its bundled copy.

Put a Hermes checkout on PYTHONPATH when running this standalone test suite.
"""

import atexit
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TEST_HOME = tempfile.TemporaryDirectory(prefix="honcho-plugin-test-")
atexit.register(_TEST_HOME.cleanup)
os.environ["HERMES_HOME"] = _TEST_HOME.name
for name in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_URL", "HERMES_HONCHO_HOST"):
    os.environ.pop(name, None)

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "hermes_honcho",
    _PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(_PLUGIN_ROOT)],
)
assert _spec is not None and _spec.loader is not None
_plugin = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _plugin
_spec.loader.exec_module(_plugin)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
