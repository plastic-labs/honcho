"""Run the shared opt-out regressions in the pinned plugin environment."""

import runpy
from pathlib import Path

_shared = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tests/plugins/test_honcho_observation_opt_out.py"))
globals().update({name: value for name, value in _shared.items() if name.startswith("test_") or name == "plugin"})
