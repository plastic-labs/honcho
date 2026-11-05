"""Config that doesn't use the model catalog resolves exactly as before.

See `model_config_snapshot.py` for what each case covers and how the
snapshot was recorded. The cases run in one test so a parallel run boots
them once, not once per worker.
"""

import json
from typing import Any

from tests.llm.model_config_snapshot import CASES, SNAPSHOT_PATH, run_all


def _differences(path: str, old: Any, new: Any) -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        old_d: dict[str, Any] = old  # pyright: ignore[reportUnknownVariableType]
        new_d: dict[str, Any] = new  # pyright: ignore[reportUnknownVariableType]
        found: list[str] = []
        for key in sorted(set(old_d) | set(new_d)):
            if key not in new_d:
                found.append(f"{path}.{key}: removed (was {old_d[key]!r})")
            elif key not in old_d:
                found.append(f"{path}.{key}: added ({new_d[key]!r})")
            else:
                found.extend(_differences(f"{path}.{key}", old_d[key], new_d[key]))
        return found
    if old != new:
        return [f"{path}: {old!r} -> {new!r}"]
    return []


def test_every_case_resolves_as_recorded() -> None:
    recorded = json.loads(SNAPSHOT_PATH.read_text())
    assert set(recorded) == set(CASES)

    current = run_all()

    differences = [
        line
        for case in sorted(CASES)
        for line in _differences(case, recorded[case], current[case])
    ]
    assert not differences, "\n".join(differences)
