"""Config that doesn't use the model catalog resolves exactly as before.

See `model_config_snapshot.py` for what each case covers and how the
snapshot was recorded. The cases run in one test so a parallel run boots
them once, not once per worker.
"""

import json

from tests.llm.model_config_snapshot import (
    CASES,
    SNAPSHOT_PATH,
    differences,
    run_all,
)


def test_every_case_resolves_as_recorded() -> None:
    recorded = json.loads(SNAPSHOT_PATH.read_text())
    assert set(recorded) == set(CASES)

    current = run_all()

    found = [
        line
        for case in sorted(CASES)
        for line in differences(case, recorded[case], current[case])
    ]
    assert not found, "\n".join(found)
