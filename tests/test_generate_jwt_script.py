import sys

import pytest

from scripts import generate_jwt


def test_admin_cannot_be_combined_with_scoped_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_jwt.py",
            "--admin",
            "--workspace",
            "my-workspace",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generate_jwt.main()

    assert exc_info.value.code == 2
