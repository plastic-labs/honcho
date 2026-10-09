import sys

import pytest

from scripts import generate_jwt
from src.config import settings
from src.security import verify_jwt


@pytest.mark.parametrize(
    "flags",
    [
        ["--scope", "private"],
        ["--workspace", "ws", "--scope", "private", "--peer", "alice"],
        ["--workspace", "ws", "--scope", "private", "--session", "s"],
        ["--admin", "--scope", "private"],
        ["--workspace", "ws", "--scope", ""],
    ],
)
def test_invalid_scope_flags(monkeypatch: pytest.MonkeyPatch, flags: list[str]):
    monkeypatch.setattr(sys, "argv", ["generate_jwt.py", *flags])
    with pytest.raises(SystemExit) as exc_info:
        generate_jwt.main()
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("flags", "claim"),
    [
        (["--admin"], "ad"),
        (["--workspace", "ws"], "w"),
        (["--workspace", "ws", "--peer", "alice"], "p"),
        (["--workspace", "ws", "--session", "s"], "s"),
        (["--workspace", "ws", "--scope", "private"], "sc"),
    ],
)
def test_generate_key_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    claim: str,
):
    monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
    monkeypatch.setattr(
        sys, "argv", ["generate_jwt.py", *flags, "--expires", "8h", "--print-only"]
    )
    generate_jwt.main()
    params = verify_jwt(capsys.readouterr().out.strip())
    assert getattr(params, claim)
    assert (params.sc is not None) == (claim == "sc")


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
