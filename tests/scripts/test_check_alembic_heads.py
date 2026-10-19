"""Tests for the check_alembic_heads script."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_alembic_heads import main


def _write_revision(versions: Path, revision: str, down_revision: str | None) -> Path:
    """Write a minimal revision file into ``versions``."""
    path = versions / f"{revision}_rev.py"
    path.write_text(
        f'"""{revision}"""\n\n'
        + f"revision = {revision!r}\n"
        + f"down_revision = {down_revision!r}\n"
        + "branch_labels = None\n"
        + "depends_on = None\n"
    )
    return path


@pytest.fixture
def script_location(tmp_path: Path) -> Path:
    (tmp_path / "versions").mkdir()
    _write_revision(tmp_path / "versions", "base0001", None)
    return tmp_path


def test_single_head_passes(
    script_location: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_revision(script_location / "versions", "head0001", "base0001")

    assert main(script_location) == 0
    assert "head0001" in capsys.readouterr().out


def test_forked_heads_fail_and_name_both(
    script_location: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _write_revision(script_location / "versions", "fork0001", "base0001")
    second = _write_revision(script_location / "versions", "fork0002", "base0001")

    assert main(script_location) == 1
    err = capsys.readouterr().err
    assert "Found 2 alembic heads" in err
    for path in (first, second):
        assert path.stem.split("_")[0] in err
        assert str(path) in err
    assert err.count("down_revision: base0001") == 2


def test_repo_migrations_have_single_head() -> None:
    assert main() == 0
