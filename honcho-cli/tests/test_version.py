"""The CLI's reported version must match the version it is published under."""

import tomllib
from pathlib import Path

from honcho_cli import __version__


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert __version__ == expected
