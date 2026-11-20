"""The CLI's reported version must match the version it is published under."""

import tomllib
from pathlib import Path

from honcho_cli import __version__


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert __version__ == expected


def test_the_sdk_floor_keeps_up_with_the_sdk():
    """The CLI imports what this repository's SDK has, so the SDK's floor moves
    with it: a release that changes the SDK's minor version raises the floor."""
    root = Path(__file__).resolve().parents[2]
    cli = tomllib.loads((root / "honcho-cli" / "pyproject.toml").read_text())
    sdk = tomllib.loads((root / "sdks" / "python" / "pyproject.toml").read_text())
    floors = [
        d.removeprefix("honcho-ai>=")
        for d in cli["project"]["dependencies"]
        if d.startswith("honcho-ai>=")
    ]

    assert len(floors) == 1
    assert floors[0].split(".")[:2] == sdk["project"]["version"].split(".")[:2]
