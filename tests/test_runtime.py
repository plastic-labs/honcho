import argparse
import sys
import time

import pytest

from src import runtime


def test_commands_match_image_entrypoints():
    assert runtime.command("api")[2:] == [
        "fastapi", "run", "--host", "0.0.0.0", "--workers", "1", "src/main.py",
    ]  # fmt: skip
    assert runtime.command("api", ["--port", "9000"])[-3:] == [
        "--port", "9000", "src/main.py",
    ]  # fmt: skip
    assert runtime.command("deriver")[1:] == ["-m", "src.deriver"]
    assert runtime.command("migrate")[1:] == ["scripts/provision_db.py"]


def test_parse_roles():
    assert runtime.parse_roles("deriver, api,api") == ["deriver", "api"]
    for bad in ("", "api,dreamer"):
        with pytest.raises(argparse.ArgumentTypeError):
            runtime.parse_roles(bad)


def test_supervise_returns_first_exit_and_stops_siblings():
    start = time.monotonic()
    code = runtime.supervise(
        [
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            [sys.executable, "-c", "import time; time.sleep(30)"],
        ]
    )
    assert code == 3
    assert time.monotonic() - start < 10
