import os
import re
import subprocess
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile"
BASE_SYNC_ARGUMENTS = [
    "sync",
    "--frozen",
    "--no-install-project",
    "--no-group",
    "dev",
]


@pytest.fixture
def dependency_install() -> tuple[dict[str, str], str]:
    dockerfile = DOCKERFILE.read_text()
    defaults = dict(re.findall(r"^ARG (INSTALL_\w+)=(.*)$", dockerfile, re.MULTILINE))
    assert defaults == {"INSTALL_LANCEDB": "false", "INSTALL_SURPRISAL": "false"}
    run_instruction = re.search(
        r"^RUN --mount=type=cache,target=/root/\.cache/uv \\\n((?:.*\\\n)*.*)",
        dockerfile,
        re.MULTILINE,
    )
    assert run_instruction is not None
    return defaults, run_instruction[1].replace("\\\n", "\n")


def run_install(
    dependency_install: tuple[dict[str, str], str],
    overrides: dict[str, str],
    uv_exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    defaults, command = dependency_install
    recording_uv = f"uv() {{ printf '%s\\n' \"$@\"; return {uv_exit_code}; }};\n"
    return subprocess.run(
        ["/bin/sh", "-c", recording_uv + command],
        env={**os.environ, **defaults, **overrides},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("overrides", "extras"),
    [
        ({}, []),
        ({"INSTALL_LANCEDB": "false", "INSTALL_SURPRISAL": "false"}, []),
        ({"INSTALL_LANCEDB": "true"}, ["--extra", "lancedb"]),
        ({"INSTALL_SURPRISAL": "true"}, ["--extra", "surprisal"]),
        (
            {"INSTALL_LANCEDB": "true", "INSTALL_SURPRISAL": "true"},
            ["--extra", "lancedb", "--extra", "surprisal"],
        ),
    ],
)
def test_dependency_extras_compose(
    dependency_install: tuple[dict[str, str], str],
    overrides: dict[str, str],
    extras: list[str],
) -> None:
    result = run_install(dependency_install, overrides)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == BASE_SYNC_ARGUMENTS + extras


@pytest.mark.parametrize("argument", ["INSTALL_LANCEDB", "INSTALL_SURPRISAL"])
@pytest.mark.parametrize(
    "value", ["", "yes", "True", "FALSE", " true", "false ", "true; exit 0"]
)
@pytest.mark.parametrize("other_value", ["false", "true"])
def test_invalid_build_argument_rejected(
    dependency_install: tuple[dict[str, str], str],
    argument: str,
    value: str,
    other_value: str,
) -> None:
    other_argument = (
        "INSTALL_SURPRISAL" if argument == "INSTALL_LANCEDB" else "INSTALL_LANCEDB"
    )
    result = run_install(
        dependency_install, {argument: value, other_argument: other_value}
    )
    assert result.returncode == 2
    assert result.stderr.strip() == f"{argument} must be 'true' or 'false'"
    assert result.stdout == ""


def test_sync_failure_propagates(
    dependency_install: tuple[dict[str, str], str],
) -> None:
    result = run_install(dependency_install, {}, uv_exit_code=7)
    assert result.returncode == 7
    assert result.stdout.splitlines() == BASE_SYNC_ARGUMENTS
