"""Build and exercise published Python artifacts without the root uv workspace.

Run with Python 3.11+ and uv on PATH. The target interpreter is independent of
the interpreter running this script; no server environment or services are used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = {"sdk": "sdks/python", "cli": "honcho-cli"}


def run(args: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def build(
    package: str, root: Path, env: dict[str, str]
) -> tuple[dict[str, str], list[Path]]:
    source = root / "sources" / package
    shutil.copytree(
        REPO / PACKAGES[package],
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            "node_modules",
        ),
    )
    metadata = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    output = root / "artifacts" / package
    # Default uv build creates an sdist and then builds its wheel from that sdist.
    # The copied package has neither a workspace root nor sibling source trees.
    run(
        [
            "uv",
            "build",
            "--no-sources",
            "--no-build-logs",
            "--python",
            sys.executable,
            "--out-dir",
            str(output),
            str(source),
        ],
        cwd=root,
        env=env,
    )
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(f"Expected one wheel and one sdist in {output}")
    return {"name": metadata["name"], "version": metadata["version"]}, wheels + sdists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=[*PACKAGES, "sdk-cli"], required=True)
    parser.add_argument("--python", required=True, help="Target Python version or path")
    parser.add_argument(
        "--artifacts", type=Path, help="Keep built archives in this directory"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="honcho-clean-install-") as temporary:
        root = Path(temporary).resolve()
        if root.is_relative_to(REPO):
            raise RuntimeError(
                "The temporary build/install directory must be outside the checkout"
            )
        # Remove project/interpreter discovery overrides. Keep ordinary network
        # and cache settings so the same command works on developer machines.
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "VIRTUAL_ENV",
                "PYTHONPATH",
                "PYTHONHOME",
                "UV_PROJECT",
                "UV_PROJECT_ENVIRONMENT",
                "UV_WORKING_DIR",
                "UV_CONFIG_FILE",
                "UV_PYTHON",
            }
            and not key.startswith("HONCHO_")
        }
        env.update(
            {
                "UV_NO_CONFIG": "1",
                "UV_NO_SOURCES": "1",
                "PYTHONNOUSERSITE": "1",
                "HONCHO_CONFIG_DIR": str(root / "config"),
                "HONCHO_NO_UPDATE_CHECK": "1",
            }
        )
        smoke = root / "smoke_installed_package.py"
        shutil.copy2(REPO / "scripts" / "smoke_installed_package.py", smoke)
        package = "sdk" if args.package == "sdk-cli" else args.package
        metadata, archives = build(package, root, env)
        cli_metadata: dict[str, str] | None = None
        cli_archives: list[Path] = []
        if args.package == "sdk-cli":
            shutil.rmtree(root / "sources")
            cli_metadata, cli_archives = build("cli", root, env)
        if args.artifacts:
            args.artifacts.mkdir(parents=True, exist_ok=True)
            for archive in archives + cli_archives:
                shutil.copy2(archive, args.artifacts / archive.name)
        # A source tree must not be available even accidentally at smoke time.
        shutil.rmtree(root / "sources")
        working = root / "unrelated-directory"
        working.mkdir()
        for archive in archives:
            kind = "wheel" if archive.suffix == ".whl" else "sdist"
            venv = root / f"venv-{kind}"
            run(
                ["uv", "venv", "--python", args.python, str(venv)], cwd=working, env=env
            )
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            requirement = str(archive)
            constraints: list[str] = []
            if cli_metadata is not None:
                requirement += "[cli]"
                cli_archive = next(
                    a for a in cli_archives if a.suffix == archive.suffix
                )
                constraint = root / "cli-constraint.txt"
                # A constraint selects our CLI artifact only if the SDK extra
                # actually depends on it; it must not install a missing extra.
                constraint.write_text(
                    f"honcho-cli @ {cli_archive.as_uri()}\n", encoding="utf-8"
                )
                constraints = ["--constraint", str(constraint)]
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    *constraints,
                    requirement,
                ],
                cwd=working,
                env=env,
            )
            run(["uv", "pip", "check", "--python", str(python)], cwd=working, env=env)
            run(
                [str(python), "-I", str(smoke), args.package, metadata["version"]],
                cwd=working,
                env=env,
            )
            if cli_metadata is not None:
                run(
                    [str(python), "-I", str(smoke), "cli", cli_metadata["version"]],
                    cwd=working,
                    env=env,
                )
            if args.package == "cli":
                # Install test tools only after proving the base wheel works.
                tests = root / f"tests-{kind}"
                shutil.copytree(
                    REPO / "honcho-cli" / "tests",
                    tests,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                config = root / "pytest.ini"
                config.write_text("[pytest]\n", encoding="utf-8")
                run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "pytest",
                        "pytest-mock",
                    ],
                    cwd=working,
                    env=env,
                )
                run(
                    [
                        str(python),
                        "-I",
                        "-m",
                        "pytest",
                        "-c",
                        str(config),
                        "--confcutdir",
                        str(tests),
                        str(tests),
                        "-q",
                    ],
                    cwd=working,
                    env=env,
                )
        print(
            json.dumps(
                {
                    "package": metadata["name"],
                    "version": metadata["version"],
                    "python": args.python,
                    "artifacts": [a.name for a in archives],
                    "result": "passed",
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
