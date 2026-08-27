import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
ENTRYPOINT = REPOSITORY_ROOT / "docker" / "entrypoint.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_entrypoint(tmp_path: Path, api_workers: str | None) -> list[str]:
    bin_dir = tmp_path / "app" / ".venv" / "bin"
    bin_dir.mkdir(parents=True)

    _write_executable(bin_dir / "python", "#!/bin/sh\nexit 0\n")

    args_file = tmp_path / "fastapi-args.txt"
    _write_executable(
        bin_dir / "fastapi",
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ENTRYPOINT_ARGS_FILE"\n',
    )

    entrypoint = tmp_path / "entrypoint.sh"
    entrypoint.write_text(
        ENTRYPOINT.read_text().replace("/app/.venv/bin", str(bin_dir))
    )

    env = os.environ.copy()
    env["ENTRYPOINT_ARGS_FILE"] = str(args_file)
    if api_workers is None:
        env.pop("API_WORKERS", None)
    else:
        env["API_WORKERS"] = api_workers

    result = subprocess.run(
        ["/bin/sh", str(entrypoint)],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    return args_file.read_text().splitlines()


def test_api_workers_defaults_to_one(tmp_path: Path) -> None:
    assert _run_entrypoint(tmp_path, api_workers=None) == [
        "run",
        "--host",
        "0.0.0.0",
        "--workers",
        "1",
        "src/main.py",
    ]


def test_api_workers_accepts_custom_count(tmp_path: Path) -> None:
    assert _run_entrypoint(tmp_path, api_workers="4") == [
        "run",
        "--host",
        "0.0.0.0",
        "--workers",
        "4",
        "src/main.py",
    ]
