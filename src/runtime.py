"""Single launcher for the server's processes: ``python -m src.runtime <command>``.

Each command runs the exact process the image runs today (Dockerfile CMD,
``docker/entrypoint.sh``, the compose deriver), so this module changes no
runtime behaviour. Imports stay stdlib-only: the launcher must not load
settings, open connections or pull in the app graph before it hands off.
"""

import argparse
import os
import signal
import subprocess  # nosec B404 - launches fixed server argv only
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("api", "deriver")


def command(name: str, extra: list[str] | None = None) -> list[str]:
    """argv for one process. ``extra`` is passed through to ``fastapi run``."""
    py = sys.executable
    if name == "api":
        workers = os.environ.get("API_WORKERS", "1")
        return [
            py, "-m", "fastapi", "run", "--host", "0.0.0.0",  # nosec B104 - same bind as the image
            "--workers", workers, *(extra or []), "src/main.py",
        ]  # fmt: skip
    if name == "deriver":
        return [py, "-m", "src.deriver"]
    if name == "migrate":
        return [py, "scripts/provision_db.py"]
    raise ValueError(f"unknown command: {name}")


def parse_roles(value: str) -> list[str]:
    roles = list(dict.fromkeys(r.strip() for r in value.split(",") if r.strip()))
    unknown = [r for r in roles if r not in ROLES]
    if unknown or not roles:
        raise argparse.ArgumentTypeError(
            f"roles must be a comma-separated subset of {','.join(ROLES)}"
        )
    return roles


def supervise(argvs: list[list[str]]) -> int:
    """Run processes side by side; when one exits, stop the rest.

    Returns the first exit code, so a crashed role fails the whole unit the
    same way a crashed container would.
    """
    procs = [subprocess.Popen(argv, cwd=ROOT) for argv in argvs]  # nosec B603
    signalled = False

    def forward(signum: int, _frame: object) -> None:
        nonlocal signalled
        signalled = True
        for p in procs:
            if p.poll() is None:
                p.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    pid, status = os.wait()
    code = os.waitstatus_to_exitcode(status)
    if code < 0:  # killed by a signal: report 128+N like a shell
        code = 128 - code
    # A second SIGTERM makes uvicorn skip its graceful drain, so only stop
    # siblings ourselves when the exit was not already a forwarded signal.
    for p in procs:
        if not signalled and p.pid != pid and p.poll() is None:
            p.terminate()
    for p in procs:
        if p.pid != pid:
            p.wait()
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="honcho-runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("api", help="API server; extra args go to `fastapi run`")
    sub.add_parser("deriver", help="background queue worker")
    sub.add_parser("migrate", help="apply database migrations")
    serve = sub.add_parser("serve", help="run several roles as child processes")
    serve.add_argument("--roles", type=parse_roles, default=list(ROLES))

    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "api":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "serve" and len(args.roles) > 1:
        return supervise([command(role) for role in args.roles])

    name = args.roles[0] if args.command == "serve" else args.command
    argv_ = command(name, extra)
    os.chdir(ROOT)
    os.execv(argv_[0], argv_)  # nosec B606


if __name__ == "__main__":
    sys.exit(main())
