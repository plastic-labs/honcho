"""Private native-process supervisor. No server imports or PID-file signalling.

The profile owns a token-authenticated Unix socket. Stop goes to that socket,
so a stale profile can never signal an unrelated process that reused a PID.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_state(directory: Path) -> dict[str, Any]:
    try:
        state = json.loads((directory / "native.json").read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(directory: Path, state: dict[str, Any]) -> None:
    temporary = directory / "native.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(directory / "native.json")


def control(state: dict[str, Any], action: str = "status") -> dict[str, Any] | None:
    if not state.get("socket") or not hasattr(socket, "AF_UNIX"):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(state["socket"])
            client.sendall(
                json.dumps({"instance": state["instance"], "action": action}).encode()
                + b"\n"
            )
            with client.makefile("rb") as reader:
                result = json.loads(reader.readline(65536))
            if result.get("instance") == state["instance"]:
                return result
    except (OSError, ValueError, KeyError):
        pass
    return None


def run(directory: Path, instance: str) -> int:
    state = read_state(directory)
    if state.get("instance") != instance:
        return 1
    children: dict[str, subprocess.Popen[bytes]] = {}
    stopping = False
    code = 0
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping, code
        stopping = True
        code = 128 + signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        sock.bind(state["socket"])
        os.chmod(state["socket"], 0o600)
        sock.listen(8)
        sock.settimeout(0.2)
        for name, argv in state["commands"].items():
            if stopping:
                break
            with (directory / f"{name}.log").open("ab") as log:
                children[name] = subprocess.Popen(
                    argv,
                    cwd=state["runtime"]["source"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        state.update(pid=os.getpid(), status="running")
        state["services"] = {
            name: {"pid": p.pid, "state": "running"} for name, p in children.items()
        }
        write_state(directory, state)
        while not stopping:
            for process in children.values():
                result = process.poll()
                if result is not None:
                    code = result if result >= 0 else 128 - result
                    stopping = True
                    break
            if stopping:
                break
            try:
                connection, _ = sock.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(0.5)
                try:
                    with connection.makefile("rb") as reader:
                        request = json.loads(reader.readline(4096))
                    if request.get("instance") != instance:
                        continue
                    if request.get("action") == "stop":
                        stopping = True
                    connection.sendall(json.dumps(state).encode() + b"\n")
                except (OSError, ValueError):
                    continue
    except Exception as exc:
        print(f"Native supervisor failed: {exc}", file=sys.stderr, flush=True)
        code = 1
    finally:
        # Send TERM once to each entry point; it owns signal forwarding/drain.
        for process in children.values():
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
        deadline = time.monotonic() + state.get("shutdownTimeout", 30)
        for process in children.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
        # Reap any descendants that survived a failed entry point or its drain.
        for process in children.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        state["services"] = {
            name: {"pid": p.pid, "state": "exited", "exitCode": p.returncode}
            for name, p in children.items()
        }
        state.update(status="stopped", exitCode=code)
        write_state(directory, state)
        sock.close()
        Path(state["socket"]).unlink(missing_ok=True)
    return code


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2]))
