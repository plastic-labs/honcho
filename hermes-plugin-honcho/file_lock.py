"""Portable bounded file locking for plugin operations."""

from __future__ import annotations

import errno
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TextIO

fcntl: ModuleType | None
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

msvcrt: ModuleType | None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

logger = logging.getLogger(__name__)


def acquire_file_lock(lock_path: Path, timeout: float = 5.0) -> TextIO | None:
    """Acquire a portable non-blocking file lock before the deadline."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - returned to the lock owner
    if msvcrt is not None:
        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write(" ")
            lock_file.flush()

    deadline = time.monotonic() + timeout
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - supported platforms provide one primitive
                logger.warning("File lock skipped: no file-lock primitive available")
                lock_file.close()
                return None
            return lock_file
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                logger.warning("File lock failed: %s", exc)
                lock_file.close()
                return None
            if time.monotonic() >= deadline:
                logger.warning("File lock skipped: timed out acquiring %s", lock_path)
                lock_file.close()
                return None
            time.sleep(0.05)


def release_file_lock(lock_file: TextIO) -> None:
    """Release a lock acquired by :func:`acquire_file_lock`."""
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError as exc:
        logger.warning("Failed to release file lock: %s", exc)
    finally:
        lock_file.close()


@contextmanager
def file_lock(lock_path: Path, timeout: float = 5.0):
    """Yield acquisition status and release the file lock on every exit path."""
    lock_file = None
    try:
        try:
            lock_file = acquire_file_lock(lock_path, timeout)
        except OSError as exc:
            logger.warning("File lock failed: %s", exc)
        yield lock_file is not None
    finally:
        if lock_file is not None:
            release_file_lock(lock_file)
