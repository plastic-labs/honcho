"""Private, atomically saved progress for historical imports."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class ImportStateError(RuntimeError):
    """Progress cannot safely be read, changed, or resumed."""


def fingerprint(value: Any) -> str:
    """A stable identifier that contains neither conversation text nor local paths."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _validate_pending(pending: Any) -> None:
    if pending is None:
        return
    if not isinstance(pending, dict) or not _hash(pending.get("batch_id")):
        raise ImportStateError("Import progress has an invalid pending batch.")
    chunks, cursors = pending.get("chunks"), pending.get("cursors")
    if (
        not isinstance(chunks, list)
        or not 1 <= len(chunks) <= 100
        or not isinstance(cursors, dict)
    ):
        raise ImportStateError("Import progress has an invalid pending batch.")
    keys = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or not all(
            _hash(chunk.get(k)) for k in ("key", "payload_sha256", "row_hash")
        ):
            raise ImportStateError("Import progress has an invalid message checkpoint.")
        if not all(
            isinstance(chunk.get(k), str) and chunk[k]
            for k in ("segment_id", "root_id", "peer_id", "created_at")
        ):
            raise ImportStateError("Import progress has an invalid message checkpoint.")
        if (
            not _integer(chunk.get("row_id"), 1)
            or not _integer(chunk.get("part_index"))
            or not _integer(chunk.get("part_count"), 1)
        ):
            raise ImportStateError("Import progress has an invalid message checkpoint.")
        if chunk["part_index"] >= chunk["part_count"] or chunk["key"] in keys:
            raise ImportStateError("Import progress has an invalid message checkpoint.")
        keys.add(chunk["key"])
    if not isinstance(pending.get("target"), str) or not pending["target"]:
        raise ImportStateError("Import progress has an invalid target.")
    if not all(isinstance(k, str) and _integer(v) for k, v in cursors.items()):
        raise ImportStateError("Import progress has invalid scan cursors.")


def _validate(state: Any) -> None:
    if (
        not isinstance(state, dict)
        or type(state.get("schema_version")) is not int
        or state["schema_version"] != SCHEMA_VERSION
    ):
        raise ImportStateError(
            "Import progress is damaged or uses an unsupported schema version. Nothing was reset."
        )
    if not isinstance(state.get("sources"), dict) or not isinstance(
        state.get("scopes"), dict
    ):
        raise ImportStateError("Import progress is damaged. Nothing was reset.")
    for anchor in state["sources"].values():
        if not isinstance(anchor, dict) or not anchor:
            raise ImportStateError("Import progress has an invalid source identity.")
    for key, scope in state["scopes"].items():
        if (
            not _hash(key)
            or not isinstance(scope, dict)
            or not isinstance(scope.get("identity"), dict)
        ):
            raise ImportStateError("Import progress has an invalid scope.")
        if fingerprint(scope["identity"]) != key or not isinstance(
            scope.get("segments"), dict
        ):
            raise ImportStateError("Import progress has an invalid scope.")
        for segment in scope["segments"].values():
            if not isinstance(segment, dict) or not _hash(segment.get("mapping")):
                raise ImportStateError(
                    "Import progress has an invalid session mapping."
                )
            if not _integer(segment.get("last_scanned_message_id")) or not isinstance(
                segment.get("rows"), dict
            ):
                raise ImportStateError("Import progress has invalid scan cursors.")
            for row_id, row in segment["rows"].items():
                if (
                    not str(row_id).isdigit()
                    or int(row_id) < 1
                    or not isinstance(row, dict)
                ):
                    raise ImportStateError(
                        "Import progress has an invalid row checkpoint."
                    )
                if not _hash(row.get("row_hash")) or not _integer(
                    row.get("part_count"), 1
                ):
                    raise ImportStateError(
                        "Import progress has an invalid row checkpoint."
                    )
                if (
                    not _integer(row.get("confirmed_parts"))
                    or row["confirmed_parts"] > row["part_count"]
                ):
                    raise ImportStateError(
                        "Import progress has an invalid row checkpoint."
                    )
        _validate_pending(scope.get("pending"))


def read_state(path: Path) -> dict:
    """Read without creating directories, a lock, or a ledger; refuse corrupt data."""
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"schema_version": SCHEMA_VERSION, "sources": {}, "scopes": {}}
    except (OSError, ValueError) as exc:
        raise ImportStateError(
            "Could not read import progress. Repair or restore it before resuming; nothing was reset."
        ) from exc
    _validate(state)
    return state


@lru_cache(maxsize=1)
def _windows_user_sid() -> str:
    import csv

    result = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        capture_output=True,
        text=True,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    sid = next(csv.reader(result.stdout.splitlines()))[1]
    if not sid.startswith("S-1-") or any(c not in "S-0123456789" for c in sid):
        raise ImportStateError("Could not resolve the current user's file permissions.")
    return sid


def restrict_private_file(path: Path) -> None:
    """Set permissions before writing source text or progress into a new file."""
    if os.name == "nt":
        subprocess.run(
            [
                "icacls",
                str(path),
                "/inheritance:r",
                "/grant:r",
                f"*{_windows_user_sid()}:(F)",
            ],
            capture_output=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        os.chmod(path, 0o600)


def write_state(path: Path, state: dict) -> None:
    """Strict atomic replace: a failed rename must never fall back to an in-place rewrite."""
    from utils import fsync_directory

    _validate(state)
    payload = (json.dumps(state, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".hermes-import-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            restrict_private_file(temporary)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def state_lock(path: Path, *, timeout: float = 10.0) -> Iterator[None]:
    """Serialize local importers with a stable lock file; never fall back to no lock."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = Path(str(path) + ".lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ImportStateError(
            "Could not open the import progress lock. No upload was started."
        ) from exc
    acquired = False
    with os.fdopen(fd, "r+b", buffering=0) as handle:
        restrict_private_file(lock_path)
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
        deadline = time.monotonic() + timeout
        while not acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if (
                    exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
                    or time.monotonic() >= deadline
                ):
                    raise ImportStateError(
                        "Another importer holds the progress lock, or the lock is unavailable. No upload was started."
                    ) from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def get_scope(
    state: dict,
    identity: dict,
    source_path_key: str,
    anchor: dict,
    *,
    create: bool = False,
) -> dict | None:
    known_anchor = state["sources"].get(source_path_key)
    if known_anchor is not None and known_anchor != anchor:
        raise ImportStateError(
            "The source database was replaced or its identity changed. Restore the original source and progress before resuming."
        )
    key = fingerprint(identity)
    scope = state["scopes"].get(key)
    if scope is None and create:
        state["sources"][source_path_key] = anchor
        scope = state["scopes"][key] = {
            "identity": identity,
            "segments": {},
            "pending": None,
            "selection": None,
        }
    return scope


def get_segment(
    scope: dict | None, segment_id: str, mapping: str, *, create: bool = False
) -> dict | None:
    segment = scope["segments"].get(segment_id) if scope else None
    if segment is not None and segment["mapping"] != mapping:
        raise ImportStateError(
            "Session, peer, or chunking rules changed since the previous import. Restore those rules before resuming."
        )
    if segment is None and create:
        segment = scope["segments"][segment_id] = {
            "mapping": mapping,
            "last_scanned_message_id": 0,
            "rows": {},
        }
    return segment


def align_selection(scope: dict, selection: str) -> None:
    """A broader window must rescan older row IDs, while retaining confirmed hashes."""
    if scope.get("pending"):
        raise ImportStateError(
            "Resolve the pending batch before changing the selection."
        )
    if scope.get("selection") != selection:
        for segment in scope["segments"].values():
            segment["last_scanned_message_id"] = 0
        scope["selection"] = selection


def confirm_batch(scope: dict, pending: dict) -> None:
    """Advance parts and cursors only after every response descriptor was verified."""
    for chunk in pending["chunks"]:
        segment = scope["segments"][chunk["segment_id"]]
        row = segment["rows"].setdefault(
            str(chunk["row_id"]),
            {
                "row_hash": chunk["row_hash"],
                "part_count": chunk["part_count"],
                "confirmed_parts": 0,
            },
        )
        if (
            row["row_hash"] != chunk["row_hash"]
            or row["part_count"] != chunk["part_count"]
        ):
            raise ImportStateError(
                "A source row changed after an earlier part was imported. No duplicate was uploaded."
            )
        if row["confirmed_parts"] != chunk["part_index"]:
            raise ImportStateError(
                "The confirmed parts do not match the pending batch. Progress was not advanced."
            )
        row["confirmed_parts"] += 1
    for segment_id, row_id in pending["cursors"].items():
        segment = scope["segments"][segment_id]
        segment["last_scanned_message_id"] = max(
            segment["last_scanned_message_id"], row_id
        )
    scope["pending"] = None
