"""Atomic checkpoints, source identity, and process-level import locking."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hermes_plugin_honcho import import_state as progress


def valid_state():
    """Create a valid isolated ledger and destination scope for state tests."""
    state = {"schema_version": 1, "sources": {}, "scopes": {}}
    identity = {
        "server": "http://localhost:8000",
        "workspace": "example",
        "profile": "default",
        "source": "a" * 64,
    }
    scope = progress.get_scope(
        state, identity, "b" * 64, {"generation": "test-generation"}, create=True
    )
    progress.get_segment(scope, "physical-one", "c" * 64, create=True)
    return state, scope


def descriptor(index=0, count=1, row_id=1):
    """Describe one synthetic source chunk without storing transcript text."""
    return {
        "segment_id": "physical-one",
        "root_id": "logical-one",
        "row_id": row_id,
        "peer_id": "owner",
        "created_at": "2026-01-01T00:00:00+00:00",
        "part_index": index,
        "part_count": count,
        "key": progress.fingerprint([row_id, index]),
        "payload_sha256": "d" * 64,
        "row_hash": "e" * 64,
    }


def pending(chunks, cursors=None):
    """Build a pending batch with explicit cursors for confirmation tests."""
    return {
        "batch_id": "f" * 64,
        "target": "destination",
        "chunks": chunks,
        "cursors": cursors or {},
    }


def test_read_missing_state_does_not_create_parent(tmp_path):
    """Read a missing ledger as empty without creating its parent directory."""
    path = tmp_path / "absent" / "state.json"
    assert progress.read_state(path) == {
        "schema_version": 1,
        "sources": {},
        "scopes": {},
    }
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "data",
    [
        "{broken",
        "[]",
        '{"schema_version": 2, "sources": {}, "scopes": {}}',
        '{"schema_version": true, "sources": {}, "scopes": {}}',
        '{"schema_version": 1}',
    ],
)
def test_corrupt_or_unknown_state_is_not_reset(tmp_path, data):
    """Reject corrupt or unsupported ledgers while preserving the original file."""
    path = tmp_path / "state.json"
    path.write_text(data, encoding="utf-8")
    with pytest.raises(progress.ImportStateError):
        progress.read_state(path)
    assert path.read_text(encoding="utf-8") == data


def test_atomic_write_preserves_original_when_replace_fails(tmp_path, monkeypatch):
    """Keep the last complete ledger if atomic replacement fails."""
    path = tmp_path / "state.json"
    state, scope = valid_state()
    progress.write_state(path, state)
    original = path.read_bytes()
    scope["pending"] = pending([descriptor()])
    monkeypatch.setattr(
        progress.os,
        "replace",
        lambda src, dst: (_ for _ in ()).throw(OSError("fake rename failure")),
    )
    with pytest.raises(OSError):
        progress.write_state(path, state)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".hermes-import-*"))


def test_private_permissions_and_no_payload_in_state(tmp_path):
    """Restrict ledger access and keep message content out of saved progress."""
    state, scope = valid_state()
    scope["pending"] = pending([descriptor()])
    path = tmp_path / "state.json"
    progress.write_state(path, state)
    assert progress.read_state(path) == state
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    assert "content" not in path.read_text(encoding="utf-8")


def test_source_replacement_and_mapping_change_are_rejected():
    """Reject source identity and destination changes against existing checkpoints."""
    state, scope = valid_state()
    with pytest.raises(progress.ImportStateError, match="source database was replaced"):
        progress.get_scope(
            state, scope["identity"], "b" * 64, {"generation": "replacement"}
        )
    with pytest.raises(progress.ImportStateError, match="rules changed"):
        progress.get_segment(scope, "physical-one", "a" * 64)


def test_scopes_include_server_workspace_profile_and_source():
    """Separate progress for each server, workspace, profile, and source identity."""
    state, scope = valid_state()
    identity = scope["identity"]
    for key, value in [
        ("server", "http://other:8000"),
        ("workspace", "other"),
        ("profile", "work"),
        ("source", "b" * 64),
    ]:
        changed = {**identity, key: value}
        assert (
            progress.get_scope(
                state, changed, "b" * 64, {"generation": "test-generation"}
            )
            is None
        )
    assert (
        progress.get_scope(state, identity, "b" * 64, {"generation": "test-generation"})
        is scope
    )


def test_confirmation_does_not_cross_an_incomplete_source_row():
    """Advance a source cursor only after every part of its row is confirmed."""
    state, scope = valid_state()
    first = pending([descriptor(index=0, count=2)])
    scope["pending"] = first
    progress.confirm_batch(scope, first)
    segment = scope["segments"]["physical-one"]
    assert segment["rows"]["1"]["confirmed_parts"] == 1
    assert segment["last_scanned_message_id"] == 0
    second = pending([descriptor(index=1, count=2)], {"physical-one": 1})
    scope["pending"] = second
    progress.confirm_batch(scope, second)
    assert segment["rows"]["1"]["confirmed_parts"] == 2
    assert segment["last_scanned_message_id"] == 1
    assert scope["pending"] is None


def test_changed_selection_resets_cursor_but_keeps_confirmed_hashes():
    """Reconsider the source window without losing confirmed chunk fingerprints."""
    state, scope = valid_state()
    progress.confirm_batch(scope, pending([descriptor()], {"physical-one": 1}))
    original_rows = copy.deepcopy(scope["segments"]["physical-one"]["rows"])
    progress.align_selection(scope, "new-selection")
    assert scope["segments"]["physical-one"]["last_scanned_message_id"] == 0
    assert scope["segments"]["physical-one"]["rows"] == original_rows
    scope["pending"] = pending([descriptor(row_id=2)])
    with pytest.raises(progress.ImportStateError, match="pending batch"):
        progress.align_selection(scope, "another-selection")


@pytest.mark.parametrize(
    "damage", ["hash", "part_index", "duplicate", "cursor", "confirmed_count"]
)
def test_invalid_checkpoint_structure_is_rejected(tmp_path, damage):
    """Reject malformed hashes, chunk counts, and cursor values in saved state."""
    state, scope = valid_state()
    scope["pending"] = pending([descriptor()])
    if damage == "hash":
        scope["pending"]["chunks"][0]["payload_sha256"] = "invalid"
    elif damage == "part_index":
        scope["pending"]["chunks"][0]["part_index"] = 2
    elif damage == "duplicate":
        scope["pending"]["chunks"].append(descriptor())
    elif damage == "cursor":
        scope["segments"]["physical-one"]["last_scanned_message_id"] = -1
    else:
        scope["segments"]["physical-one"]["rows"]["1"] = {
            "row_hash": "e" * 64,
            "part_count": 1,
            "confirmed_parts": 2,
        }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(progress.ImportStateError):
        progress.read_state(path)


def test_second_process_cannot_import_through_the_same_lock(tmp_path):
    """Reject a competing importer while another process owns the ledger lock."""
    path = tmp_path / "state.json"
    script = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('import_progress', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.state_lock(Path(sys.argv[2])):
    print('locked', flush=True)
    sys.stdin.read(1)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(Path(progress.__file__)), str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with (
            pytest.raises(progress.ImportStateError, match="Another importer"),
            progress.state_lock(path, timeout=0),
        ):
            pytest.fail("The second process entered the critical section")
        assert Path(str(path) + ".lock").exists()
    finally:
        child.communicate("q", timeout=5)
    assert child.returncode == 0
    with progress.state_lock(path, timeout=0):
        pass
