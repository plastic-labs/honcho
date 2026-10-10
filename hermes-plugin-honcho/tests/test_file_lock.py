"""Contracts for reusable portable file-lock ownership."""

import pytest
from hermes_honcho.file_lock import file_lock


def test_file_lock_releases_after_normal_exit(tmp_path):
    lock_path = tmp_path / "operation.lock"
    with file_lock(lock_path, timeout=0) as acquired:
        assert acquired is True
    with file_lock(lock_path, timeout=0) as acquired:
        assert acquired is True


def test_file_lock_releases_after_exception(tmp_path):
    lock_path = tmp_path / "operation.lock"
    with (
        pytest.raises(RuntimeError, match="operation failed"),
        file_lock(lock_path, timeout=0) as acquired,
    ):
        assert acquired is True
        raise RuntimeError("operation failed")
    with file_lock(lock_path, timeout=0) as acquired:
        assert acquired is True


def test_file_lock_contention_reports_not_acquired(tmp_path):
    lock_path = tmp_path / "operation.lock"
    with file_lock(lock_path, timeout=0) as acquired:
        assert acquired is True
        with file_lock(lock_path, timeout=0) as contender_acquired:
            assert contender_acquired is False
