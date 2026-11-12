"""Focused regression contracts for migration observation settings."""

from types import SimpleNamespace

import pytest
from test_migration_ledger import _manager, _write_memory_files


def _conversation(manager):
    """Return the cached conversation whose observation settings must be preserved."""
    return manager._sessions_cache["cli-test"]


@pytest.mark.parametrize(
    "kind,origin",
    [
        (kind, origin)
        for kind in ("user", "ai")
        for origin in ("local", "conversation", "peer", "marker")
    ],
)
def test_migration_preserves_each_observation_opt_out(
    tmp_path, monkeypatch, kind, origin
):
    """No configuration layer or retry may turn an explicit opt-out back on."""
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md", "SOUL.md")
    manager, marker = _manager(tmp_path, monkeypatch)
    peer_id = "user" if kind == "user" else "assistant"
    if origin == "local":
        setattr(manager, f"_{kind}_observe_me", False)
    elif origin == "conversation":
        _conversation(manager).get_peer_configuration.side_effect = lambda peer: (
            SimpleNamespace(observe_me=False if peer == peer_id else None)
        )
    elif origin == "peer":
        manager._peers_cache[peer_id].get_configuration.return_value = SimpleNamespace(
            observe_me=False
        )
    else:
        marker.peers.return_value = [SimpleNamespace(id=peer_id)]
        marker.get_peer_configuration.side_effect = lambda peer: SimpleNamespace(
            observe_me=False if peer == peer_id else None
        )
    assert manager.migrate_memory_files("cli:test", str(memory_dir)) is True
    entries = {p: cfg for p, cfg in marker.add_peers.call_args.args[0]}
    assert entries[peer_id].observe_me is False
    assert all(cfg.observe_others is False for cfg in entries.values())
    assert all(cfg.observe_me is not True for cfg in entries.values())


def test_unavailable_observation_policy_does_not_upload(tmp_path, monkeypatch):
    """Policy reads fail closed before any file reaches an unconfigured marker."""
    memory_dir = tmp_path / "memories"
    _write_memory_files(memory_dir, "MEMORY.md")
    manager, marker = _manager(tmp_path, monkeypatch)
    manager._peers_cache["user"].get_configuration.side_effect = RuntimeError(
        "policy unavailable"
    )
    assert manager.migrate_memory_files("cli:test", str(memory_dir)) is False
    marker.upload_file.assert_not_called()
