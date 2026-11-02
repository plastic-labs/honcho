"""Route against real repository/SessionDB state, without contacting Honcho."""

import subprocess

import pytest

from agent import runtime_cwd
from hermes_state import SessionDB
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig


@pytest.mark.parametrize("cwd_source", ["explicit", "context", "terminal"])
def test_provider_routes_by_logical_workspace(monkeypatch, tmp_path, cwd_source):
    launch = tmp_path / "launch"
    repo = tmp_path / "project"
    launch.mkdir()
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    child = repo / "src"
    child.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv("TERMINAL_CWD", str(child if cwd_source == "terminal" else launch))
    token = runtime_cwd.set_session_cwd(str(child) if cwd_source == "context" else None)
    cfg = HonchoClientConfig(session_strategy="per-repo", sessions={str(launch): "wrong-bucket"})
    try:
        key = HonchoMemoryProvider()._resolve_session_key(
            cfg, "conversation", **({"cwd": str(child)} if cwd_source == "explicit" else {})
        )
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert key == repo.name


@pytest.fixture
def workspace_repo(monkeypatch, tmp_path):
    """A git repo to route by, with the process launched from an unrelated directory (the gateway's
    launch dir), which the sessions map deliberately points at the wrong bucket."""
    launch = tmp_path / "launch"
    repo = tmp_path / "project"
    launch.mkdir()
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(launch)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    return launch, repo


LOCAL_GATEWAY_KEY = "agent:main:local:dm:local-98538a20971158c0308ab21c193d3dbc508f54b85c00b71677ae2a91caaa6f95"


def test_local_gateway_session_routes_by_strategy(workspace_repo):
    """A Desktop/CLI chat inside the one-gateway runtime: ``_memory_provider_init_kwargs`` forwards its
    ``gateway_session_key`` and no ``cwd`` (the agent is built without one), while ``policy_scope``
    has pinned the session's frozen workspace in the cwd contextvar. ``per-repo`` must win."""
    launch, repo = workspace_repo
    cfg = HonchoClientConfig(session_strategy="per-repo", sessions={str(launch): "wrong-bucket"})
    token = runtime_cwd.set_session_cwd(str(repo))
    try:
        key = HonchoMemoryProvider()._resolve_session_key(
            cfg, "local-98538a20971158", gateway_session_key=LOCAL_GATEWAY_KEY, platform="desktop",
        )
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert key == repo.name


def test_local_gateway_session_without_workspace_lands_in_launch_dir_bucket(workspace_repo):
    """No pinned workspace (a chat opened from Home): the launch directory's mapping applies, which is
    what a non-gateway Desktop chat with the same cwd gets. Never a per-chat ``agent-main-local-...``."""
    launch, _repo = workspace_repo
    cfg = HonchoClientConfig(session_strategy="per-repo", sessions={str(launch): "home-bucket"})
    key = HonchoMemoryProvider()._resolve_session_key(
        cfg, "local-98538a20971158", gateway_session_key=LOCAL_GATEWAY_KEY, platform="desktop",
    )
    assert key == "home-bucket"


def test_served_desktop_session_routes_by_strategy(workspace_repo):
    """A Desktop client attached to ``hermes serve`` reaches the agent through the TUI gateway: an
    explicit construction ``cwd`` and no gateway key. Unchanged by the LOCAL-key fall-through."""
    launch, repo = workspace_repo
    cfg = HonchoClientConfig(session_strategy="per-repo", sessions={str(launch): "wrong-bucket"})
    key = HonchoMemoryProvider()._resolve_session_key(cfg, "20260415_a3f2b1", cwd=str(repo), platform="desktop")
    assert key == repo.name


def test_messaging_gateway_session_keeps_per_chat_isolation(workspace_repo):
    """A Telegram DM through the same gateway still gets its own session, whatever cwd is pinned."""
    launch, repo = workspace_repo
    cfg = HonchoClientConfig(session_strategy="per-repo", sessions={str(launch): "wrong-bucket"})
    token = runtime_cwd.set_session_cwd(str(repo))
    try:
        key = HonchoMemoryProvider()._resolve_session_key(
            cfg, "20260415_a3f2b1", gateway_session_key="agent:main:telegram:dm:8439114563", platform="telegram",
        )
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert key == "agent-main-telegram-dm-8439114563"
