"""Scopes against a real repository, without contacting Honcho: a chat's session joins its repository's scope
(found from the chat's logical workspace, and shared by every worktree of the repo) and this machine's."""

import subprocess

import pytest

from agent import runtime_cwd
from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import session_scopes


def _git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd, check=True,
                   capture_output=True)


@pytest.fixture
def workspace_repo(monkeypatch, tmp_path):
    """A git repo to scope by, with the process launched from an unrelated directory (the gateway's launch dir)."""
    launch = tmp_path / "launch"
    repo = tmp_path / "project"
    launch.mkdir()
    repo.mkdir()
    _git("init", "-q", str(repo), cwd=tmp_path)
    monkeypatch.chdir(launch)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    return launch, repo


@pytest.mark.parametrize("cwd_source", ["explicit", "context", "terminal"])
def test_chat_joins_its_logical_workspace_repo(monkeypatch, workspace_repo, cwd_source):
    launch, repo = workspace_repo
    child = repo / "src"
    child.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(child if cwd_source == "terminal" else launch))
    token = runtime_cwd.set_session_cwd(str(child) if cwd_source == "context" else None)
    try:
        scopes = HonchoMemoryProvider._resolve_scopes(**({"cwd": str(child)} if cwd_source == "explicit" else {}))
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert scopes == ["project", *session_scopes(None)]


def test_a_worktree_shares_its_repos_scope(workspace_repo):
    _launch, repo = workspace_repo
    _git("commit", "-q", "--allow-empty", "-m", "init", cwd=repo)
    worktree = repo.parent / "project-feature"
    _git("worktree", "add", "-q", str(worktree), cwd=repo)
    assert session_scopes(str(worktree))[0] == "project"


LOCAL_GATEWAY_KEY = "agent:main:local:dm:local-98538a20971158c0308ab21c193d3dbc508f54b85c00b71677ae2a91caaa6f95"


def test_local_gateway_chat_joins_its_pinned_workspace_repo(workspace_repo):
    """A Desktop/CLI chat inside the one-gateway runtime: ``_memory_provider_init_kwargs`` forwards its
    ``gateway_session_key`` and no ``cwd``, while ``policy_scope`` has pinned the session's frozen workspace in
    the cwd contextvar. That workspace's repository is the chat's scope."""
    _launch, repo = workspace_repo
    token = runtime_cwd.set_session_cwd(str(repo))
    try:
        scopes = HonchoMemoryProvider._resolve_scopes(gateway_session_key=LOCAL_GATEWAY_KEY, platform="desktop")
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert scopes == ["project", *session_scopes(None)]


def test_messaging_gateway_chat_joins_only_the_machine(workspace_repo):
    """A Telegram DM runs wherever the gateway started, whatever cwd is pinned: no repository scope."""
    _launch, repo = workspace_repo
    token = runtime_cwd.set_session_cwd(str(repo))
    try:
        scopes = HonchoMemoryProvider._resolve_scopes(
            gateway_session_key="agent:main:telegram:dm:8439114563", platform="telegram")
    finally:
        runtime_cwd._SESSION_CWD.reset(token)
    assert scopes == session_scopes(None)
