"""Per-channel project workspace routing ($HERMES_HOME/honcho-projects.json).

One gateway profile can serve channels that belong to different projects. A mapped session key is served by a child
manager bound to the project's workspace, under a short session name; everything else stays in the configured
workspace. These tests pin the matching rule (terminal segment, longest match), mtime reload, the unmapped fallback,
that the parent config is never mutated, that children acquire clients through the one cached/refreshing
``get_honcho_client(config)`` path, that writes follow the session's workspace stamp across manager instances, and
that shutdown drains the children's lazily-started writers.

Run with a Hermes checkout on PYTHONPATH:

    python -m pytest -c hermes-plugin-honcho/pyproject.toml \\
        --confcutdir=hermes-plugin-honcho/tests hermes-plugin-honcho/tests -q
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_honcho import client as client_mod
from hermes_honcho.client import HonchoClientConfig, get_honcho_client, reset_honcho_client
from hermes_honcho.session import HonchoSession, HonchoSessionManager

_MAP = {
    "projects": {
        "myproject": {"sessions": {
            "telegram-group--100123456789-1": "telegram-topic-one",
            "slack-group-C0EXAMPLE123": "slack",
        }},
        "otherproject": {"sessions": {"telegram-group--100123456789-1578": "telegram"}},
    }
}


@pytest.fixture(autouse=True)
def _fresh_clients():
    reset_honcho_client()
    yield
    reset_honcho_client()


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])  # tmp dir per test (conftest)


def _write_map(payload, home: Path | None = None) -> Path:
    path = (home or _home()) / "honcho-projects.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def _config(write_frequency="turn", **kw) -> HonchoClientConfig:
    # "turn" keeps writes synchronous; peer_name gives every session a user peer.
    return HonchoClientConfig(api_key="test-key", peer_name="alice", write_frequency=write_frequency, **kw)


def _manager(**kw) -> HonchoSessionManager:
    return HonchoSessionManager(config=_config(**kw))


@contextmanager
def _clients(default=None, by_workspace=None):
    """Patch the one acquisition seam the manager uses, dispatching on the config's workspace_id: a test asserting
    a child never touched the default client is then asserting the routing, not the mock wiring."""
    default = default if default is not None else MagicMock(name="default-client")
    by_workspace = by_workspace or {}
    with patch("hermes_honcho.session.get_honcho_client",
               side_effect=lambda config=None: by_workspace.get(getattr(config, "workspace_id", None), default)) as acquire:
        yield acquire


def _stamped_session() -> HonchoSession:
    session = HonchoSession(key="slack", user_peer_id="alice", assistant_peer_id="hermes",
                            honcho_session_id="slack", workspace="myproject")
    session.add_message("user", "hello there")
    return session


class TestMatching:
    def test_terminal_match(self):
        _write_map(_MAP)
        assert _manager()._match_project_route("telegram:group:-100123456789:1578") == ("otherproject", "telegram")

    def test_topic_id_prefix_does_not_collide(self):
        # Plain substring matching would send topic 1578 into topic 1's project.
        _write_map(_MAP)
        mgr = _manager()
        assert mgr._match_project_route("telegram:group:-100123456789:1") == ("myproject", "telegram-topic-one")
        assert mgr._match_project_route("telegram:group:-100123456789:1578") == ("otherproject", "telegram")

    def test_pattern_followed_by_separator_matches(self):
        _write_map(_MAP)
        assert _manager()._match_project_route("slack:group:C0EXAMPLE123:thread-4567") == ("myproject", "slack")

    def test_pattern_mid_key_without_separator_does_not_match(self):
        _write_map(_MAP)
        assert _manager()._match_project_route("slack:group:C0EXAMPLE123999") is None

    def test_longest_pattern_wins(self):
        _write_map({"projects": {
            "broad": {"sessions": {"group--100123456789-1": "broad"}},
            "narrow": {"sessions": {"telegram-group--100123456789-1": "narrow"}},
        }})
        assert _manager()._match_project_route("telegram:group:-100123456789:1") == ("narrow", "narrow")

    def test_unmapped_key_does_not_route(self):
        _write_map(_MAP)
        assert _manager()._match_project_route("discord:999888777") is None

    def test_no_mapping_file_does_not_route(self):
        assert _manager()._match_project_route("telegram:group:-100123456789:1") is None

    def test_configless_manager_does_not_route(self):
        _write_map(_MAP)
        assert HonchoSessionManager()._match_project_route("telegram:group:-100123456789:1") is None

    def test_child_does_not_route_again(self):
        _write_map(_MAP)
        child = _manager()._project_manager("myproject")
        assert child._match_project_route("telegram:group:-100123456789:1") is None

    def test_malformed_file_warns_and_routes_nothing(self, caplog):
        _write_map("{not json")
        with caplog.at_level(logging.WARNING):
            assert _manager()._match_project_route("telegram:group:-100123456789:1") is None
        assert any("malformed" in r.getMessage().lower() for r in caplog.records)

    def test_reloads_on_mtime_change(self):
        path = _write_map(_MAP)
        mgr = _manager()
        assert mgr._match_project_route("discord:999888777") is None
        _write_map({"projects": {"myproject": {"sessions": {"discord-999888777": "discord"}}}})
        st = path.stat()
        os.utime(path, (st.st_atime + 10, st.st_mtime + 10))  # visible regardless of filesystem granularity
        assert mgr._match_project_route("discord:999888777") == ("myproject", "discord")

    def test_mapping_is_read_from_the_configs_bound_hermes_home(self, tmp_path):
        # Daemon threads can't see the ambient profile, so the bound home wins over $HERMES_HOME.
        profile_home = tmp_path / "profile"
        _write_map(_MAP, home=profile_home)
        assert _manager()._match_project_route("slack:group:C0EXAMPLE123") is None
        assert _manager(hermes_home=profile_home)._match_project_route("slack:group:C0EXAMPLE123") == ("myproject", "slack")


class TestRoutedSessions:
    def test_mapped_key_lands_in_project_workspace_under_short_name(self):
        _write_map(_MAP)
        default, project = MagicMock(), MagicMock()
        mgr = _manager()
        with _clients(default, {"myproject": project}) as acquire:
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")
        assert (session.workspace, session.key, session.honcho_session_id) == ("myproject", "slack", "slack")
        project.session.assert_called_once_with("slack")
        default.session.assert_not_called()
        assert {c.args[0].workspace_id for c in acquire.call_args_list} == {"myproject"}

    def test_repeat_lookup_hits_the_childs_cache(self):
        _write_map(_MAP)
        project = MagicMock()
        mgr = _manager()
        with _clients(by_workspace={"myproject": project}):
            assert mgr.get_or_create("slack:group:C0EXAMPLE123") is mgr.get_or_create("slack:group:C0EXAMPLE123")
        project.session.assert_called_once()

    def test_unmapped_key_uses_the_configured_workspace(self):
        _write_map(_MAP)
        default, project = MagicMock(), MagicMock()
        mgr = _manager()
        with _clients(default, {"myproject": project}) as acquire:
            session = mgr.get_or_create("discord:999888777")
        assert (session.workspace, session.key) == (None, "discord:999888777")
        default.session.assert_called_once_with("discord-999888777")
        project.session.assert_not_called()
        assert {c.args[0].workspace_id for c in acquire.call_args_list} == {"hermes"}
        assert not mgr._project_children()

    def test_prefetch_cache_follows_the_route(self):
        _write_map(_MAP)
        mgr = _manager()
        mgr.set_context_result("slack:group:C0EXAMPLE123", {"representation": "r"})
        assert mgr._project_manager("myproject")._context_cache == {"slack": {"representation": "r"}}
        assert mgr.pop_context_result("slack:group:C0EXAMPLE123") == {"representation": "r"}
        assert mgr._context_cache == {}

    @pytest.mark.parametrize("name", [
        "get_or_create", "prefetch_context", "set_context_result", "pop_context_result", "get_prefetch_context",
        "get_session_context", "get_peer_card", "search_context", "create_conclusion", "delete_conclusion",
        "list_conclusions", "set_peer_card", "seed_ai_identity", "get_ai_representation", "dialectic_query",
        "migrate_memory_files",
    ])
    def test_every_session_key_method_is_routed(self, name):
        assert getattr(getattr(HonchoSessionManager, name), "__wrapped__", None) is not None


class TestChildClientIdentity:
    def test_child_config_carries_the_workspace_and_parent_config_is_not_mutated(self):
        mgr = _manager()
        parent_config = mgr._config
        child = mgr._project_manager("myproject")
        assert child._config.workspace_id == "myproject"
        assert child._config is not parent_config
        assert mgr._config is parent_config and parent_config.workspace_id == "hermes"

    def test_separate_workspaces_get_separate_children(self):
        mgr = _manager()
        a, b = mgr._project_manager("myproject"), mgr._project_manager("otherproject")
        assert a is not b and a is mgr._project_manager("myproject")
        assert {a._config.workspace_id, b._config.workspace_id} == {"myproject", "otherproject"}

    def test_child_acquires_through_the_shared_seam_on_every_access(self):
        # No pinned client: every access re-acquires (OAuth refresh lives there) with the child's own config.
        default, project = MagicMock(), MagicMock()
        mgr = _manager()
        child = mgr._project_manager("myproject")
        with _clients(default, {"myproject": project}) as acquire:
            assert child.honcho is project
            assert child.honcho is project
            assert mgr.honcho is default
        assert acquire.call_count == 3
        assert all(c.args[0] is child._config for c in acquire.call_args_list[:2])

    def test_parent_and_child_clients_are_both_cached(self):
        # Alternating workspaces in one profile must not evict each other's cached client.
        builds = []

        def _build(config):
            builds.append(config.workspace_id)
            return MagicMock(name=f"client-{config.workspace_id}")

        mgr = _manager()
        child = mgr._project_manager("myproject")
        with patch.object(client_mod, "_build_client", side_effect=_build):
            first_parent, first_child = get_honcho_client(mgr._config), get_honcho_client(child._config)
            assert get_honcho_client(mgr._config) is first_parent
            assert get_honcho_client(child._config) is first_child
        assert first_parent is not first_child
        assert builds == ["hermes", "myproject"]


class TestWriteRouting:
    def test_flush_through_another_instance_follows_the_stamp(self):
        # The gateway runs several manager instances; any of them must flush a routed session into its workspace.
        _write_map(_MAP)
        session = _stamped_session()
        default, project = MagicMock(), MagicMock()
        other = _manager()
        with _clients(default, {"myproject": project}):
            assert other._flush_session(session) is True
        project.session.assert_called_once_with("slack")
        default.session.assert_not_called()
        assert all(m["_synced"] for m in session.messages)

    def test_save_follows_the_stamp(self):
        _write_map(_MAP)
        session = _stamped_session()
        default, project = MagicMock(), MagicMock()
        with _clients(default, {"myproject": project}):
            _manager().save(session)
        project.session.assert_called_once_with("slack")
        default.session.assert_not_called()
        assert all(m["_synced"] for m in session.messages)

    def test_unstamped_session_writes_to_the_configured_workspace(self):
        session = HonchoSession(key="discord:999888777", user_peer_id="alice", assistant_peer_id="hermes",
                                honcho_session_id="discord-999888777")
        session.add_message("user", "hi")
        default = MagicMock()
        mgr = _manager()
        with _clients(default):
            assert mgr._flush_session(session) is True
        default.session.assert_called_once_with("discord-999888777")
        assert not mgr._project_children()

    def test_flush_all_fans_out_to_children(self):
        _write_map(_MAP)
        project = MagicMock()
        mgr = _manager(write_frequency="session")
        with _clients(by_workspace={"myproject": project}):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")
            session.add_message("user", "deferred until flush_all")
            mgr.save(session)
            assert not session.messages[-1].get("_synced")
            mgr.flush_all()
        assert session.messages[-1]["_synced"] is True
        project.session.return_value.add_messages.assert_called_once()


class TestShutdown:
    def test_shutdown_drains_a_childs_lazily_started_writer(self):
        _write_map(_MAP)
        project = MagicMock()
        mgr = _manager(write_frequency="async")
        with _clients(by_workspace={"myproject": project}):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")
            child = mgr._project_manager("myproject")
            assert mgr._async_thread is None and child._async_thread is None  # lazy: nothing enqueued yet
            session.add_message("user", "hello there")
            mgr.save(session)  # enqueued on the child, which starts its writer
            assert mgr._async_thread is None and child._async_thread is not None
            # Registered to the root manager, so the provider's join_plugin_threads((provider, manager)) waits on it.
            assert child._async_thread in set(client_mod._plugin_threads.get(mgr, ()))
            mgr.shutdown(timeout=5.0)
        assert not child._async_thread.is_alive()
        assert child._shutting_down and mgr._shutting_down
        assert all(m["_synced"] for m in session.messages)

    def test_child_first_reached_after_shutdown_flushes_inline(self):
        _write_map(_MAP)
        project = MagicMock()
        mgr = _manager(write_frequency="async")
        mgr.shutdown(timeout=1.0)
        with _clients(by_workspace={"myproject": project}):
            session = mgr.get_or_create("slack:group:C0EXAMPLE123")
            session.add_message("user", "late")
            mgr.save(session)
        child = mgr._project_manager("myproject")
        assert child._async_thread is None
        assert session.messages[-1]["_synced"] is True
