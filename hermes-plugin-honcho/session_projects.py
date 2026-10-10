"""Opt-in per-channel workspace routing for HonchoSessionManager, driven by $HERMES_HOME/honcho-projects.json.

One profile's manager normally writes every session into the one ``workspace`` its config names. A gateway profile
that serves several channels belonging to different projects can map those channels to per-project workspaces:

    {"projects": {"myproject": {"sessions": {
        "telegram-group--100123456789-42": "telegram",
        "slack-group-C0EXAMPLE123": "slack"}}}}

A matched session key is served by a child manager whose config is a copy with ``workspace_id`` swapped for the
project workspace, under the short session name. The child acquires its client through the ordinary
``get_honcho_client(config)``, so caching and OAuth refresh stay on the one path the default workspace uses. With no
mapping file nothing routes and the manager behaves exactly as before.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger("plugins.memory.honcho.session")

PROJECT_MAP_FILENAME = "honcho-projects.json"

_F = TypeVar("_F", bound=Callable[..., Any])


def project_routed(method: _F) -> _F:
    """Serve a session-key-first manager method from the project child the key maps to, under the short session
    name. Unmapped keys, and installs without a mapping file, call the method unchanged."""

    @functools.wraps(method)
    def routed(self, key, *args, **kwargs):
        route = self._match_project_route(key)
        if route is None:
            return method(self, key, *args, **kwargs)
        workspace, short_name = route
        return method(self._project_manager(workspace), short_name, *args, **kwargs)

    return routed  # type: ignore[return-value]


class SessionProjectsMixin:
    # Class-level defaults, so a manager behaves as an unrouted root until routing first runs.
    _project_workspace: str | None = None  # set on children: the workspace this manager was routed to
    _project_parent: Any = None  # set on children: the manager that created them
    _project_managers: dict[str, Any] | None = None
    _project_map_cache: tuple[int, dict[str, tuple[str, str]]] | None = None

    def _project_routing_enabled(self) -> bool:
        """Only a root manager with a dataclass config routes: children never route again, and a child config
        is built with dataclasses.replace() so the parent's config is never mutated."""
        return self._project_workspace is None and dataclasses.is_dataclass(self._config)

    def _project_map_path(self) -> Path:
        """The mapping file of this manager's profile. Bound configs carry their HERMES_HOME because the ambient
        resolver reads a ContextVar background threads can't see."""
        home = getattr(self._config, "hermes_home", None)
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return Path(home) / PROJECT_MAP_FILENAME

    def _project_session_map(self) -> dict[str, tuple[str, str]]:
        """pattern -> (workspace, short session name), reloaded whenever the file's mtime changes, so projects can be
        added without a restart. A missing file routes nothing; a malformed one warns once per change and routes
        nothing."""
        path = self._project_map_path()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return {}
        cached = self._project_map_cache
        if cached is not None and cached[0] == mtime:
            return cached[1]
        mapping: dict[str, tuple[str, str]] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for workspace, block in (raw.get("projects") or {}).items():
                for pattern, short_name in ((block or {}).get("sessions") or {}).items():
                    if workspace and pattern and short_name:
                        mapping[str(pattern)] = (str(workspace), str(short_name))
        except Exception as e:
            logger.warning("Ignoring malformed %s: %s", path, e)
            mapping = {}
        self._project_map_cache = (mtime, mapping)
        return mapping

    def _match_project_route(self, key: str) -> tuple[str, str] | None:
        """(workspace, short session name) for a mapped key, else None.

        A pattern matches the sanitized key as a terminal segment: the key ends with it, or it is followed by a
        ``-`` (a thread suffix). Plain substring matching would collide on numeric prefixes: a pattern ending in
        ``-1`` (topic 1) must not capture a key ending in ``-1578`` (topic 1578). The longest match wins, so a
        specific mapping shadows a broader one."""
        if not self._project_routing_enabled():
            return None
        mapping = self._project_session_map()
        if not mapping:
            return None
        sanitized = self._sanitize_id(str(key))
        best: tuple[str, tuple[str, str]] | None = None
        for pattern, target in mapping.items():
            if (sanitized.endswith(pattern) or f"{pattern}-" in sanitized) and (best is None or len(pattern) > len(best[0])):
                best = (pattern, target)
        return best[1] if best else None

    def _project_manager(self, workspace: str) -> Any:
        """This manager's child for ``workspace``, created on first use from config alone, so any manager instance
        builds an equivalent child. No client is passed: the child's ``honcho`` property acquires one through
        ``get_honcho_client(config)`` with the routed ``workspace_id``, which is part of the client cache identity."""
        with self._cache_lock:
            managers = self._project_managers
            if managers is None:
                managers = self._project_managers = {}
            child = managers.get(workspace)
            if child is None:
                child = type(self)(
                    context_tokens=self._context_tokens, config=dataclasses.replace(self._config, workspace_id=workspace),
                    runtime_user_peer_name=self._runtime_user_peer_name,
                    runtime_user_peer_name_alt=self._runtime_user_peer_name_alt,
                )
                child._project_workspace = workspace
                child._project_parent = self
                # A child first reached after shutdown began flushes inline instead of starting a writer.
                child._shutting_down = self._shutting_down
                managers[workspace] = child
                logger.info("Honcho project workspace '%s' routing active", workspace)
        return child

    def _project_owner(self, session: Any) -> Any:
        """The manager that writes ``session``: its project child when the session was created by routing (stamped
        with its workspace, so a flush issued through any manager instance lands in the right workspace), else
        this manager."""
        workspace = getattr(session, "workspace", None)
        if not workspace or not self._project_routing_enabled():
            return self
        return self._project_manager(workspace)

    def _project_children(self) -> list[Any]:
        with self._cache_lock:
            return list((self._project_managers or {}).values())

    def _fan_out_to_project_children(self, method: str, timeout: float | None) -> float | None:
        """Call ``method(timeout=...)`` on every project child within ``timeout``; returns what is left of it."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for child in self._project_children():
            try:
                getattr(child, method)(timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
            except Exception as e:
                logger.error("Honcho %s failed for project workspace '%s': %s", method, child._project_workspace, e)
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    def _thread_owner(self) -> Any:
        """Threads a child spawns are registered to the root manager, so the provider's shutdown joins them too."""
        return self._project_parent if self._project_parent is not None else self
