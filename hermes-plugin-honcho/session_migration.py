"""One-time upload of local memory files (MEMORY.md / USER.md / SOUL.md) into Honcho."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from utils import atomic_json_write

from .client import resolve_effective_base_url
from .file_lock import file_lock
from .session_auth import HonchoAuthError

logger = logging.getLogger("plugins.memory.honcho.session")

_MIGRATION_LOCK_TIMEOUT_SECONDS = 5.0
_MIGRATION_THREAD_LOCK = threading.Lock()
_HONCHO_ENVIRONMENT_ENDPOINTS = {
    "local": "http://localhost:8000",
    "production": "https://api.honcho.dev",
}

# (filename, upload name, description, peer kind) — peer kind picks user vs assistant peer.
_MEMORY_FILES = (
    (
        "MEMORY.md",
        "consolidated_memory.md",
        "Long-term agent notes and preferences",
        "user",
    ),
    ("USER.md", "user_profile.md", "User profile and preferences", "user"),
    ("SOUL.md", "agent_soul.md", "Agent persona and identity configuration", "ai"),
)


@contextmanager
def _migration_lock(lock_path: Path):
    """Bound concurrent migration attempts across threads and processes."""
    if not _MIGRATION_THREAD_LOCK.acquire(timeout=_MIGRATION_LOCK_TIMEOUT_SECONDS):
        logger.warning("Honcho migration skipped: timed out acquiring thread lock")
        yield False
        return

    try:
        with file_lock(lock_path, _MIGRATION_LOCK_TIMEOUT_SECONDS) as acquired:
            yield acquired
    finally:
        _MIGRATION_THREAD_LOCK.release()


def _load_migration_state(state_path: Path) -> dict[str, Any] | None:
    """Load valid migration state, failing closed on unreadable state."""
    if not state_path.exists():
        return {"version": 1, "targets": {}}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Honcho migration state is unreadable: %s", exc)
        return None
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or not isinstance(state.get("targets"), dict)
    ):
        logger.warning(
            "Honcho migration state has an unsupported format: %s", state_path
        )
        return None
    return state


def _migration_completed_files(
    state: dict[str, Any],
    target_key: str,
    target: dict[str, str],
    source_key: str,
    source_path: str,
) -> dict[str, Any] | None:
    """Initialize and validate the target/source/file ledger mappings."""
    current = state["targets"]
    for key, default, kind, identity in (
        (target_key, {**target, "sources": {}}, "target", target_key),
        ("sources", {}, "target", target_key),
        (source_key, {"path": source_path, "files": {}}, "source", source_path),
        ("files", None, "source", source_path),
    ):
        current = (
            current.setdefault(key, default)
            if default is not None
            else current.get(key)
        )
        if not isinstance(current, dict):
            logger.warning("Honcho migration %s state is invalid: %s", kind, identity)
            return None
    return current


class SessionMigrationMixin:
    def _migration_marker_session(self, session, marker_id: str, target_key: str):
        """Keep explicit observation opt-outs without enabling cross-peer observation."""
        from honcho.session import SessionPeerConfig

        marker = self._sdk_session(
            marker_id,
            metadata={"source": "hermes_memory_migration", "target_key": target_key},
        )
        member_ids = {peer.id for peer in marker.peers()}
        conversation = self._sdk_session(session.honcho_session_id)
        flags = self._observation_flags(session.honcho_session_id)
        entries = []
        for kind, peer_id in (
            ("user", session.user_peer_id),
            ("ai", session.assistant_peer_id),
        ):
            peer = self._get_or_create_peer(peer_id)
            policies = [
                getattr(self, f"_{kind}_observe_me"),
                flags[f"{kind}_observe_me"],
                conversation.get_peer_configuration(peer_id).observe_me,
                peer.get_configuration().observe_me,
            ]
            if peer_id in member_ids:
                policies.append(marker.get_peer_configuration(peer_id).observe_me)
            # Inherit server defaults rather than overriding them with True.
            entries.append(
                (
                    peer_id,
                    SessionPeerConfig(
                        observe_me=False
                        if any(value is False for value in policies)
                        else None,
                        observe_others=False,
                    ),
                )
            )
        marker.add_peers(entries)
        return marker

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        """Upload local memory files once per source and Honcho destination.

        A local ledger prevents repeat uploads across sessions. Before each upload,
        a deterministic remote migration marker is reconciled so a successful
        upload followed by a failed ledger write cannot produce a duplicate.
        """
        memory_path = Path(memory_dir).expanduser().resolve()
        if not memory_path.exists():
            return False

        session = self._cached_session(session_key)
        if not session:
            logger.warning(
                "No local session cached for '%s', skipping memory migration",
                session_key,
            )
            return False
        if session.honcho_session_id not in self._sessions_cache:
            logger.warning(
                "No Honcho session cached for '%s', skipping memory migration",
                session_key,
            )
            return False

        # Owner-scoped: these files describe the install owner; uploading them under another
        # human's peer would make Honcho attribute the owner's facts to that person. The owner is
        # the CONFIG peerName, never a re-resolution of the session's own peer (that would compare
        # the triggering user to themselves). No declared owner: nobody can be proven to be the owner.
        owner_peer_id = self._declared_owner_peer_id()
        if owner_peer_id is None or session.user_peer_id != owner_peer_id:
            logger.info(
                "Skipping memory-file migration: session user peer '%s' is not the declared owner (peerName=%s)",
                session.user_peer_id,
                owner_peer_id or "unset",
            )
            return False

        config = self._config
        endpoint = resolve_effective_base_url(config) if config is not None else None
        environment = config.environment if config is not None else "production"
        endpoint_key = endpoint or _HONCHO_ENVIRONMENT_ENDPOINTS.get(
            environment,
            f"environment:{environment}",
        )
        workspace_id = config.workspace_id if config is not None else "hermes"
        target = {
            "endpoint": endpoint_key,
            "workspace_id": workspace_id,
            "user_peer_id": session.user_peer_id,
            "ai_peer_id": session.assistant_peer_id,
        }
        target_json = json.dumps(target, sort_keys=True, separators=(",", ":"))
        target_key = hashlib.sha256(target_json.encode("utf-8")).hexdigest()
        remote_marker_session_id = f"hermes-memory-migration-{target_key}"
        source_path = str(memory_path)
        source_key = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
        hermes_home = (
            config.hermes_home
            if config is not None and config.hermes_home is not None
            else get_hermes_home()
        )
        state_path = hermes_home / "state" / "honcho_migration.json"
        lock_path = state_path.with_suffix(".json.lock")

        with _migration_lock(lock_path) as acquired:
            if not acquired:
                return False
            state = _load_migration_state(state_path)
            if state is None:
                return False

            completed_files = _migration_completed_files(
                state, target_key, target, source_key, source_path
            )
            if completed_files is None:
                return False

            marker_ready = False
            uploaded = False
            for filename, upload_name, description, target_kind in _MEMORY_FILES:
                if completed_files.get(filename) is True:
                    continue
                filepath = memory_path / filename
                content = (
                    filepath.read_text(encoding="utf-8-sig").strip()
                    if filepath.exists()
                    else ""
                )
                if not content:
                    continue

                migration_id = hashlib.sha256(
                    json.dumps(
                        {
                            "version": 1,
                            "target": target_key,
                            "source": source_key,
                            "filename": filename,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                migration_filter = {"metadata": {"migration_id": migration_id}}

                try:
                    if not marker_ready:
                        self._authed_call(
                            "memory migration marker session setup",
                            lambda: self._migration_marker_session(
                                session,
                                remote_marker_session_id,
                                target_key,
                            ),
                        )
                        marker_ready = True
                    remote_messages = self._authed_call(
                        "memory migration reconciliation",
                        lambda migration_filter=migration_filter: self._sdk_session(
                            remote_marker_session_id
                        ).messages(
                            filters=migration_filter,
                            page=1,
                            size=1,
                        ),
                    )
                except HonchoAuthError:
                    logger.warning(
                        "Honcho memory migration stopped before %s: auth failed",
                        filename,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Failed to reconcile Honcho migration for %s; skipping upload: %s",
                        filename,
                        exc,
                    )
                    continue

                if len(remote_messages) > 0:
                    completed_files[filename] = True
                    try:
                        atomic_json_write(state_path, state, mode=0o600, sort_keys=True)
                    except Exception as exc:
                        logger.warning(
                            "Failed to record reconciled Honcho migration for %s: %s",
                            filename,
                            exc,
                        )
                        break
                    logger.info("Reconciled prior Honcho migration for %s", filename)
                    continue

                wrapped = (
                    "<prior_memory_file>\n<context>\n"
                    "This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                    f"{description}. Treat as foundational context for this user.\n"
                    f"</context>\n\n{content}\n</prior_memory_file>\n"
                )
                target_peer_id = (
                    session.user_peer_id
                    if target_kind == "user"
                    else session.assistant_peer_id
                )

                try:

                    def _upload(
                        upload_name=upload_name,
                        wrapped=wrapped,
                        target_peer_id=target_peer_id,
                        filename=filename,
                        target_kind=target_kind,
                        migration_id=migration_id,
                    ) -> None:
                        self._sdk_session(remote_marker_session_id).upload_file(
                            file=(upload_name, wrapped.encode("utf-8"), "text/plain"),
                            peer=self._get_or_create_peer(target_peer_id),
                            metadata={
                                "source": "local_memory",
                                "original_file": filename,
                                "target_peer": target_kind,
                                "migration_id": migration_id,
                            },
                        )

                    self._authed_call("memory migration upload", _upload)
                except HonchoAuthError:
                    logger.warning(
                        "Honcho memory migration stopped after %s: auth failed",
                        filename,
                    )
                    break
                except Exception as exc:
                    logger.error("Failed to upload %s to Honcho: %s", filename, exc)
                    continue

                uploaded = True
                completed_files[filename] = True
                try:
                    atomic_json_write(state_path, state, mode=0o600, sort_keys=True)
                except Exception as exc:
                    logger.warning(
                        "Failed to record Honcho migration for %s: %s", filename, exc
                    )
                    break
                logger.info(
                    "Uploaded %s to Honcho for %s (%s peer)",
                    filename,
                    session_key,
                    target_kind,
                )

            return uploaded
