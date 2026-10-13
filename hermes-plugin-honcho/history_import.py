"""Deterministic historical conversation import, independent of the live writer."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent.memory_manager import sanitize_context
from hermes_constants import get_hermes_home

from .client import (
    _DEFAULT_HTTP_TIMEOUT,
    HonchoClientConfig,
    _host_block,
    _read_config,
    _resolve_optional_float,
)
from .import_state import (
    ImportStateError,
    align_selection,
    confirm_batch,
    fingerprint,
    get_scope,
    get_segment,
    read_state,
    restrict_private_file,
    state_lock,
    write_state,
)
from .session_peers import (
    BOT_AUTHOR_PREFIX,
    SessionPeersMixin,
    a2a_session_key_for,
    assistant_peer_id_for,
)

IMPORT_VERSION = 1
PAGE_SIZE = 200
BATCH_SIZE = 100


class ImportInputError(ValueError):
    """An input cannot be interpreted without guessing a source or destination."""


@dataclass(frozen=True)
class ImportOptions:
    days: int = 30
    sources: tuple[str, ...] = ("cli", "desktop")
    before: datetime | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cutoff(self) -> datetime:
        return self.now - timedelta(days=self.days)

    @property
    def selection(self) -> str:
        return fingerprint(
            {
                "version": IMPORT_VERSION,
                "days": self.days,
                "sources": sorted(self.sources),
                "before": self.before.isoformat() if self.before else None,
            }
        )


def parse_options(args: Any) -> ImportOptions:
    from gateway.config import Platform

    days = getattr(args, "days", 30)
    if type(days) is not int or days <= 0:
        raise ImportInputError("--days must be a positive integer.")
    try:
        timedelta(days=days)
    except OverflowError as exc:
        raise ImportInputError("--days is too large.") from exc
    sources = tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in getattr(args, "source", "cli,desktop").split(",")
            if part.strip()
        )
    )
    if not sources:
        raise ImportInputError("--source must contain at least one source.")
    for source in sources:
        if source not in {"cli", "desktop", "subagent"}:
            try:
                Platform(source)
            except ValueError as exc:
                raise ImportInputError(
                    "--source contains an unknown Hermes source."
                ) from exc
    before = None
    if raw_before := getattr(args, "before", None):
        try:
            before = datetime.fromisoformat(raw_before.replace("Z", "+00:00"))
            if before.tzinfo is None or before.utcoffset() is None:
                raise ValueError("timezone missing")
            before = before.astimezone(UTC)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ImportInputError(
                "--before must be an ISO 8601 timestamp with a timezone."
            ) from exc
    return ImportOptions(days=days, sources=sources, before=before)


def normalize_endpoint(value: str, *, strip_version: bool = True) -> str:
    """Match the SDK's version-prefix handling; reject credentials in a preview URL."""
    try:
        if (
            not isinstance(value, str)
            or not value
            or any(ord(c) < 32 or ord(c) > 126 for c in value)
        ):
            raise ValueError("invalid address")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("invalid address")
        if parsed.query or parsed.fragment:
            raise ValueError("query or fragment")
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        netloc = (
            host
            if port is None or (parsed.scheme, port) in {("http", 80), ("https", 443)}
            else f"{host}:{port}"
        )
        path = re.sub(r"/v\d+/*$", "", parsed.path) if strip_version else parsed.path
        path = path.rstrip("/")
        return urlunsplit((parsed.scheme, netloc, path, "", ""))
    except (ValueError, TypeError) as exc:
        raise ImportInputError(
            "The Honcho address must be an HTTP(S) URL without user credentials, query parameters, or a fragment."
        ) from exc


def _yaml_connection_fields() -> dict:
    """Use Hermes' pure merge rules, bypassing its loader's mkdir and backup writes."""
    from hermes_cli.config import (
        DEFAULT_CONFIG,
        _canonicalize_config,
        _deep_merge,
        _expand_env_vars,
        _merge_managed_overlay,
        fast_safe_load,
        get_config_path,
    )

    path = get_config_path()
    try:
        with path.open(encoding="utf-8") as handle:
            raw = fast_safe_load(handle) or {}
    except FileNotFoundError:
        raw = {}
    except Exception as exc:
        raise ImportInputError(
            "Hermes config.yaml could not be read or parsed. Repair it before importing."
        ) from exc
    if not isinstance(raw, dict):
        raise ImportInputError("Hermes config.yaml must contain a mapping.")
    from hermes_cli import managed_scope

    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is not None:
        try:
            with (managed_dir / "config.yaml").open(encoding="utf-8") as handle:
                managed_raw = fast_safe_load(handle) or {}
            if not isinstance(managed_raw, dict):
                raise ValueError("invalid managed configuration")
        except FileNotFoundError:
            pass
        except Exception as exc:
            raise ImportInputError(
                "Managed Hermes configuration could not be parsed. Repair it before importing."
            ) from exc
    merged = _canonicalize_config(_deep_merge(copy.deepcopy(DEFAULT_CONFIG), raw))
    effective, _managed = _merge_managed_overlay(_expand_env_vars(merged))
    fields = effective.get("honcho") or {}
    if not isinstance(fields, dict):
        raise ImportInputError(
            "The honcho section in config.yaml must contain a mapping."
        )
    return fields


def _path_signatures(paths: tuple[Path, ...]) -> tuple:
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino))
        except FileNotFoundError:
            result.append((str(path), None))
    return tuple(result)


def load_import_config(args: Any) -> tuple[HonchoClientConfig, str, Path]:
    """Freeze the current profile and resolve transport without client/OAuth calls."""
    from hermes_cli.profiles import get_active_profile_name

    from .client import resolve_active_host, resolve_config_path

    home = get_hermes_home().resolve()
    profile = get_active_profile_name()
    target_profile = getattr(args, "target_profile", None)
    if target_profile and target_profile != profile:
        raise ImportInputError(
            "Use 'hermes -p NAME honcho import' to select that profile; --target-profile does not switch its database."
        )
    path, host = resolve_config_path(), resolve_active_host()
    try:
        raw = _read_config(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("hosts", {}), dict):
            raise ValueError("invalid configuration")
        block = _host_block(raw, host)
        if not isinstance(block, dict):
            raise ValueError("invalid host")
        endpoint = raw.get("endpoint") or {}
        raw_url = (
            block.get("baseUrl")
            or block.get("base_url")
            or (endpoint.get("baseUrl") if isinstance(endpoint, dict) else None)
            or raw.get("baseUrl")
            or raw.get("base_url")
        )
        if raw_url:
            normalize_endpoint(raw_url)
        config = copy.deepcopy(
            HonchoClientConfig.from_global_config(
                host=host, config_path=path, strict=True
            )
        )
    except ImportInputError:
        raise
    except Exception as exc:
        raise ImportInputError(
            "Honcho configuration could not be read or parsed. Repair it before importing; no environment fallback was used."
        ) from exc
    config.hermes_home, config.config_path = home, path
    if not isinstance(config.peer_name, str) or not config.peer_name.strip():
        raise ImportInputError(
            "Set Honcho peerName to the install owner before importing history."
        )
    if (
        not isinstance(config.ai_peer, str)
        or not config.ai_peer.strip()
        or not isinstance(config.sessions, dict)
    ):
        raise ImportInputError("Honcho peer or session configuration is invalid.")
    if type(config.message_max_chars) is not int or config.message_max_chars <= len(
        "[continued] "
    ):
        raise ImportInputError(
            "messageMaxChars must exceed the continuation prefix length (12 characters)."
        )
    yaml_fields = _yaml_connection_fields()
    from honcho.client import ENVIRONMENTS

    base_url = (
        config.base_url
        or yaml_fields.get("base_url")
        or ENVIRONMENTS.get(config.environment)
    )
    if not base_url:
        raise ImportInputError("The Honcho environment has no known server address.")
    config.base_url = normalize_endpoint(base_url)
    config.timeout = (
        config.timeout
        or _resolve_optional_float(
            yaml_fields.get("timeout"), yaml_fields.get("request_timeout")
        )
        or _DEFAULT_HTTP_TIMEOUT
    )
    if not math.isfinite(config.timeout) or config.timeout <= 0:
        raise ImportInputError("The Honcho timeout must be finite and positive.")
    if workspace := getattr(args, "workspace", None):
        config.workspace_id = workspace
    if not isinstance(config.workspace_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,512}", config.workspace_id
    ):
        raise ImportInputError(
            "The workspace must be 1-512 letters, digits, underscores, or hyphens."
        )
    if not config.enabled:
        raise ImportInputError(
            "Honcho is disabled for this profile. Configure it before importing."
        )
    from hermes_cli import managed_scope
    from hermes_cli.config import get_config_path

    managed_dir = managed_scope.get_managed_dir()
    paths = tuple(
        dict.fromkeys(
            [
                path,
                home / "honcho.json",
                get_config_path(),
                *([managed_dir / "config.yaml"] if managed_dir else []),
            ]
        )
    )
    config._import_config_paths = paths
    config._import_config_signatures = _path_signatures(paths)
    return config, profile, home / "state.db"


@dataclass
class SourceSnapshot:
    db: Any
    source_path: Path
    profile: str
    source_path_key: str
    source_id: str
    anchor: dict


def _require_state_interfaces(session_db: type) -> None:
    required = {
        "list_sessions_rich": {
            "include_children",
            "include_hidden",
            "include_archived",
            "project_compression_tips",
            "limit",
            "offset",
        },
        "get_messages": {"after_id", "limit", "include_inactive", "include_compacted"},
    }
    try:
        if "read_only" not in inspect.signature(session_db).parameters:
            raise ValueError("read-only attach missing")
        for method, parameters in required.items():
            if (
                not parameters
                <= inspect.signature(getattr(session_db, method)).parameters.keys()
            ):
                raise ValueError("pagination missing")
        for method in ("get_compression_lineage", "get_session", "get_meta", "close"):
            if not callable(getattr(session_db, method, None)):
                raise ValueError("state method missing")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ImportInputError(
            "This Hermes build lacks the required read-only SessionDB and paging interfaces. Update Hermes before importing."
        ) from exc


@contextmanager
def snapshot_source(path: Path, profile: str) -> Iterator[SourceSnapshot]:
    """Back up committed WAL data, close the original, then read only the private copy."""
    from hermes_state import SessionDB

    _require_state_interfaces(SessionDB)
    try:
        source_path = path.resolve(strict=True)
        opening_stat = source_path.stat()
        if not source_path.is_file():
            raise OSError("not a file")
    except OSError as exc:
        raise ImportInputError(
            "The current profile's state.db is missing or unreadable. No database was created."
        ) from exc
    with tempfile.TemporaryDirectory(prefix="honcho-history-") as temporary_dir:
        copy_path = Path(temporary_dir) / "state.db"
        fd = os.open(copy_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        restrict_private_file(copy_path)
        deadline = time.monotonic() + 30.0

        def progress(status, remaining, total):
            if time.monotonic() > deadline:
                raise ImportInputError(
                    "Could not obtain a stable database snapshot. Pause live writes and retry."
                )

        try:
            original = sqlite3.connect(
                source_path.as_uri() + "?mode=ro", uri=True, timeout=5
            )
            try:
                destination = sqlite3.connect(copy_path)
                try:
                    original.backup(
                        destination, pages=512, progress=progress, sleep=0.05
                    )
                finally:
                    destination.close()
            finally:
                original.close()
            closing_stat = source_path.stat()
            if (opening_stat.st_dev, opening_stat.st_ino) != (
                closing_stat.st_dev,
                closing_stat.st_ino,
            ):
                raise ImportInputError(
                    "The source database was replaced during the snapshot. No upload was started."
                )
            db = SessionDB(db_path=copy_path, read_only=True)
        except sqlite3.Error as exc:
            raise ImportInputError(
                "state.db could not be read with this Hermes build. Update Hermes or repair the source before importing."
            ) from exc
        try:
            generation = db.get_meta("db_file_generation")
            anchor = (
                {"generation": generation}
                if generation
                else {"device": opening_stat.st_dev, "inode": opening_stat.st_ino}
            )
            if not generation and not opening_stat.st_ino:
                raise ImportInputError(
                    "This legacy database has no stable source identity. Update Hermes before importing."
                )
            path_key = fingerprint(
                {"path": os.path.normcase(str(source_path)), "profile": profile}
            )
            source_id = fingerprint({"path_key": path_key, "anchor": anchor})
            yield SourceSnapshot(db, source_path, profile, path_key, source_id, anchor)
        except sqlite3.Error as exc:
            raise ImportInputError(
                "The historical database schema is incompatible with this Hermes build. Update Hermes or repair the source before importing."
            ) from exc
        finally:
            db.close()


@dataclass(frozen=True)
class ImportTarget:
    segment_id: str
    root_id: str
    session_id: str
    user_peer_id: str
    assistant_peer_id: str
    mapping: str


class _PeerResolver(SessionPeersMixin):
    """Use the live resolver without constructing a manager, client, or writer."""

    def __init__(self, config, runtime_id=None, runtime_alt=None):
        self._config = config
        self._runtime_user_peer_name = runtime_id
        self._runtime_user_peer_name_alt = runtime_alt


def _object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _map_segment(
    record: dict, root_id: str, config: HonchoClientConfig, profile: str
) -> tuple[ImportTarget | None, str | None]:
    if (
        record.get("hidden")
        or record.get("source") == "subagent"
        or _object(record.get("model_config")).get("_delegate_from")
    ):
        return None, "sessions_hidden_or_subagent"
    if record.get("profile_name") not in (None, "", profile):
        return None, "sessions_other_profile"
    source = record.get("source")
    origin = _object(record.get("origin_json"))
    runtime_id = str(record.get("user_id") or "").strip()
    runtime_alt = str(origin.get("user_id_alt") or "").strip() or None
    owner = _PeerResolver(config)._declared_owner_peer_id()
    is_bot = runtime_id.startswith(BOT_AUTHOR_PREFIX)
    if source in {"cli", "desktop"} and not is_bot:
        if origin.get("user_id") and str(origin["user_id"]).strip() != runtime_id:
            return None, "sessions_ambiguous_author"
        resolver = _PeerResolver(config, runtime_id or None, runtime_alt)
        if runtime_id:
            unpinned = copy.copy(config)
            unpinned.pin_peer_name = False
            if (
                _PeerResolver(unpinned, runtime_id, runtime_alt)._resolve_user_peer_id(
                    record["id"]
                )
                != owner
            ):
                return None, "sessions_other_owner"
        user_peer = resolver._resolve_user_peer_id(record["id"])
    else:
        if (
            not runtime_id
            or str(origin.get("user_id") or "").strip() != runtime_id
            or origin.get("chat_type") != "dm"
        ):
            return None, "sessions_ambiguous_author"
        if not record.get("chat_id") or str(origin.get("chat_id") or "") != str(
            record["chat_id"]
        ):
            return None, "sessions_ambiguous_author"
        if origin.get("platform") not in {
            source,
            "local" if source == "cli" else source,
        } or origin.get("profile") not in (None, "", profile):
            return None, "sessions_ambiguous_author"
        if not record.get("session_key"):
            return None, "sessions_missing_gateway_key"
        resolver = _PeerResolver(config, runtime_id, runtime_alt)
        if is_bot:
            if not config.a2a_sessions:
                return None, "sessions_a2a_disabled"
            user_peer = resolver.resolve_author_peer_id(
                record["id"], runtime_id, is_bot=True
            )
            if not user_peer or user_peer in {owner, assistant_peer_id_for(config)}:
                return None, "sessions_ambiguous_bot"
        else:
            # pinPeerName chooses a peer; it does not prove who wrote old messages.
            unpinned = copy.copy(config)
            unpinned.pin_peer_name = False
            if (
                _PeerResolver(unpinned, runtime_id, runtime_alt)._resolve_user_peer_id(
                    record["id"]
                )
                != owner
            ):
                return None, "sessions_other_owner"
            user_peer = resolver._resolve_user_peer_id(record["id"])
    session_id = (
        config.resolve_session_name(
            cwd=record.get("cwd"),
            session_id=record["id"],
            gateway_session_key=record.get("session_key"),
            session_title=record.get("title"),
            session_title_source=record.get("title_source"),
            use_current_cwd=False,
        )
        or config.workspace_id
    )
    if is_bot:
        session_id = a2a_session_key_for(
            session_id, config, {"id": runtime_id, "is_bot": True}
        )
    # get_or_create sanitizes the resolved key immediately before using it as the SDK ID.
    session_id = resolver._sanitize_id(session_id)
    mapping = fingerprint(
        {
            "session": session_id,
            "user": user_peer,
            "assistant": assistant_peer_id_for(config),
            "root": root_id,
            "started_at": record.get("started_at"),
            "message_max_chars": config.message_max_chars,
            "import_version": IMPORT_VERSION,
        }
    )
    return ImportTarget(
        record["id"],
        root_id,
        session_id,
        user_peer,
        assistant_peer_id_for(config),
        mapping,
    ), None


def _targets(
    snapshot: SourceSnapshot,
    config: HonchoClientConfig,
    options: ImportOptions,
    skipped: Counter,
) -> list[ImportTarget]:
    records, offset = {}, 0
    while True:
        rows = snapshot.db.list_sessions_rich(
            limit=PAGE_SIZE,
            offset=offset,
            include_children=True,
            include_hidden=False,
            include_archived=True,
            project_compression_tips=False,
        )
        for row in rows:
            records[row["id"]] = row
        if len(rows) < PAGE_SIZE:
            break
        offset += len(rows)
    roots, seen = [], set()
    for record in sorted(
        records.values(), key=lambda r: (r.get("started_at") or 0, r["id"])
    ):
        if record["id"] in seen:
            continue
        lineage = snapshot.db.get_compression_lineage(record["id"]) or [record["id"]]
        seen.update(lineage)
        roots.append((lineage[0], lineage))
    targets = []
    for root_id, lineage in roots:
        for segment_id in lineage:
            record = records.get(segment_id)
            if not record:
                continue
            if record.get("source") not in options.sources:
                skipped["sessions_source_not_selected"] += 1
                continue
            target, reason = _map_segment(record, root_id, config, snapshot.profile)
            if target:
                targets.append(target)
            else:
                skipped[reason] += 1
    return targets


@dataclass(frozen=True)
class SourceMessage:
    segment_id: str
    row_id: int
    content: str
    peer_id: str
    created_at: datetime


@dataclass(frozen=True)
class PlannedChunk:
    target: ImportTarget
    row_id: int
    content: str
    peer_id: str
    created_at: datetime
    part_index: int
    part_count: int
    key: str
    payload_sha256: str
    row_hash: str

    def checkpoint(self) -> dict:
        return {
            "segment_id": self.target.segment_id,
            "root_id": self.target.root_id,
            "row_id": self.row_id,
            "peer_id": self.peer_id,
            "created_at": self.created_at.isoformat(),
            "part_index": self.part_index,
            "part_count": self.part_count,
            "key": self.key,
            "payload_sha256": self.payload_sha256,
            "row_hash": self.row_hash,
        }

    def metadata(self, source_id: str) -> dict:
        return {
            "backfill": True,
            "honcho_import_version": IMPORT_VERSION,
            "honcho_import_source": source_id,
            "honcho_import_key": self.key,
            "hermes_root_session_id": self.target.root_id,
            "hermes_session_id": self.target.segment_id,
            "hermes_message_id": self.row_id,
            "part_index": self.part_index,
            "part_count": self.part_count,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class PlannedRow:
    row_id: int
    chunks: tuple[PlannedChunk, ...] = ()
    skipped: str | None = None
    data_issue: bool = False


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
    return None


def payload_hash(
    content: str, peer_id: str, created_at: datetime, session_id: str
) -> str:
    return fingerprint(
        {
            "content": content,
            "peer_id": peer_id,
            "created_at": created_at.astimezone(UTC).isoformat(),
            "session_id": session_id,
        }
    )


def planned_rows(
    snapshot: SourceSnapshot, target: ImportTarget, options: ImportOptions, limit: int
) -> Iterator[PlannedRow]:
    from . import HonchoMemoryProvider, _is_internal_gateway_turn

    cursor, suppress_turn = 0, False
    while True:
        rows = snapshot.db.get_messages(
            target.segment_id,
            include_inactive=False,
            include_compacted=False,
            after_id=cursor,
            limit=PAGE_SIZE,
        )
        for row in rows:
            row_id = row.get("id")
            if type(row_id) is not int or row_id <= cursor:
                raise ImportInputError(
                    "state.db returned invalid message row IDs; paging stopped before upload."
                )
            cursor = row_id
            role, text = row.get("role"), _text(row.get("content"))
            if role == "user":
                suppress_turn = text is None or _is_internal_gateway_turn(text)
            if (
                role not in {"user", "assistant"}
                or row.get("_compressed_summary")
                or row.get("compacted")
            ):
                yield PlannedRow(row_id, skipped="messages_tool_system_or_summary")
                continue
            if text is None:
                yield PlannedRow(
                    row_id, skipped="messages_unsupported_content", data_issue=True
                )
                continue
            if suppress_turn or row.get("display_kind") in {
                "hidden",
                "async_delegation_complete",
            }:
                yield PlannedRow(row_id, skipped="messages_control_turn")
                continue
            if role == "assistant" and (
                row.get("tool_calls")
                or row.get("finish_reason") in {"tool_calls", "function_call"}
            ):
                yield PlannedRow(row_id, skipped="messages_intermediate_tool_call")
                continue
            try:
                timestamp = row.get("timestamp")
                if (
                    isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)
                ):
                    raise ValueError("timestamp invalid")
                created_at = datetime.fromtimestamp(timestamp, UTC)
            except (ValueError, TypeError, OSError, OverflowError):
                yield PlannedRow(
                    row_id, skipped="messages_invalid_timestamp", data_issue=True
                )
                continue
            if (
                created_at < options.cutoff
                or created_at > options.now
                or (options.before and created_at >= options.before)
            ):
                yield PlannedRow(row_id, skipped="messages_outside_time_window")
                continue
            cleaned = sanitize_context(text).strip()
            if not cleaned:
                yield PlannedRow(row_id, skipped="messages_empty_after_cleanup")
                continue
            message = SourceMessage(
                target.segment_id,
                row_id,
                cleaned,
                target.user_peer_id if role == "user" else target.assistant_peer_id,
                created_at,
            )
            parts = HonchoMemoryProvider._chunk_message(message.content, limit)
            row_hash = payload_hash(
                cleaned, message.peer_id, created_at, target.session_id
            )
            chunks = tuple(
                PlannedChunk(
                    target,
                    row_id,
                    part,
                    message.peer_id,
                    created_at,
                    index,
                    len(parts),
                    fingerprint(
                        {
                            "source": snapshot.source_id,
                            "session": target.segment_id,
                            "row": row_id,
                            "part": index,
                        }
                    ),
                    payload_hash(part, message.peer_id, created_at, target.session_id),
                    row_hash,
                )
                for index, part in enumerate(parts)
            )
            yield PlannedRow(row_id, chunks=chunks)
        if len(rows) < PAGE_SIZE:
            break


@dataclass
class ImportSummary:
    source_messages: int = 0
    planned_chunks: int = 0
    ledger_covered: int = 0
    logical_sessions: int = 0
    target_sessions: int = 0
    data_issues: int = 0
    uploaded_chunks: int = 0
    remote_covered: int = 0
    recovered_chunks: int = 0
    blocked_targets: int = 0
    pending_batches: int = 0
    skipped: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImportPlan:
    targets: tuple[ImportTarget, ...]
    candidate_target_ids: frozenset[str]
    identity: dict
    summary: ImportSummary


def _row_progress(segment: dict | None, row: PlannedRow) -> int:
    record = segment["rows"].get(str(row.row_id)) if segment else None
    if record and not row.chunks and row.skipped != "messages_outside_time_window":
        raise ImportStateError(
            "An imported row no longer matches the eligible source data. Restore the original source before resuming."
        )
    if record and row.chunks:
        if record["row_hash"] != row.chunks[0].row_hash or record["part_count"] != len(
            row.chunks
        ):
            raise ImportStateError(
                "A source row changed after import. No duplicate was uploaded; restore the original source before resuming."
            )
        return record["confirmed_parts"]
    return 0


def build_plan(
    snapshot: SourceSnapshot,
    config: HonchoClientConfig,
    options: ImportOptions,
    state: dict,
) -> ImportPlan:
    summary = ImportSummary()
    identity = {
        "server": config.base_url,
        "workspace": config.workspace_id,
        "profile": snapshot.profile,
        "source": snapshot.source_id,
    }
    scope = get_scope(state, identity, snapshot.source_path_key, snapshot.anchor)
    targets = _targets(snapshot, config, options, summary.skipped)
    roots, sessions, candidates = set(), set(), set()
    for target in targets:
        segment = get_segment(scope, target.segment_id, target.mapping)
        for row in planned_rows(snapshot, target, options, config.message_max_chars):
            covered = _row_progress(segment, row)
            if row.skipped:
                summary.skipped[row.skipped] += 1
                summary.data_issues += int(row.data_issue)
                continue
            summary.source_messages += 1
            summary.planned_chunks += len(row.chunks)
            summary.ledger_covered += covered
            roots.add(target.root_id)
            sessions.add(target.session_id)
            if covered < len(row.chunks):
                candidates.add(target.session_id)
    summary.logical_sessions, summary.target_sessions = len(roots), len(sessions)
    summary.data_issues += sum(
        summary.skipped[reason]
        for reason in (
            "sessions_ambiguous_author",
            "sessions_ambiguous_bot",
            "sessions_missing_gateway_key",
        )
    )
    summary.pending_batches = int(bool(scope and scope.get("pending")))
    return ImportPlan(tuple(targets), frozenset(candidates), identity, summary)


def _safe_label(value: Any) -> str:
    from agent.redact import redact_sensitive_text

    try:
        text = redact_sensitive_text(
            str(value), force=True, redact_url_credentials=True
        )
        text = re.sub(r"(://)[^/\s?#@]+@", r"\1***@", text)
        return " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text).split())[:1000]
    except Exception:
        return "[details unavailable]"


def print_preview(
    plan: ImportPlan,
    snapshot: SourceSnapshot,
    config: HonchoClientConfig,
    options: ImportOptions,
) -> None:
    summary = plan.summary
    print("Honcho history import preview")
    print(f"  Profile: {_safe_label(snapshot.profile)}")
    print(f"  Source database: {_safe_label(snapshot.source_path)}")
    print(f"  Server: {_safe_label(config.base_url)}")
    print(f"  Workspace: {_safe_label(config.workspace_id)}")
    print(f"  Window (UTC): {options.cutoff.isoformat()} to {options.now.isoformat()}")
    if options.before:
        print(f"  Strictly before: {options.before.isoformat()}")
    print(f"  Sources: {', '.join(options.sources)}")
    print(
        f"  Logical sessions: {summary.logical_sessions}; target sessions: {summary.target_sessions}"
    )
    print(
        f"  Source messages: {summary.source_messages}; message chunks: {summary.planned_chunks}"
    )
    print(f"  Already covered by local progress: {summary.ledger_covered} chunks")
    if summary.pending_batches:
        print("  Pending batch: must be reconciled before uploading")
    if summary.skipped:
        print("  Skipped sessions/messages:")
        for reason, count in sorted(summary.skipped.items()):
            print(f"    {reason}: {count}")
    print(
        "  Remote overlap checks have not run. Missing historical cwd uses the global workspace fallback."
    )


def _private_client(config: HonchoClientConfig):
    from .client import _build_client, _refresh_oauth

    # This config and client belong to one confirmed import, never to the shared cache.
    if (
        _path_signatures(config._import_config_paths)
        != config._import_config_signatures
    ):
        raise ImportInputError(
            "Configuration changed after the preview. Run the command again to confirm its destination before uploading."
        )
    private_config = copy.deepcopy(config)
    _refresh_oauth(private_config)
    # An absent scoped key must not make the SDK read an unrelated process-wide key.
    if not private_config.api_key:
        private_config.api_key = "local"
    client = _build_client(private_config, max_retries=0)
    try:
        if (
            normalize_endpoint(client._http.base_url, strip_version=False)
            != config.base_url
            or client._http.max_retries != 0
        ):
            raise ImportInputError(
                "The SDK destination or retry settings differ from the confirmed preview. No upload was started."
            )
    except Exception:
        client._http.close()
        raise
    return client


@contextmanager
def _remote_reads(client) -> Iterator[None]:
    """Avoid the SDK's lazy workspace creation during overlap and pending reads."""
    ensured = client._workspace_ensured
    client._workspace_ensured = True
    try:
        yield
    finally:
        client._workspace_ensured = ensured


def _remote_messages(session, *, filters: dict | None = None) -> list:
    """Read every page; incomplete or changing pagination cannot prove safe overlap."""
    page = session.messages(filters=filters, size=BATCH_SIZE)
    if (
        type(page.total) is not int
        or page.total < 0
        or type(page.pages) is not int
        or page.pages < 0
    ):
        raise ImportStateError(
            "Honcho returned incomplete message pagination. No new upload was started."
        )
    total, page_count = page.total, max(1, page.pages)
    messages = []
    for number in range(1, page_count + 1):
        if (
            page.page != number
            or page.total != total
            or max(1, page.pages) != page_count
        ):
            raise ImportStateError(
                "Remote message pagination changed during the read. Pause live writes and retry."
            )
        messages.extend(page.items)
        if number < page_count:
            page = page.get_next_page()
            if page is None:
                raise ImportStateError(
                    "Honcho returned incomplete message pagination. No new upload was started."
                )
    if len(messages) != total:
        raise ImportStateError(
            "Remote message pagination was incomplete. No new upload was started."
        )
    return messages


def _import_marker(message) -> tuple[str, str] | None:
    metadata = message.metadata
    if (
        not isinstance(metadata, dict)
        or metadata.get("backfill") is not True
        or type(metadata.get("honcho_import_version")) is not int
    ):
        return None
    if metadata["honcho_import_version"] != IMPORT_VERSION:
        return None
    source, key = (
        metadata.get("honcho_import_source"),
        metadata.get("honcho_import_key"),
    )
    digest = metadata.get("payload_sha256")
    if not all(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
        for value in (source, key, digest)
    ):
        return None
    if not all(
        isinstance(metadata.get(field), str) and metadata[field]
        for field in ("hermes_session_id", "hermes_root_session_id")
    ):
        return None
    row_id, part, count = (
        metadata.get("hermes_message_id"),
        metadata.get("part_index"),
        metadata.get("part_count"),
    )
    if (
        type(row_id) is not int
        or row_id < 1
        or type(part) is not int
        or type(count) is not int
        or not 0 <= part < count
    ):
        return None
    if (
        not isinstance(message.created_at, datetime)
        or message.created_at.tzinfo is None
    ):
        return None
    if (
        payload_hash(
            message.content, message.peer_id, message.created_at, message.session_id
        )
        != digest
    ):
        return None
    return source, key


def _matches_checkpoint(
    message, descriptor: dict, source_id: str, target_id: str
) -> bool:
    metadata = message.metadata
    if _import_marker(message) != (source_id, descriptor["key"]):
        return False
    for key, expected in {
        "payload_sha256": descriptor["payload_sha256"],
        "hermes_session_id": descriptor["segment_id"],
        "hermes_root_session_id": descriptor["root_id"],
        "hermes_message_id": descriptor["row_id"],
        "part_index": descriptor["part_index"],
        "part_count": descriptor["part_count"],
    }.items():
        if metadata.get(key) != expected:
            return False
    created_at = message.created_at
    if (
        not isinstance(created_at, datetime)
        or created_at.tzinfo is None
        or message.peer_id != descriptor["peer_id"]
    ):
        return False
    if (
        created_at.astimezone(UTC).isoformat() != descriptor["created_at"]
        or message.session_id != target_id
    ):
        return False
    return (
        payload_hash(message.content, message.peer_id, created_at, target_id)
        == descriptor["payload_sha256"]
    )


def _indexed_remote(messages: list) -> tuple[dict, bool]:
    indexed, unmarked = {}, False
    for message in messages:
        marker = _import_marker(message)
        if marker is None:
            unmarked = True
        elif marker in indexed:
            raise ImportStateError(
                "A target has duplicate import keys. Reconcile that history before importing more."
            )
        else:
            indexed[marker] = message
    return indexed, unmarked


def _pending_batch(target_id: str, chunks: list[PlannedChunk], cursors: dict) -> dict:
    descriptors = [chunk.checkpoint() for chunk in chunks]
    return {
        "batch_id": fingerprint({"target": target_id, "chunks": descriptors}),
        "target": target_id,
        "chunks": descriptors,
        "cursors": dict(cursors),
    }


def _recover_pending(
    client,
    scope: dict,
    state: dict,
    state_path: Path,
    source_id: str,
    summary: ImportSummary,
) -> bool:
    from honcho.http.exceptions import NotFoundError
    from honcho.session import Session

    pending = scope.get("pending")
    if not pending:
        return True
    filters = {
        "OR": [
            {
                "metadata": {
                    "honcho_import_source": source_id,
                    "honcho_import_key": chunk["key"],
                }
            }
            for chunk in pending["chunks"]
        ]
    }
    session = Session(pending["target"], client)
    with _remote_reads(client):
        try:
            messages = _remote_messages(session, filters=filters)
        except NotFoundError:
            messages = []
    indexed, _unmarked = _indexed_remote(messages)
    all_confirmed = all(
        (message := indexed.get((source_id, chunk["key"]))) is not None
        and _matches_checkpoint(message, chunk, source_id, pending["target"])
        for chunk in pending["chunks"]
    )
    if not all_confirmed:
        summary.pending_batches = 1
        summary.errors.append(
            f"Pending batch {pending['batch_id'][:12]} is not fully confirmed remotely. Nothing was resent; wait for processing and retry the check."
        )
        return False
    confirm_batch(scope, pending)
    try:
        write_state(state_path, state)
    except Exception:
        summary.pending_batches = 1
        summary.errors.append(
            "The remote pending batch was confirmed, but progress could not be saved. Re-run the check before uploading more."
        )
        return False
    summary.recovered_chunks += len(pending["chunks"])
    summary.pending_batches = 0
    return True


def _prepare_session(
    client, target_id: str, targets: list[ImportTarget], config: HonchoClientConfig
):
    from honcho.session import SessionPeerConfig

    session = client.session(target_id)
    existing_peers = {peer.id for peer in session.peers()}
    desired = {}
    for target in targets:
        desired[target.user_peer_id] = SessionPeerConfig(
            observe_me=config.user_observe_me, observe_others=config.user_observe_others
        )
        desired[target.assistant_peer_id] = SessionPeerConfig(
            observe_me=config.ai_observe_me, observe_others=config.ai_observe_others
        )
    missing = [
        (peer_id, settings)
        for peer_id, settings in desired.items()
        if peer_id not in existing_peers
    ]
    if missing:
        session.add_peers(missing)
    # Never set configurations for peers already present: the server/UI owns them.
    return session


def _send_batch(
    client,
    session,
    chunks: list[PlannedChunk],
    cursors: dict,
    scope: dict,
    state: dict,
    state_path: Path,
    source_id: str,
    summary: ImportSummary,
) -> int:
    from honcho.http.exceptions import APIError
    from honcho.peer import Peer

    messages = [
        Peer(chunk.peer_id, client).message(
            chunk.content,
            created_at=chunk.created_at,
            metadata=chunk.metadata(source_id),
        )
        for chunk in chunks
    ]
    pending = _pending_batch(session.id, chunks, cursors)
    scope["pending"] = pending
    try:
        write_state(state_path, state)
    except Exception:
        summary.errors.append(
            "Could not save the pending batch. No messages in that batch were sent."
        )
        return 1
    try:
        response = session.add_messages(messages)
    except APIError as exc:
        status = exc.status
        if type(status) is int and 400 <= status < 500 and status != 408:
            scope["pending"] = None
            write_state(state_path, state)
            hint = (
                "Run 'hermes honcho setup' for this profile, then resume."
                if status == 401
                else "Fix access or input before resuming."
            )
            summary.errors.append(
                f"Honcho rejected the batch (HTTP {status}); progress did not advance. {hint}"
            )
            return 1
        summary.pending_batches = 1
        summary.errors.append(
            "The batch outcome is unknown after a server error. Re-run to reconcile it; nothing will be automatically resent."
        )
        return 3
    except Exception:
        summary.pending_batches = 1
        summary.errors.append(
            "The batch outcome is unknown after a transport failure. Re-run to reconcile it; nothing will be automatically resent."
        )
        return 3
    try:
        indexed, unmarked = _indexed_remote(response)
        verified = (
            not unmarked
            and len(response) == len(chunks)
            and all(
                (message := indexed.get((source_id, chunk.key))) is not None
                and _matches_checkpoint(
                    message, chunk.checkpoint(), source_id, session.id
                )
                for chunk in chunks
            )
        )
    except Exception:
        verified = False
    if not verified:
        summary.pending_batches = 1
        summary.errors.append(
            "Honcho returned an incomplete or conflicting batch response. Progress was not advanced; re-run to reconcile it."
        )
        return 3
    summary.uploaded_chunks += len(chunks)
    confirm_batch(scope, pending)
    try:
        write_state(state_path, state)
    except Exception:
        summary.pending_batches = 1
        summary.errors.append(
            "Honcho accepted the batch but local progress could not be saved. The pending batch must be reconciled before another upload."
        )
        return 3
    return 0


def upload_plan(
    snapshot: SourceSnapshot,
    config: HonchoClientConfig,
    options: ImportOptions,
    state_path: Path,
) -> tuple[ImportSummary, int]:
    from honcho.http.exceptions import NotFoundError
    from honcho.session import Session

    with state_lock(state_path):
        state = read_state(state_path)
        plan = build_plan(snapshot, config, options, state)
        summary = plan.summary
        if not plan.candidate_target_ids and not summary.pending_batches:
            return summary, 3 if summary.data_issues else 0
        trusted_progress = (
            get_scope(state, plan.identity, snapshot.source_path_key, snapshot.anchor)
            is not None
        )
        scope = get_scope(
            state, plan.identity, snapshot.source_path_key, snapshot.anchor, create=True
        )
        for target in plan.targets:
            get_segment(scope, target.segment_id, target.mapping, create=True)
        client = _private_client(config)
        try:
            if not _recover_pending(
                client, scope, state, state_path, snapshot.source_id, summary
            ):
                return summary, 3
            align_selection(scope, options.selection)
            # Recovery may have changed which parts are confirmed. Rebuild from the same snapshot.
            refreshed_plan = build_plan(snapshot, config, options, state)
            target_ids = refreshed_plan.candidate_target_ids
            remote_indexes, blocked, unmarked_blocked = {}, set(), 0
            with _remote_reads(client):
                for target_id in sorted(target_ids):
                    try:
                        messages = _remote_messages(Session(target_id, client))
                    except NotFoundError:
                        messages = []
                    indexed, unmarked = _indexed_remote(messages)
                    if unmarked and options.before is None:
                        blocked.add(target_id)
                        summary.blocked_targets += 1
                        unmarked_blocked += 1
                        continue
                    target_segments = [
                        target
                        for target in plan.targets
                        if target.session_id == target_id
                    ]
                    if not trusted_progress and any(
                        marker[0] == snapshot.source_id for marker in indexed
                    ):
                        complete = all(
                            (message := indexed.get((snapshot.source_id, chunk.key)))
                            is not None
                            and _matches_checkpoint(
                                message,
                                chunk.checkpoint(),
                                snapshot.source_id,
                                target_id,
                            )
                            for target in target_segments
                            for row in planned_rows(
                                snapshot, target, options, config.message_max_chars
                            )
                            for chunk in row.chunks
                        )
                        if not complete:
                            blocked.add(target_id)
                            summary.blocked_targets += 1
                            summary.errors.append(
                                "Remote data exists for this source but its local progress is missing. Some parts may still be processing; restore the original ledger before uploading more."
                            )
                            continue
                    # Detect remote hash conflicts and partial new rows before any message write.
                    conflict = False
                    for target in target_segments:
                        segment = get_segment(scope, target.segment_id, target.mapping)
                        for row in planned_rows(
                            snapshot, target, options, config.message_max_chars
                        ):
                            found = [
                                indexed.get((snapshot.source_id, chunk.key))
                                for chunk in row.chunks
                            ]
                            if any(
                                message is not None
                                and not _matches_checkpoint(
                                    message,
                                    chunk.checkpoint(),
                                    snapshot.source_id,
                                    target_id,
                                )
                                for chunk, message in zip(
                                    row.chunks, found, strict=True
                                )
                            ):
                                conflict = True
                            count_found = sum(message is not None for message in found)
                            if 0 < count_found < len(row.chunks) and not _row_progress(
                                segment, row
                            ):
                                conflict = True
                    if conflict:
                        blocked.add(target_id)
                        summary.blocked_targets += 1
                        summary.errors.append(
                            "A target contains conflicting or partial imported source data. It was not resent; reconcile its original progress first."
                        )
                        continue
                    remote_indexes[target_id] = indexed
            if unmarked_blocked:
                summary.errors.append(
                    "Some targets contain unmarked live history. They were not imported; use --before only if you know the true historical cutoff."
                )
            for target_id in sorted(target_ids - blocked):
                targets = [
                    target for target in plan.targets if target.session_id == target_id
                ]
                remote = remote_indexes[target_id]
                session, batch, cursors = None, [], {}
                for target in targets:
                    segment = get_segment(
                        scope, target.segment_id, target.mapping, create=True
                    )
                    for row in planned_rows(
                        snapshot, target, options, config.message_max_chars
                    ):
                        covered = _row_progress(segment, row)
                        if not row.chunks:
                            cursors[target.segment_id] = row.row_id
                            continue
                        for chunk in row.chunks[covered:]:
                            if existing := remote.get((snapshot.source_id, chunk.key)):
                                if not _matches_checkpoint(
                                    existing,
                                    chunk.checkpoint(),
                                    snapshot.source_id,
                                    target_id,
                                ):
                                    raise ImportStateError(
                                        "Remote import data conflicts with the source payload. No duplicate was uploaded."
                                    )
                                # Flush earlier new chunks before committing later remote-confirmed parts.
                                if batch:
                                    session = session or _prepare_session(
                                        client, target_id, targets, config
                                    )
                                    code = _send_batch(
                                        client,
                                        session,
                                        batch,
                                        cursors,
                                        scope,
                                        state,
                                        state_path,
                                        snapshot.source_id,
                                        summary,
                                    )
                                    if code:
                                        return (
                                            summary,
                                            3
                                            if summary.uploaded_chunks or code == 3
                                            else 1,
                                        )
                                    batch, cursors = [], {}
                                part_cursors = (
                                    {target.segment_id: row.row_id}
                                    if chunk.part_index + 1 == chunk.part_count
                                    else {}
                                )
                                confirm_batch(
                                    scope,
                                    _pending_batch(target_id, [chunk], part_cursors),
                                )
                                write_state(state_path, state)
                                summary.remote_covered += 1
                                continue
                            batch.append(chunk)
                            if chunk.part_index + 1 == chunk.part_count:
                                cursors[target.segment_id] = row.row_id
                            if len(batch) == BATCH_SIZE:
                                session = session or _prepare_session(
                                    client, target_id, targets, config
                                )
                                code = _send_batch(
                                    client,
                                    session,
                                    batch,
                                    cursors,
                                    scope,
                                    state,
                                    state_path,
                                    snapshot.source_id,
                                    summary,
                                )
                                if code:
                                    return (
                                        summary,
                                        3
                                        if summary.uploaded_chunks or code == 3
                                        else 1,
                                    )
                                batch, cursors = [], {}
                        if covered == len(row.chunks):
                            cursors[target.segment_id] = row.row_id
                if batch:
                    session = session or _prepare_session(
                        client, target_id, targets, config
                    )
                    code = _send_batch(
                        client,
                        session,
                        batch,
                        cursors,
                        scope,
                        state,
                        state_path,
                        snapshot.source_id,
                        summary,
                    )
                    if code:
                        return summary, 3 if summary.uploaded_chunks or code == 3 else 1
                    cursors = {}
                for segment_id, row_id in cursors.items():
                    segment = scope["segments"][segment_id]
                    segment["last_scanned_message_id"] = max(
                        segment["last_scanned_message_id"], row_id
                    )
                write_state(state_path, state)
            return summary, 3 if summary.blocked_targets or summary.data_issues else 0
        finally:
            client._http.close()


def print_summary(summary: ImportSummary, code: int) -> None:
    status = (
        "complete" if code == 0 else "partial or unresolved" if code == 3 else "failed"
    )
    print(
        f"Import {status}: {summary.uploaded_chunks} chunks uploaded, {summary.remote_covered} already remote, "
        f"{summary.recovered_chunks} recovered, {summary.blocked_targets} targets blocked, {summary.data_issues} data issues."
    )
    for error in summary.errors:
        print(f"  {error}")


def run_import(args: Any) -> int:
    """Preview first; only confirmation permits a lock, OAuth refresh, or remote call."""
    options = parse_options(args)
    config, profile, source_path = load_import_config(args)
    state_path = Path.home() / ".honcho" / "hermes-import-state.json"
    state = read_state(state_path)
    with snapshot_source(source_path, profile) as snapshot:
        plan = build_plan(snapshot, config, options, state)
        print_preview(plan, snapshot, config, options)
        if getattr(args, "dry_run", False):
            print(
                "Dry run complete. No remote calls or progress/configuration writes were made."
            )
            return 0
        if not getattr(args, "yes", False):
            import sys

            if not sys.stdin.isatty():
                raise ImportInputError(
                    "Non-interactive import requires --yes. Use --dry-run to preview first."
                )
            print(
                "Pause live writes for this profile. Imported messages trigger Honcho background processing."
            )
            if input(
                "Upload this history to the displayed server and workspace? [y/N] "
            ).strip().lower() not in {"y", "yes"}:
                print("Import cancelled. No remote calls or progress writes were made.")
                return 0
        else:
            print(
                "Pause live writes for this profile. Imported messages trigger Honcho background processing."
            )
        if not plan.candidate_target_ids and not plan.summary.pending_batches:
            code = 3 if plan.summary.data_issues else 0
            print_summary(plan.summary, code)
            return code
        summary, code = upload_plan(snapshot, config, options, state_path)
        print_summary(summary, code)
        return code
