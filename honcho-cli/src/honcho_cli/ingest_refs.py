"""Name an imported peer or session the way `honcho ingest status` shows it.

`ingest status` numbers the people and sessions it lists and saves that
numbering per instance, credential and workspace, so `honcho peer exclude 3` means the third
person in the last listing. A reference can also be the peer or session ID, or
a display name that matches exactly one of them (ignoring case).

Only peers and sessions an import created resolve: a mistyped name is an error,
never a get-or-create of a new peer or session.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from honcho_cli import config
from honcho_cli.output import print_error


_KEY_ITERATIONS = 600_000
"""PBKDF2 rounds for the credential's part of the path (about 35 ms)."""


@functools.cache
def _instance_dir(base_url: str, credential: str) -> str:
    # The credential picks the tenant: two tenants can share an instance and a
    # workspace name, and generated peer names, so it is part of the key. It's a
    # secret, so it goes through a slow key-derivation function, salted with the
    # instance: the directory name doesn't make the key any cheaper to guess.
    # Cached, so a command that resolves several references derives it once.
    return hashlib.pbkdf2_hmac(
        "sha256", credential.encode(), base_url.rstrip("/").encode(), _KEY_ITERATIONS
    ).hex()[:16]


def _listing_path(base_url: str, credential: str, workspace_id: str) -> Path:
    instance = _instance_dir(base_url, credential)
    workspace = hashlib.sha256(workspace_id.encode()).hexdigest()[:16]
    return config.CONFIG_DIR / "ingest" / instance / f"{workspace}.json"


def save_listing(
    base_url: str,
    credential: str,
    workspace_id: str,
    people: list[str],
    sessions: list[str],
) -> None:
    """Remember the numbering of the last `ingest status`. Best effort."""
    path = _listing_path(base_url, credential, workspace_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"people": people, "sessions": sessions}))
    except OSError:
        pass


def _saved(base_url: str, credential: str, workspace_id: str, kind: str) -> list[str]:
    try:
        data: dict[str, Any] = json.loads(
            _listing_path(base_url, credential, workspace_id).read_text()
        )
    except (OSError, ValueError):
        return []
    ids = data.get(kind)
    return [str(i) for i in ids] if isinstance(ids, list) else []


def resolve(
    ref: str,
    *,
    kind: str,
    base_url: str,
    credential: str,
    workspace_id: str,
    candidates: Iterable[tuple[str, str | None]],
) -> str:
    """The ID that `ref` names among an import's people or sessions.

    Args:
        ref: A number from the last `ingest status`, an ID, or a display name.
        kind: ``"people"`` or ``"sessions"``.
        candidates: ``(id, display_name)`` for every imported peer or session.
    """
    label = "person" if kind == "people" else "session"
    listed = list(candidates)
    if ref.isdigit():
        numbered = _saved(base_url, credential, workspace_id, kind)
        index = int(ref) - 1
        if not numbered:
            print_error(
                "NO_LISTING",
                "No numbered list for this workspace yet. Run `honcho ingest status`,"
                + f" or name the {label} by ID or display name.",
            )
            raise typer.Exit(1)
        if not 0 <= index < len(numbered):
            print_error(
                "NOT_LISTED",
                f"There is no {label} {ref} in the last `honcho ingest status`.",
            )
            raise typer.Exit(1)
        chosen = numbered[index]
        if chosen not in {i for i, _ in listed}:
            print_error(
                "STALE_LISTING",
                f"{label.title()} {ref} is no longer in this workspace's import."
                + " Run `honcho ingest status` again.",
            )
            raise typer.Exit(1)
        return chosen

    if ref in {i for i, _ in listed}:
        return ref
    named = [i for i, name in listed if name and name.casefold() == ref.casefold()]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        print_error(
            "AMBIGUOUS",
            f"{len(named)} imported {kind} are named '{ref}'. Use the number from"
            + " `honcho ingest status`, or the ID.",
            {"matches": named},
        )
        raise typer.Exit(1)
    print_error(
        f"{label.upper()}_NOT_FOUND",
        f"No imported {label} is named '{ref}'. `honcho ingest status` lists them.",
    )
    raise typer.Exit(1)
