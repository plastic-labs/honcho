#!/usr/bin/env python3
"""
Compute the `migration` block a release posts to the add_honcho_version webhook.

    {"has_migration": bool, "alembic_revision": str, "compat_floor": "X.Y.Z"}

- has_migration: whether migrations/versions/ changed since the previous release tag.
- alembic_revision: the single alembic head at the release ref.
- compat_floor: read from migrations/compat.toml, which must be declared against that
  head, name a release tag reachable from the ref, and not drop below the previous
  release's floor.

Reads from git objects at --ref, not the working tree, and needs only the standard
library. Used by the push-gcp-registry workflows; if it is moved, they will need to be
updated.

Usage:
    python3 scripts/release_manifest.py --ref v3.4.0
    python3 scripts/release_manifest.py --ref main --version 3.4.0  # before tagging
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = "migrations/versions"
COMPAT_FILE = "migrations/compat.toml"

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

Version = tuple[int, int, int]


class ManifestError(Exception):
    """The release manifest can't be computed; carries every problem found."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems: list[str] = problems


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def parse_version(text: str) -> Version | None:
    """Parse `X.Y.Z` or `vX.Y.Z`; pre-release and other suffixes return None."""
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def _format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _read_at(repo: Path, ref: str, path: str) -> str | None:
    """Contents of `path` at `ref`, or None if it doesn't exist there."""
    try:
        return _git(repo, "show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def release_tags(repo: Path, ref: str) -> dict[Version, str]:
    """Release tags (`vX.Y.Z`) reachable from `ref`, keyed by version."""
    tags: dict[Version, str] = {}
    for tag in _git(repo, "tag", "--merged", ref, "--list", "v*").split():
        version = parse_version(tag)
        if version is not None:
            tags[version] = tag
    return tags


def release_version(repo: Path, ref: str) -> Version:
    """The `project.version` in pyproject.toml at `ref`."""
    pyproject = _read_at(repo, ref, "pyproject.toml")
    if pyproject is None:
        raise ManifestError([f"pyproject.toml not found at {ref}"])
    raw = str(tomllib.loads(pyproject)["project"]["version"])
    version = parse_version(raw)
    if version is None:
        raise ManifestError([f"pyproject.toml version {raw!r} is not X.Y.Z"])
    return version


def _module_revisions(source: str) -> tuple[str | None, list[str]]:
    """The `revision` and `down_revision` values assigned in a migration module."""
    revision: str | None = None
    down: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id == "revision":
            revision = ast.literal_eval(value)
        elif target.id == "down_revision":
            literal = ast.literal_eval(value)
            if isinstance(literal, str):
                down = [literal]
            elif literal is not None:
                down = list(literal)
    return revision, down


def alembic_heads(repo: Path, ref: str) -> list[str]:
    """Revisions at `ref` that no other revision names as its down_revision."""
    revisions: set[str] = set()
    parents: set[str] = set()
    listing = _git(repo, "ls-tree", "--name-only", f"{ref}:{VERSIONS_DIR}")
    for name in listing.split():
        if not name.endswith(".py"):
            continue
        source = _read_at(repo, ref, f"{VERSIONS_DIR}/{name}")
        assert source is not None
        revision, down = _module_revisions(source)
        if revision is not None:
            revisions.add(revision)
            parents.update(down)
    return sorted(revisions - parents)


def has_migration_since(repo: Path, previous_tag: str | None, ref: str) -> bool:
    """Whether any migration file changed between `previous_tag` and `ref`."""
    if previous_tag is None:
        return True
    changed = _git(
        repo, "diff", "--name-only", previous_tag, ref, "--", f"{VERSIONS_DIR}/*.py"
    )
    return bool(changed.strip())


def _load_compat(repo: Path, ref: str) -> dict[str, object] | None:
    source = _read_at(repo, ref, COMPAT_FILE)
    return None if source is None else tomllib.loads(source)


def build_manifest(
    repo: Path, ref: str, version: Version | None = None
) -> dict[str, object]:
    """Compute and validate the `migration` block for releasing `ref` as `version`."""
    version = version or release_version(repo, ref)
    tags = release_tags(repo, ref)
    earlier = [tag_version for tag_version in tags if tag_version < version]
    previous_tag = tags[max(earlier)] if earlier else None

    problems: list[str] = []

    heads = alembic_heads(repo, ref)
    if len(heads) != 1:
        problems.append(f"expected one alembic head at {ref}, found {heads}")
    head = heads[0] if len(heads) == 1 else None

    compat = _load_compat(repo, ref)
    if compat is None:
        raise ManifestError([*problems, f"{COMPAT_FILE} not found at {ref}"])

    declared_revision = compat.get("alembic_revision")
    if head is not None and declared_revision != head:
        problems.append(
            f"{COMPAT_FILE} declares alembic_revision {declared_revision!r} but the "
            + f"head at {ref} is {head!r}. Restate compat_floor for the new revision "
            + "and update alembic_revision."
        )

    raw_floor = str(compat.get("compat_floor", ""))
    floor = parse_version(raw_floor)
    if floor is None:
        problems.append(f"{COMPAT_FILE} compat_floor {raw_floor!r} is not X.Y.Z")
    else:
        if floor > version:
            problems.append(
                f"compat_floor {raw_floor} is newer than the release {_format_version(version)}"
            )
        elif floor != version and floor not in tags:
            problems.append(
                f"compat_floor {raw_floor} is not a release tag reachable from {ref}"
            )
        previous_compat = (
            _load_compat(repo, previous_tag) if previous_tag is not None else None
        )
        previous_floor = (
            parse_version(str(previous_compat.get("compat_floor", "")))
            if previous_compat is not None
            else None
        )
        if previous_floor is not None and floor < previous_floor:
            problems.append(
                f"compat_floor {raw_floor} is lower than {_format_version(previous_floor)}, "
                + f"the floor declared at {previous_tag}; floors only rise"
            )

    if problems:
        raise ManifestError(problems)

    assert floor is not None
    return {
        "has_migration": has_migration_since(repo, previous_tag, ref),
        "alembic_revision": head,
        "compat_floor": _format_version(floor),
    }


def main(argv: list[str] | None = None, repo: Path = PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the release manifest's migration block."
    )
    parser.add_argument("--ref", default="HEAD", help="git ref being released")
    parser.add_argument(
        "--version", help="release version (default: pyproject.toml at --ref)"
    )
    args = parser.parse_args(argv)

    try:
        version = None
        if args.version is not None:
            version = parse_version(args.version)
            if version is None:
                raise ManifestError([f"--version {args.version!r} is not X.Y.Z"])
        manifest = build_manifest(repo, args.ref, version)
    except ManifestError as exc:
        prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") else "error: "
        for problem in exc.problems:
            print(f"{prefix}{problem}", file=sys.stderr)
        return 1

    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
