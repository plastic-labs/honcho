"""Tests for the release_manifest script."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.check_alembic_heads import load_script_directory
from scripts.release_manifest import (
    PROJECT_ROOT,
    ManifestError,
    alembic_heads,
    build_manifest,
    main,
)

# Settings a fixture could clobber in the real repository's config.
GUARDED_CONFIG_KEYS = (
    "core.bare",
    "core.worktree",
    "user.name",
    "user.email",
    "commit.gpgsign",
    "tag.gpgsign",
)


def _scrubbed_env() -> dict[str, str]:
    """The environment without inherited GIT_* variables (git hooks export GIT_DIR)."""
    return {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }


def _real_repo_config() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for key in GUARDED_CONFIG_KEYS:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "config", "--local", "--get", key],
            capture_output=True,
            text=True,
            env=_scrubbed_env(),
        )
        values[key] = result.stdout.strip() if result.returncode == 0 else None
    return values


class Repo:
    """A throwaway git repo shaped like Honcho's release layout."""

    def __init__(self, root: Path):
        self.root: Path = root
        (root / "migrations" / "versions").mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> None:
        """Run git pinned to this repo, ignoring inherited and user-level git config."""
        env = _scrubbed_env() | {
            "GIT_DIR": str(self.root / ".git"),
            "GIT_WORK_TREE": str(self.root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                *args,
            ],
            check=True,
            env=env,
        )

    def revision(self, revision: str, down: str | tuple[str, ...] | None) -> None:
        (self.root / "migrations" / "versions" / f"{revision}_rev.py").write_text(
            f'"""{revision}"""\n\n'
            + f"revision: str = {revision!r}\n"
            + f"down_revision: str | None = {down!r}\n"
        )

    def compat(self, revision: str, floor: str) -> None:
        (self.root / "migrations" / "compat.toml").write_text(
            f'alembic_revision = "{revision}"\ncompat_floor = "{floor}"\n'
        )

    def release(self, version: str, *, tag: bool = True) -> None:
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "honcho"\nversion = "{version}"\n'
        )
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", version)
        if tag:
            self.git("tag", f"v{version}")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Repo]:
    """A repo with v1.0.0 (one migration) and v1.1.0 (a second migration)."""
    # The script under test calls git with the process environment.
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    real_config = _real_repo_config()

    repo = Repo(tmp_path)
    repo.revision("base0001", None)
    repo.compat("base0001", "1.0.0")
    repo.release("1.0.0")
    repo.revision("head0001", "base0001")
    repo.compat("head0001", "1.0.0")
    repo.release("1.1.0")
    yield repo

    assert _real_repo_config() == real_config


def test_release_with_migration(repo: Repo) -> None:
    assert build_manifest(repo.root, "v1.1.0") == {
        "has_migration": True,
        "alembic_revision": "head0001",
        "compat_floor": "1.0.0",
    }


def test_code_only_release_carries_floor_over(repo: Repo) -> None:
    repo.release("1.2.0", tag=False)

    assert build_manifest(repo.root, "HEAD") == {
        "has_migration": False,
        "alembic_revision": "head0001",
        "compat_floor": "1.0.0",
    }


def test_code_only_release_can_raise_floor(repo: Repo) -> None:
    repo.compat("head0001", "1.1.0")
    repo.release("1.2.0", tag=False)

    manifest = build_manifest(repo.root, "HEAD")
    assert manifest["has_migration"] is False
    assert manifest["compat_floor"] == "1.1.0"


def test_version_override_picks_previous_tag(repo: Repo) -> None:
    # HEAD is tagged v1.1.0; releasing it as 1.1.0 diffs against v1.0.0.
    assert build_manifest(repo.root, "HEAD", (1, 1, 0))["has_migration"] is True
    assert build_manifest(repo.root, "HEAD", (1, 2, 0))["has_migration"] is False


def test_merge_revision_is_single_head(repo: Repo) -> None:
    repo.revision("fork0001", "base0001")
    repo.revision("merge001", ("head0001", "fork0001"))
    repo.compat("merge001", "1.1.0")
    repo.release("1.2.0", tag=False)

    assert alembic_heads(repo.root, "HEAD") == ["merge001"]
    assert build_manifest(repo.root, "HEAD")["alembic_revision"] == "merge001"


@pytest.mark.parametrize(
    ("new_revision", "floor", "message"),
    [
        pytest.param(
            ("next0001", "head0001"),
            "1.0.0",
            "declares alembic_revision 'head0001' but the head at HEAD is 'next0001'",
            id="stale-revision",
        ),
        pytest.param(
            ("fork0001", "base0001"),
            "1.0.0",
            "expected one alembic head",
            id="two-heads",
        ),
        pytest.param(
            None,
            "0.9.0",
            "compat_floor 0.9.0 is not a release tag",
            id="floor-not-a-tag",
        ),
        pytest.param(
            None,
            "1.3.0",
            "compat_floor 1.3.0 is newer than the release 1.2.0",
            id="floor-above-release",
        ),
        pytest.param(
            None, "v1.1", "compat_floor 'v1.1' is not X.Y.Z", id="floor-malformed"
        ),
    ],
)
def test_invalid_manifest_fails(
    repo: Repo, new_revision: tuple[str, str] | None, floor: str, message: str
) -> None:
    if new_revision is not None:
        repo.revision(*new_revision)
    repo.compat("head0001", floor)
    repo.release("1.2.0", tag=False)

    with pytest.raises(ManifestError, match=message):
        build_manifest(repo.root, "HEAD")


def test_floor_cannot_drop(repo: Repo) -> None:
    repo.compat("head0001", "1.1.0")
    repo.release("1.2.0")
    repo.compat("head0001", "1.0.0")
    repo.release("1.3.0", tag=False)

    with pytest.raises(
        ManifestError, match="lower than 1.1.0, the floor declared at v1.2.0"
    ):
        build_manifest(repo.root, "HEAD")


def test_floor_equal_to_release_needs_no_tag(repo: Repo) -> None:
    repo.compat("head0001", "1.2.0")
    repo.release("1.2.0", tag=False)

    assert build_manifest(repo.root, "HEAD")["compat_floor"] == "1.2.0"


def test_main_prints_json_and_reports_problems(
    repo: Repo, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--ref", "v1.1.0"], repo=repo.root) == 0
    assert json.loads(capsys.readouterr().out)["alembic_revision"] == "head0001"

    repo.revision("next0001", "head0001")
    repo.release("1.2.0", tag=False)
    assert main([], repo=repo.root) == 1
    assert "declares alembic_revision" in capsys.readouterr().err


def test_heads_match_alembic() -> None:
    assert alembic_heads(PROJECT_ROOT, "HEAD") == sorted(
        load_script_directory().get_heads()
    )
