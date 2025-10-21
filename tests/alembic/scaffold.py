"""Utility to scaffold migration hook test modules."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = PROJECT_ROOT / "migrations" / "versions"
TESTS_REVISION_DIR = Path(__file__).resolve().parent / "revisions"
REVISION_INIT_PATH = TESTS_REVISION_DIR / "__init__.py"

TEMPLATE = '''"""Hooks for revision {revision}{slug_note}."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("{revision}")
def prepare_{identifier}(_verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to {revision}."""


@register_after_upgrade("{revision}")
def verify_{identifier}(_verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of {revision}."""
'''

HEADER = '"""Register revision-specific hooks for migration verification."""'


@dataclass(slots=True)
class MigrationInfo:
    revision: str
    slug: str
    path: Path

    @property
    def identifier(self) -> str:
        """Return a Python-safe identifier derived from the migration slug."""

        base = re.sub(r"[^0-9a-zA-Z]+", "_", self.slug)
        base = base.strip("_").lower() or "revision"
        if base[0].isdigit():
            base = f"revision_{base}"
        return base

    @property
    def test_filename(self) -> str:
        return f"test_{self.path.stem}.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate a revision test stub with before/after upgrade hooks.")
    )
    parser.add_argument(
        "revision",
        help="Revision id (e.g. a1b2c3d4e5f6) or path to a migration file.",
    )
    parser.add_argument(
        "--migrations-dir",
        default=MIGRATIONS_DIR,
        type=Path,
        help="Path to the Alembic migrations directory (defaults to migrations/versions).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target test file if it already exists.",
    )
    return parser.parse_args()


def resolve_migration(path_or_revision: str, migrations_dir: Path) -> MigrationInfo:
    candidate = Path(path_or_revision)
    if candidate.suffix == ".py" and candidate.exists():
        return MigrationInfo(
            revision=candidate.stem.split("_", 1)[0],
            slug=candidate.stem.split("_", 1)[1] if "_" in candidate.stem else "",
            path=candidate.resolve(),
        )

    matches = sorted(migrations_dir.glob(f"{path_or_revision}_*.py"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find migration matching {path_or_revision!r} in {migrations_dir}."
        )
    if len(matches) > 1:
        options = ", ".join(match.name for match in matches)
        raise ValueError(
            f"Revision prefix {path_or_revision!r} matches multiple migrations: {options}."
            + " Provide a more specific revision or path."
        )
    match = matches[0]
    stem = match.stem
    if "_" in stem:
        revision, slug = stem.split("_", 1)
    else:
        revision, slug = stem, ""
    return MigrationInfo(revision=revision, slug=slug, path=match.resolve())


def build_template(info: MigrationInfo) -> str:
    slug_note = f" ({info.slug})" if info.slug else ""
    content = TEMPLATE.format(
        revision=info.revision,
        identifier=info.identifier,
        slug_note=slug_note,
    )
    return dedent(content).rstrip() + "\n"


def write_stub(info: MigrationInfo, overwrite: bool) -> Path:
    TESTS_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    target = TESTS_REVISION_DIR / info.test_filename
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Test module {target} already exists. Use --overwrite to replace it."
        )
    target.write_text(build_template(info), encoding="utf-8")
    return target


def refresh_revision_init() -> None:
    modules = sorted(
        path.stem for path in TESTS_REVISION_DIR.glob("test_*.py") if path.is_file()
    )
    import_block = "\n".join(f"    {module}," for module in modules)
    all_block = "\n".join(f'    "{module}",' for module in modules)
    new_content = (
        dedent(
            f"""{HEADER}

from . import (
{import_block}
)

__all__ = [
{all_block}
]
"""
        ).rstrip()
        + "\n"
    )
    REVISION_INIT_PATH.write_text(new_content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    info = resolve_migration(args.revision, args.migrations_dir)
    target = write_stub(info, args.overwrite)
    refresh_revision_init()
    print(f"Created {target.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
