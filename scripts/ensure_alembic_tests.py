#!/usr/bin/env uv run python
"""
Script that validates that all alembic migration revisions have a corresponding test file.
Note that this script is actively used within our precommit hooks and should not be removed.
If this script is moved, the corresponding precommit hook will need to be updated.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = PROJECT_ROOT / "migrations" / "versions"
TESTS_DIR = PROJECT_ROOT / "tests" / "alembic" / "revisions"


def main() -> int:
    missing: list[str] = []

    # Gather migration base names (without extension)
    migration_files = [
        p for p in MIGRATIONS_DIR.glob("*.py") if p.name != "__init__.py"
    ]
    migration_basenames = {p.stem for p in migration_files}
    print(f"Migration basenames: {migration_basenames}")

    # Gather test file base names mapped back to migration names by stripping leading 'test_'
    test_files = [p for p in TESTS_DIR.glob("test_*.py") if p.name != "__init__.py"]
    test_targets = {p.stem.removeprefix("test_") for p in test_files}

    for migration_basename in sorted(migration_basenames):
        if migration_basename not in test_targets:
            missing.append(migration_basename)
    if missing:
        print(
            "Missing Alembic tests for the following migration revisions:",
            file=sys.stderr,
        )
        for name in missing:
            print(f" - {name}", file=sys.stderr)
        print(
            "\nExpected test files under tests/alembic/revisions named as:",
            file=sys.stderr,
        )
        for name in missing:
            print(f" - tests/alembic/revisions/test_{name}.py", file=sys.stderr)
        print("\nScaffold helper commands:", file=sys.stderr)
        for name in missing:
            print(
                f" - python -m tests.alembic.scaffold {name.split('_', 1)[0]}",
                file=sys.stderr,
            )
        return 1

    # Optional: warn if there are tests without corresponding migrations (stale tests)
    stale_tests = sorted(test_targets - migration_basenames)
    if stale_tests:
        print(
            "Warning: Found tests without corresponding migration files (stale?):",
            file=sys.stderr,
        )
        for name in stale_tests:
            print(f" - tests/alembic/revisions/test_{name}.py", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
