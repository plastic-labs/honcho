#!/usr/bin/env python3
"""
Run alembic tests selectively based on changed files.

Note that this script is actively used within our precommit hooks and should not be removed.
If this script is moved, the corresponding precommit hook will need to be updated.

This script determines which specific alembic tests to run based on:
1. If a test file changed, run the test for that revision
2. If a migration file changed, run the corresponding test

The alembic test system uses a parameterized test in test_pipeline.py that runs
for each revision. We filter these tests using pytest's -k flag with the revision IDs.

Usage: python scripts/run_alembic_tests_selective.py <file1> <file2> ...
"""

import re
import subprocess
import sys
from pathlib import Path


def extract_revision_id(filepath: Path) -> str | None:
    """Extract the revision ID from a migration or test filename.

    Migration files: {revision_id}_{description}.py
    Test files: test_{revision_id}_{description}.py

    Returns the revision_id (e.g., "05486ce795d5") or None if not found.
    """
    filename = filepath.name

    # Remove .py extension
    if not filename.endswith(".py"):
        return None

    filename = filename[:-3]

    # Remove test_ prefix if present
    if filename.startswith("test_"):
        filename = filename[5:]

    # Extract revision ID (first part before underscore)
    # Revision IDs are typically 12 characters of hex
    match = re.match(r"^([a-f0-9]{12})_", filename)
    if match:
        return match.group(1)

    return None


def main():
    if len(sys.argv) < 2:
        print("No files to check, skipping alembic tests")
        sys.exit(0)

    changed_files = [Path(f) for f in sys.argv[1:]]

    # Paths
    repo_root = Path(__file__).parent.parent
    migrations_dir = repo_root / "migrations" / "versions"
    tests_dir = repo_root / "tests" / "alembic" / "revisions"
    alembic_tests_dir = repo_root / "tests" / "alembic"

    # Collect revision IDs to test
    revision_ids: set[str] = set()
    run_full_suite = False

    for filepath in changed_files:
        filepath = Path(filepath).resolve()

        # Check if file is under tests/alembic (including subdirectories)
        # If it's not a revision-specific test file, run full suite
        if (
            filepath.parent == alembic_tests_dir
            or alembic_tests_dir in filepath.parents
        ) and not (filepath.parent == tests_dir and filepath.name.startswith("test_")):
            run_full_suite = True
            print(
                f"Infrastructure file changed: {filepath.name} -> will run full test suite"
            )
            continue

        # Case 1: Test file changed - extract its revision ID
        if filepath.parent == tests_dir and filepath.name.startswith("test_"):
            revision_id = extract_revision_id(filepath)
            if revision_id:
                revision_ids.add(revision_id)
                print(
                    f"Test file changed: {filepath.name} -> testing revision {revision_id}"
                )

        # Case 2: Migration file changed - extract its revision ID
        elif filepath.parent == migrations_dir:
            revision_id = extract_revision_id(filepath)
            if revision_id:
                revision_ids.add(revision_id)
                print(
                    f"Migration changed: {filepath.name} -> testing revision {revision_id}"
                )

    if run_full_suite:
        # Run full test suite without -k filter
        print("\nRunning full alembic test suite due to infrastructure file changes\n")
        cmd = [
            "uv",
            "run",
            "pytest",
            "tests/alembic/test_pipeline.py",
        ]
    elif revision_ids:
        # Build a -k expression to filter tests by revision ID
        # pytest -k "rev1 or rev2 or rev3"
        k_expression = " or ".join(sorted(revision_ids))

        print(
            f"\nRunning tests for {len(revision_ids)} revision(s): {', '.join(sorted(revision_ids))}"
        )
        print()

        # Run pytest on test_pipeline.py with -k filter
        cmd = [
            "uv",
            "run",
            "pytest",
            "tests/alembic/test_pipeline.py",
            "-k",
            k_expression,
        ]
    else:
        print("No alembic tests to run")
        sys.exit(0)

    result = subprocess.run(cmd, cwd=repo_root)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
