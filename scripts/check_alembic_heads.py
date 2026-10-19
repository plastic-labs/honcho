#!/usr/bin/env uv run python
"""
Script that fails when the alembic migration graph has more than one head.
Note that this script is actively used within CI and our precommit hooks and should not be removed.
If this script is moved, the corresponding workflow job and precommit hook will need to be updated.
"""

import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def load_script_directory(script_location: Path | None = None) -> ScriptDirectory:
    """Load the migration graph from alembic.ini without touching env.py or a database."""
    cfg = Config(str(ALEMBIC_INI))
    location = script_location or PROJECT_ROOT / cfg.get_main_option(
        "script_location", "migrations"
    )
    cfg.set_main_option("script_location", str(location))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return ScriptDirectory.from_config(cfg)


def main(script_location: Path | None = None) -> int:
    script = load_script_directory(script_location)
    heads = script.get_heads()
    if len(heads) <= 1:
        print(f"Single alembic head: {heads[0] if heads else '<none>'}")
        return 0

    print(f"Found {len(heads)} alembic heads, expected 1:", file=sys.stderr)
    for head in sorted(heads):
        revision = script.get_revision(head)
        assert revision is not None
        print(f" - {revision.revision}", file=sys.stderr)
        print(f"     down_revision: {revision.down_revision}", file=sys.stderr)
        print(f"     path: {revision.path}", file=sys.stderr)
    print(
        "\nRebase onto main and chain the new migration's down_revision off the "
        + "current head, or add a merge revision with `alembic merge heads`.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
