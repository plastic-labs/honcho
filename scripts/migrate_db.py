# honcho/scripts/migrate_db.py
import os
import sys

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from src.db import scaffold_db  # noqa: E402

if __name__ == "__main__":
    scaffold_db()
    # run alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
