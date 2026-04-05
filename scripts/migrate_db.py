# honcho/scripts/migrate_db.py
import os
import sys

from alembic import command
from alembic.config import Config

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


if __name__ == "__main__":
    # run alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
