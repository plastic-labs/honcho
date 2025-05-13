# honcho/scripts/provision_db.py
import os
import sys

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.db import init_db  # noqa: E402

if __name__ == "__main__":
    print("Initializing database using Alembic migrations...")
    init_db()
    print("Database initialized successfully")
