# honcho/scripts/migrate_db.py
import os
import sys

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.migrate import main  # noqa: E402

if __name__ == "__main__":
    main()
