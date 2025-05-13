# honcho/scripts/provision_db.py
import os
import sys

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

## First import the models to register them with Base
from src import (  # noqa: E402
    models,  # noqa: F401
)  # This registers all models with Base Now you can import from src
from src.db import scaffold_db  # noqa: E402

if __name__ == "__main__":
    scaffold_db()
    print("Database created")
