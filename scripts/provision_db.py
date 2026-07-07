# honcho/scripts/provision_db.py
import asyncio
import logging
import os
import subprocess
import sys

# Add the project root to the path
# This assumes the script is run from the scripts directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.db import init_db  # noqa: E402

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("Initializing database using Alembic migrations...")
    asyncio.run(init_db())
    print("Database initialized successfully")

    # Align pgvector dimensions after migration (migration hardcodes Vector(1536)).
    # This is a one-shot DDL step that runs as part of DB provisioning (not per
    # replica in the entrypoint) to avoid multiple replicas racing on ALTER COLUMN.
    # configure_embeddings.py is safe to re-run: it skips a no-op when dimensions
    # already match and refuses with SystemExit when tables contain data.
    print("Configuring embedding dimensions...")
    configure_py = os.path.join(project_root, "scripts", "configure_embeddings.py")
    result = subprocess.run(
        [sys.executable, configure_py, "--yes"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"configure_embeddings exited with {result.returncode} (non-fatal):")
        print(result.stderr.strip() or result.stdout.strip())
