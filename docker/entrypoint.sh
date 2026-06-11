#!/bin/sh
set -e

echo "Running database migrations..."
/app/.venv/bin/python scripts/provision_db.py

echo "Starting API server on port ${PORT:-8000}..."
exec /app/.venv/bin/fastapi run --host 0.0.0.0 --port "${PORT:-8000}" src/main.py
