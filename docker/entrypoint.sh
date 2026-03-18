#!/bin/sh
set -e

echo "Running database migrations..."
/app/.venv/bin/python scripts/provision_db.py

echo "Starting API server..."
exec /app/.venv/bin/fastapi run --host 0.0.0.0 src/main.py
