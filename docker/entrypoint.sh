#!/bin/sh
set -e

echo "Running database migrations..."
uv run python scripts/provision_db.py

echo "Starting API server..."
exec fastapi run --host 0.0.0.0 src/main.py
