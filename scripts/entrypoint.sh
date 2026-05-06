#!/bin/bash
set -e

echo "=== Honcho API Server ==="
echo "Waiting for database to be ready..."

# Run Alembic migrations (idempotent — safe to run on every startup)
echo "Running database migrations..."
cd /app && uv run alembic upgrade head

echo "Starting Honcho API server..."
exec uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
