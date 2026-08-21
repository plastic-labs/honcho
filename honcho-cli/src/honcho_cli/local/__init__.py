"""Local Honcho stack: profiles, Compose rendering, Docker, health checks."""

from __future__ import annotations

DEFAULT_PROFILE = "local"
DEFAULT_API_PORT = 8000
DEFAULT_DB_PORT = 5432
DEFAULT_REDIS_PORT = 6379
DEFAULT_IMAGE = "ghcr.io/plastic-labs/honcho:latest"
DEFAULT_HEALTH_TIMEOUT = 180

STACK_SERVICES = ("api", "deriver", "database", "redis")
