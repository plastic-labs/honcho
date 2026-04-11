# https://pythonspeed.com/articles/base-image-python-docker-images/
# https://testdriven.io/blog/docker-best-practices/
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /bin/uv

# Set Working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Python optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-group dev

# Copy only requirements to cache them in docker layer
COPY uv.lock pyproject.toml /app/

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-group dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/app
ENV UV_CACHE_DIR=/tmp/uv-cache

# Create non-root user and set ownership
RUN addgroup --system app && adduser --system --group app && mkdir -p /tmp/uv-cache && chown -R app:app /app /tmp/uv-cache

COPY --chown=app:app src/ /app/src/
COPY --chown=app:app migrations/ /app/migrations/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app docker/ /app/docker/
COPY --chown=app:app alembic.ini /app/alembic.ini
# Copy config files - this will copy config.toml if it exists, and config.toml.example
COPY --chown=app:app config.toml* /app/

# Switch to non-root user
USER app

EXPOSE 8000

# HTTP healthcheck for the API server. Services that do not expose HTTP
# (e.g. the deriver worker) should disable this in their compose config:
#   healthcheck:
#     disable: true
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["fastapi", "run", "--host", "0.0.0.0", "src/main.py"]
