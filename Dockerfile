# syntax=docker/dockerfile:1

# https://pythonspeed.com/articles/base-image-python-docker-images/
# https://testdriven.io/blog/docker-best-practices/
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Python optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy only requirements to cache them in docker layer
COPY uv.lock pyproject.toml /app/

# Optionall include lancedb with:
#   docker build --build-arg INSTALL_LANCEDB=true .
ARG INSTALL_LANCEDB=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_LANCEDB" = "true" ]; then \
        uv sync --frozen --no-install-project --no-group dev --extra lancedb; \
    elif [ "$INSTALL_LANCEDB" = "false" ]; then \
        uv sync --frozen --no-install-project --no-group dev; \
    else \
        echo "INSTALL_LANCEDB must be 'true' or 'false'" >&2; \
        exit 2; \
    fi

FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create the runtime user before copying dependencies with their final owner.
# A recursive chown in a later layer would copy the whole virtualenv and nearly
# double the image size.
RUN addgroup --system app \
    && adduser --system --group app \
    && chown app:app /app \
    # Pre-create the LanceDB dir so a named volume mounted here inherits app
    # ownership instead of defaulting to root.
    && mkdir /app/lancedb_data \
    && chown app:app /app/lancedb_data

COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV HOME=/app

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

CMD ["fastapi", "run", "--host", "0.0.0.0", "src/main.py"]
