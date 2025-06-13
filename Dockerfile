# https://pythonspeed.com/articles/base-image-python-docker-images/
# https://testdriven.io/blog/docker-best-practices/
FROM python:3.11-slim-bullseye

COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

# Set Working directory
WORKDIR /app

RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy only requirements to cache them in docker layer
COPY uv.lock pyproject.toml /app/

# Install dependencies (no cache mount)
RUN uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

COPY --chown=app:app src/ /app/src/
COPY --chown=app:app migrations/ /app/migrations/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app alembic.ini /app/alembic.ini

EXPOSE 8000

# https://stackoverflow.com/questions/29663459/python-app-does-not-print-anything-when-running-detached-in-docker
CMD ["fastapi", "run", "--host", "0.0.0.0", "src/main.py"]
