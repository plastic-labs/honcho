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
ENV UV_PYTHON_DOWNLOADS=never

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-group dev

# Copy only requirements to cache them in docker layer
COPY uv.lock pyproject.toml /app/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Sync the project into a virtual environment, using the frozen lockfile
COPY . /app

# Install the project itself (extras omitted for runtime image)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-group dev

# Pre-seed tiktoken's o200k_base cache so containers do not depend on live DNS
# resolution to openaipublic.blob.core.windows.net at first startup.
RUN python - <<'PY'
import hashlib
import pathlib
import urllib.request
url = 'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
cache_key = hashlib.sha1(url.encode()).hexdigest()
cache_path = pathlib.Path('/tmp/data-gym-cache') / cache_key
cache_path.parent.mkdir(parents=True, exist_ok=True)
if not cache_path.exists():
    with urllib.request.urlopen(url) as resp:
        cache_path.write_bytes(resp.read())
print(f'cached {url} -> {cache_path}')
PY
ENV HOME=/app
ENV UV_CACHE_DIR=/tmp/uv-cache

# Create non-root user and set ownership
RUN addgroup --system app && adduser --system --group app && mkdir -p /tmp/uv-cache && chown -R app:app /app /tmp/uv-cache

COPY --chown=app:app src/ /app/src/
COPY --chown=app:app migrations/ /app/migrations/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app alembic.ini /app/alembic.ini
# Copy config files - this will copy config.toml if it exists, and config.toml.example
COPY --chown=app:app config.toml* /app/

# Switch to non-root user
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json')" || exit 1

CMD ["fastapi", "run", "--host", "0.0.0.0", "src/main.py"]
