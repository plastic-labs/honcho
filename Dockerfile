# https://pythonspeed.com/articles/base-image-python-docker-images/
# https://testdriven.io/blog/docker-best-practices/
FROM python:3.11-slim-bullseye

# Copy the uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv

# Set working directory
WORKDIR /app

# Create and use non-root user
RUN addgroup --system app && adduser --system --group app
RUN chown -R app:app /app
USER app

# Enable Python bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Use copy mode instead of linking
ENV UV_LINK_MODE=copy

# Copy only requirement files first to leverage Docker layer caching
COPY uv.lock pyproject.toml /app/

# Install dependencies (no cache mount here)
RUN uv sync --frozen --no-dev

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files (own by app user)
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app migrations/ /app/migrations/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app alembic.ini /app/alembic.ini

# Expose the application port
EXPOSE 8000

# Run FastAPI app
CMD ["fastapi", "run", "--host", "0.0.0.0", "src/main.py"]
