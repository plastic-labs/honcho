---
name: local-development
description: Set up and run Honcho locally for development. Use when configuring the local environment, running the API server, deriver, migrations, or troubleshooting local development issues.
---

# Local Honcho Development

This skill helps you set up and run Honcho locally for development.

## Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/plastic-labs/honcho.git
   cd honcho
   ```

   ```

2. **Install dependencies:**
   ```bash
   # Install uv (Python package manager) if needed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

   ```

3. **Set up environment:**
   ```bash
   # Copy and configure environment template
   cp .env.template .env
   
   # Edit .env with your configuration
   # You'll need at minimum:
   # - HONCHO_API_KEY (get from https://app.honcho.dev)
   # - Database settings (see Database Setup below)
   ```

   ```

4. **Set up the database:**
   ```bash
   # Option 1: Use Docker (recommended for local dev)
   docker compose up -d
   
   # Option 2: Use Supabase
   # Update .env with your Supabase connection details
   ```

   ```

5. **Run migrations:**
   ```bash
   uv run alembic upgrade head
   ```

   ```

6. **Run Honcho:**
   ```bash
   # In one terminal - API server
   uv run fastapi dev src/main.py
   
   # In another terminal - Deriver (background reasoning engine)
   uv run python -m src.deriver
   ```

   ```

---

## Environment Setup

### Installing Dependencies

Honcho uses `uv` for dependency management (faster than pip/poetry).

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all project dependencies
uv sync

# Add a new dependency (if needed during dev)
uv add <package-name>
```

```

### Setting Up .env

The `.env.template`' file contains all available configuration options:

```bash
# Copy the template
cp .env.template .env

# Edit .env with your values
nano .env
```

```

**Required variables:**

- `HONCHO_API_KEY` - Get this from https://app.honcho.dev
- `DATABASE_URL` - PostgreSQL connection string

**Optional but recommended:**

- `OPENAI_API_KEY` - If using OpenAI models
- `ANTHROPIC_API_KEY` - If using Anthropic models

### Database Setup

**Option 1: Docker Compose (Recommended for Local)**

```bash
# Start the database container
docker compose up -d

# Check it's running
docker compose ps

# View logs
docker compose logs -f

# Stop the database
docker compose down

# Stop and remove volumes (reset database)
docker compose down -v
```

```

**Option 2: Supabase**

1. Create a free Supabase project at https://supabase.com
2. Go to Settings → Database → Connection string
3. Copy the "URI" format connection string
4. Add to `.env`:
   ```

   ```
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres
   ```

   ```

**Option 3: Local PostgreSQL**

```bash
# Using docker (simpler)
docker run -d \
  --name honcho-postgres \
  -e POSTGRES_PASSWORD=honcho \
  -e POSTGRES_DB=honcho \
  -p 5432:5432 \
  postgres:16

# Then set DATABASE_URL in .env:
# DATABASE_URL=postgresql://postgres:honcho@localhost:5432/honcho
```

```

### Running Migrations

After setting up the database, run migrations to create the schema:

```bash
# Run all pending migrations
uv run alembic upgrade head

# Check migration status
uv run alembic current

# View migration history
uv run alembic history

# Rollback one migration
uv run alembic downgrade -1
```

```

---

## Running Honcho

### Starting the API Server

The API server handles HTTP requests:

```bash
# Development mode (auto-reload on changes)
uv run fastapi dev src/main.py

# Production mode (use uvicorn directly)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# On a different port
uv run fastapi dev src/main.py --port 8080
```

```

The API will be available at http://localhost:8000

- **API Documentation:** http://localhost:8000/docs
- **Interactive API Explorer:** http://localhost:8000/redoc

### Starting the Deriver

The deriver is the background reasoning engine that processes messages asynchronously. It must be running for Honcho's memory system to work:

```bash
# Run the deriver
uv run python -m src.deriver

# With logging visible
uv run python -m src.deriver --log-level INFO
```

```

**The deriver needs to run continuously.** It polls for new messages and processes them with Honcho's reasoning models.

### Development Workflow

**Typical local development setup:**

1. Open Terminal 1:
   ```bash
   uv run fastapi dev src/main.py
   ```

   ```

2. Open Terminal 2:
   ```bash
   uv run python -m src.deriver
   ```

   ```

3. Open Terminal 3 (for running tests, etc.):
   ```bash
   # Run tests
   uv run pytest
   
   # Or make changes and watch reload in Terminal 1
   ```

   ```

---

## Development Tasks

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_specific.py

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run with verbose output
uv run pytest -v

# Run only fast tests (skip slow integration tests)
uv run pytest -m "not slow"
```

```

### Pre-commit Hooks

Honcho uses pre-commit for code quality checks:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit manually on all files
uv run pre-commit run --all-files

# Run on staged files
uv run pre-commit run
```

```

Pre-commit hooks will run:
- Code formatting (black, isort)
- Linting (pylint, flake8)
- Type checking (mypy)
- Markdown linting

### Creating New Migrations

When you change the database schema, create a migration:

```bash
# Create a migration (autogenerate from model changes)
uv run alembic revision --autogenerate -m "description of changes"

# Create a blank migration (for manual SQL)
uv run alembic revision -m "description of changes"

# Edit the migration file
nano migrations/versions/<timestamp>_description_of_changes.py

# Apply the migration
uv run alembic upgrade head
```

```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name
```

```

---

## Troubleshooting

### Database Connection Issues

**Problem:** `psycopg2.OperationalError: could not connect to server`

**Solutions:**

1. **Check if Docker database is running:**
   ```bash
   docker compose ps
   # If not running, start it:
   docker compose up -d
   ```

   ```

2. **Verify DATABASE_URL in .env:**
   - Make sure the format is correct: `postgresql://user:password@host:port/database`
   - Check for typos or missing values
   - Ensure the port matches (usually 5432)

3. **Test the connection manually:**
   ```bash
   uv run python -c "import os; from sqlalchemy import create_engine; engine = create_engine(os.getenv('DATABASE_URL')); print(engine.connect())"
   ```

   ```

4. **Check docker logs:**
   ```bash
   docker compose logs postgres
   ```

   ```

### Missing API Key Errors

**Problem:** `HONCHO_API_KEY` not found or invalid

**Solutions:**

1. **Get an API key:**
   - Go to https://app.honcho.dev
   - Sign up/log in
   - Create a workspace
   - Copy your API key from Settings

2. **Add to .env:**
   ```

   ```
   HONCHO_API_KEY=your_api_key_here
   ```

   ```

3. **Verify it's loaded:**
   ```bash
   uv run python -c "import os; print(os.getenv('HONCHO_API_KEY'))"
   ```

   ```

### Deriver Not Processing Messages

**Problem:** Messages are stored but not being processed/reasoned about

**Solutions:**

1. **Check if deriver is running:**
   ```bash
   ps aux | grep deriver
   ```

   ```

2. **Check deriver logs for errors:**
   - Look at the terminal where deriver is running
   - Check for API key errors, database errors, or connection issues

3. **Verify HONCHO_API_KEY is valid:**
   - Invalid keys will cause deriver to fail silently
   - Test with a simple API call from Python

4. **Check if messages are being stored:**
   - Query the database directly to see if messages exist
   - Look at the `messages` table

### Import Errors

**Problem:** `ModuleNotFoundError: No module named '...'`

**Solutions:**

1. **Reinstall dependencies:**
   ```bash
   uv sync
   ```

   ```

2. **Check PYTHONPATH:**
   ```bash
   # Should include src/ directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

   ```

3. **Verify installation:**
   ```bash
   uv run python -c "import honcho; print(honcho.__file__)"
   ```

   ```

### Port Already in Use

**Problem:** `Address already in use` on port 8000

**Solutions:**

1. **Kill the process using the port:**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

   ```

2. **Or use a different port:**
   ```bash
   uv run fastapi dev src/main.py --port 8080
   ```

   ```

### Migration Conflicts

**Problem:** Migration fails or database is out of sync

**Solutions:**

1. **Check current migration:**
   ```bash
   uv run alembic current
   ```

   ```

2. **Stamp database to current revision (use with caution):**
   ```bash
   uv run alembic stamp head
   ```

   ```

3. **Reset database (destructive!):**
   ```bash
   docker compose down -v
   docker compose up -d
   uv run alembic upgrade head
   ```

   ```

### Slow Performance

**Problem:** API responses are slow

**Solutions:**

1. **Check deriver performance:**
   - Deriver might be backlogged
   - Look at deriver logs for processing times

2. **Database indexing:**
   - Ensure database indexes exist
   - Run `EXPLAIN ANALYZE` on slow queries

3. **API caching:**
   - Consider adding caching for frequently accessed data
   - Use Redis for session caching (optional)

4. **Connection pooling:**
   - Check database connection pool settings in config

---

## Useful Commands Reference

```bash
# Dependencies
uv sync                                    # Install dependencies
uv add <package>                           # Add package
uv remove <package>                        # Remove package

# Database
docker compose up -d                       # Start database
docker compose down                        # Stop database
uv run alembic upgrade head                # Run migrations
uv run alembic revision -m "msg"           # Create migration
uv run alembic downgrade -1                # Rollback migration

# Running Honcho
uv run fastapi dev src/main.py             # API server (dev)
uv run uvicorn src.main:app                # API server (prod)
uv run python -m src.deriver               # Deriver

# Development
uv run pytest                              # Run tests
uv run pytest -v                           # Verbose tests
uv run pytest --cov=src                    # Coverage report
uv run pre-commit run --all-files           # Run pre-commit hooks
uv run black .                             # Format code
uv run mypy src/                           # Type checking

# Troubleshooting
docker compose logs -f                     # View logs
lsof -ti:8000                             # Find process on port
```

```

---

## Getting Help

- **Documentation:** https://docs.honcho.dev
- **GitHub Issues:** https://github.com/plastic-labs/honcho/issues
- **Discord:** https://discord.gg/plasticlabs
- **Email:** support@honcho.dev

For specific bugs or feature requests, open an issue on GitHub with:
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages
