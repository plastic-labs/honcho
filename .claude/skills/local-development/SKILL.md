# Local Honcho Development

## What This Skill Does

This skill guides you through setting up and running Honcho locally for development. Use it when you need to get the server running, run the deriver, execute migrations, run tests, or troubleshoot common setup issues.

**Trigger phrases:**
- "Help me set up Honcho locally"
- "How do I run the deriver?"
- "My database connection isn't working"
- "How do I run the tests?"
- "How do I create a migration?"
- "How do I install pre-commit hooks?"

---

## Phase 1: Environment Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) 0.4.9 or higher
- Docker with Compose v2 (recommended for the database) — use `docker compose` not `docker-compose`

### Install Dependencies

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
uv sync
```

Activate the virtual environment:

```bash
# Unix/macOS
source .venv/bin/activate

# Windows (Command Prompt): .venv\Scripts\activate
# Windows (PowerShell):     .venv\Scripts\activate.ps1
```

> **Note:** All commands in this guide use `uv run`, which automatically uses the virtual environment — explicit activation is optional.

### Configure Environment Variables

```bash
cp .env.template .env
```

Open `.env` and set the required values:

| Variable | Required | Description |
|----------|----------|-------------|
| `DB_CONNECTION_URI` | Yes | PostgreSQL connection string — must use `postgresql+psycopg` prefix |
| `LLM_GEMINI_API_KEY` | One required (default) | Honcho uses Gemini by default — get a key at [aistudio.google.com](https://aistudio.google.com) |
| `LLM_ANTHROPIC_API_KEY` | One required (alternative) | Use instead of Gemini — get a key at [console.anthropic.com](https://console.anthropic.com) |
| `LLM_OPENAI_API_KEY` | One required (alternative) | Use instead of Gemini — get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `LLM_GROQ_API_KEY` | One required (alternative) | Use instead of Gemini — get a key at [console.groq.com/keys](https://console.groq.com/keys) |
| `AUTH_USE_AUTH` | No | Defaults to `false`. If set to `true`, every request requires a valid auth token and `AUTH_JWT_SECRET` must also be set |
| `AUTH_JWT_SECRET` | Conditional | Required when `AUTH_USE_AUTH=true` — JWT signing secret for auth tokens |
| `LLM_PROVIDER` | No | Override the default provider — valid values: `google`, `anthropic`, `openai`, `groq` |
| `LLM_OPENAI_COMPATIBLE_API_KEY` | Optional | API key for OpenAI-compatible endpoints (e.g. vLLM, local models) |
| `LLM_OPENAI_COMPATIBLE_BASE_URL` | Optional | Base URL for the OpenAI-compatible endpoint — required when `LLM_OPENAI_COMPATIBLE_API_KEY` is set |
| `SENTRY_ENABLED` | No | Set to `false` for local development |

> **Important:** Honcho requires at least one LLM API key. You can set multiple keys, but only the provider matching `LLM_PROVIDER` will be used (default: `google`/Gemini). If you see errors about missing model configuration, an LLM key is the most likely cause.

Minimal local development `.env` (Gemini example):

```env
DB_CONNECTION_URI=postgresql+psycopg://user:password@localhost:5432/honcho
LLM_GEMINI_API_KEY=your-gemini-key-here
SENTRY_ENABLED=false
```

### Set Up the Database

Using Docker (recommended):

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d database
```

Or use [Supabase](https://supabase.com) as an alternative — set `DB_CONNECTION_URI` to your Supabase connection string.

---

## Phase 2: Running Honcho

### Run Database Migrations

Always run migrations before starting the server for the first time or after pulling new changes:

```bash
uv run alembic upgrade head
```

### Start the API Server

```bash
uv run fastapi dev src/main.py
```

The API will be available at `http://localhost:8000`. The interactive docs are at `http://localhost:8000/docs`.

### Start the Deriver (Background Worker)

The deriver processes messages asynchronously and builds user representations. Run it in a **separate terminal**:

```bash
uv run python -m src.deriver
```

> **Note:** Both the API server and the deriver must be running for Honcho's memory and reasoning features to work. The API accepts messages; the deriver processes them in the background.

To verify the deriver is working, send a message via the API and watch the deriver terminal for log output showing it picked up the task. If the deriver log is silent, check that a valid LLM API key is set in `.env`.

### Verify Your Setup

With both the API server and deriver running, confirm everything is working:

```bash
curl http://localhost:8000/docs
```

If the interactive docs page loads, the API server is running correctly. You can also confirm the server started without errors by checking the terminal where you ran `uv run fastapi dev src/main.py`.

### Run Everything with Docker

If you prefer to run the full stack via Docker:

```bash
cp .env.template .env
cp docker-compose.yml.example docker-compose.yml
docker compose up
```

> **Note:** The API container automatically runs database migrations on startup via `docker/entrypoint.sh`. You do not need to run `alembic upgrade head` manually when using Docker.

### Clean Up

Stop the API server and deriver with `Ctrl+C` in each terminal.

Stop Docker containers:

```bash
docker compose down
```

Deactivate the virtual environment:

```bash
deactivate
```

---

## Phase 3: Development Tasks

### Run Tests

```bash
uv run pytest
```

Run a specific test file:

```bash
uv run pytest tests/test_<filename>.py
```

Run with verbose output:

```bash
uv run pytest -v
```

### Create a New Migration

After modifying a SQLAlchemy model in `src/`:

```bash
uv run alembic revision --autogenerate -m "description of your change"
uv run alembic upgrade head
```

Review the generated migration file in `migrations/versions/` before applying it.

### Set Up Pre-commit Hooks

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

Run hooks manually against all files:

```bash
uv run pre-commit run --all-files
```

Hooks enforce formatting, linting, type checking, and conventional commit messages.

### Code Standards

Follow these conventions when contributing:

- Format code: `uv run ruff format .`
- Lint code: `uv run ruff check .`
- Type check: `uv run basedpyright` (a strict fork of pyright — stricter than standard mypy)
- Use type hints on all function signatures
- Write Google-style docstrings
- Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

---

## Phase 4: Troubleshooting

### ModuleNotFoundError at startup

**Symptom:** `ModuleNotFoundError: No module named 'honcho'` or similar import errors

**Fix:**

```bash
uv sync --reinstall
```

This reinstalls all dependencies from scratch, which resolves corrupted or incomplete installs.

### Database connection error

**Symptom:** `sqlalchemy.exc.OperationalError` or `could not connect to server`

**Fix:**
1. Confirm Docker is running: `docker compose up -d database`
2. Check `DB_CONNECTION_URI` in `.env` — must use `postgresql+psycopg` (not `postgresql+psycopg2`)
3. Verify the database is healthy: `docker compose ps`

### Missing LLM API key

**Symptom:** `KeyError: LLM_GEMINI_API_KEY` or model-related errors at startup or when the deriver runs

**Fix:**
Honcho defaults to Gemini. Add your key to `.env`:

```env
LLM_GEMINI_API_KEY=your-key-here
```

Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).

If you prefer a different provider, set the corresponding key and set `LLM_PROVIDER` in `.env` to the matching value (`anthropic`, `openai`, `groq`, or `google` for Gemini).

### Deriver not processing messages

**Symptom:** Messages are stored but representations never update; `peer.chat()` returns empty or generic answers

**Fix:**
1. Confirm the deriver is running in a separate terminal: `uv run python -m src.deriver`
2. Confirm a valid LLM API key is set — the deriver calls the LLM to build representations
3. Check deriver logs for errors

### Migration errors

**Symptom:** `alembic.util.exc.CommandError` or `relation does not exist`

**Fix:**

```bash
uv run alembic upgrade head
```

If you have conflicts or a dirty migration state:

> **Warning:** `alembic downgrade base` drops all tables and deletes all data. Only use this in a local development database.

```bash
uv run alembic downgrade base
uv run alembic upgrade head
```

### Port already in use

**Symptom:** `ERROR: [Errno 48] Address already in use` (macOS) or `ERROR: [Errno 98] Address already in use` (Linux) on port 8000

**Fix:**

Unix/macOS:

```bash
lsof -ti:8000 | xargs kill -9
```

Windows (Command Prompt):

```cmd
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8000') do taskkill /F /PID %a
```

Windows (PowerShell):

```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

Then restart the server:

```bash
uv run fastapi dev src/main.py
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Run migrations | `uv run alembic upgrade head` |
| Start API server | `uv run fastapi dev src/main.py` |
| Start deriver | `uv run python -m src.deriver` |
| Run tests | `uv run pytest` |
| Format code | `uv run ruff format .` |
| New migration | `uv run alembic revision --autogenerate -m "description"` |
| Run all pre-commit hooks | `uv run pre-commit run --all-files` |

---

## Resources

- [Honcho Documentation](https://docs.honcho.dev)
- [Contributing Guide](../../../CONTRIBUTING.md)
- [API Reference](https://docs.honcho.dev/v3/api-reference/introduction)
- [Discord Community](https://discord.com/invite/honcho)
