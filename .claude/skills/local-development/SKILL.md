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

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) 0.4.9 or higher
- Docker (recommended for the database)

### Install Dependencies

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
uv sync
source .venv/bin/activate
```

### Configure Environment Variables

```bash
cp .env.template .env
```

Open `.env` and set the required values:

| Variable | Required | Description |
|----------|----------|-------------|
| `DB_CONNECTION_URI` | Yes | PostgreSQL connection string — must use `postgresql+psycopg` prefix |
| `LLM_GEMINI_API_KEY` | Yes (default) | Honcho uses Gemini by default — get a key at [aistudio.google.com](https://aistudio.google.com) |
| `LLM_ANTHROPIC_API_KEY` | Alternative | Use instead of Gemini if preferred |
| `LLM_OPENAI_API_KEY` | Alternative | Use instead of Gemini if preferred |
| `LLM_GROQ_API_KEY` | Alternative | Use instead of Gemini if preferred |
| `AUTH_USE_AUTH` | No | Set to `false` for local development |
| `SENTRY_ENABLED` | No | Set to `false` for local development |

> **Important:** Honcho requires at least one LLM API key to function. The default is Gemini (`LLM_GEMINI_API_KEY`). If you see errors about missing model configuration, this is the most likely cause.

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

### Run Everything with Docker

If you prefer to run the full stack via Docker:

```bash
cp .env.template .env
cp docker-compose.yml.example docker-compose.yml
docker compose up
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
uv add --dev pre-commit
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

Run hooks manually against all files:

```bash
uv run pre-commit run --all-files
```

Hooks enforce PEP 8, Black formatting, type hints, and conventional commit messages.

### Code Standards

Follow these conventions when contributing:

- Format with [Black](https://black.readthedocs.io/) (`uv run black .`)
- Use type hints on all function signatures
- Write Google-style docstrings
- Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

---

## Phase 4: Troubleshooting

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
```
LLM_GEMINI_API_KEY=your-key-here
```
Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com).

If you prefer a different provider, set the corresponding key and update `LLM_PROVIDER` in `.env` if that setting is present.

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
```bash
uv run alembic downgrade base
uv run alembic upgrade head
```

### Port already in use

**Symptom:** `ERROR: [Errno 48] Address already in use` on port 8000

**Fix:**
```bash
lsof -ti:8000 | xargs kill -9
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
| Format code | `uv run black .` |
| New migration | `uv run alembic revision --autogenerate -m "description"` |
| Run all pre-commit hooks | `uv run pre-commit run --all-files` |

---

## Resources

- [Honcho Documentation](https://docs.honcho.dev)
- [Contributing Guide](../../CONTRIBUTING.md)
- [API Reference](https://docs.honcho.dev/v3/api-reference/introduction)
- [Discord Community](https://discord.com/invite/honcho)
