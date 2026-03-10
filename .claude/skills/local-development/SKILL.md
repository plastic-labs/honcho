---
name: local-development
description: Set up, run, and troubleshoot Honcho locally for development. Use when cloning the repo for the first time, configuring `.env`, starting Postgres, running migrations, launching the API server and deriver, running tests, or debugging local setup issues.
allowed-tools: Read, Glob, Grep, Bash(uv:*), Bash(docker:*), Bash(psql:*), Edit, Write, AskUserQuestion
---

# Honcho Local Development

## What this skill is for

Use this skill when someone asks things like:

- "Help me set up Honcho locally"
- "How do I run the deriver?"
- "How do I configure the database?"
- "Why is my local Honcho server failing to start?"
- "How do I run tests or create a migration?"

This skill is optimized for contributors working inside the Honcho repository.

## First, figure out what the user needs

Before changing files or running commands, determine which path fits the user:

1. **Fresh setup** — they just cloned the repo and need the full environment
2. **Run the app** — dependencies are installed, but they need the correct startup commands
3. **Run tests** — they only need the test workflow
4. **Fix an error** — they already tried setup and something is broken

Ask a short clarifying question if the user intent is ambiguous.

## The fastest happy path

If the user wants the standard local setup, use this sequence.

### 1. Install dependencies

From the repo root:

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
uv sync --group dev
```

Notes:

- Honcho requires Python `>=3.10`
- `uv` is the expected package manager and environment manager
- `uv sync` creates `.venv/` automatically

### 2. Create local config files

```bash
cp .env.template .env
cp docker-compose.yml.example docker-compose.yml
```

### 3. Start a local Postgres database

The easiest path is Docker:

```bash
docker compose up -d database
```

Alternative: use an existing Postgres or Supabase instance, but the database must support pgvector and the connection string must use the `postgresql+psycopg` prefix.

If you use the repo's Docker setup, the default credentials from `.env.template` and `docker-compose.yml.example` are:

```env
DB_CONNECTION_URI=postgresql+psycopg://testuser:testpwd@localhost:5432/honcho
```

If you use Supabase or another Postgres instance, make sure the target database has the vector extension available before running Honcho migrations.

### 4. Fill in required environment variables

At minimum, set:

```env
DB_CONNECTION_URI=postgresql+psycopg://...
LLM_ANTHROPIC_API_KEY=
LLM_GEMINI_API_KEY=
LLM_GROQ_API_KEY=
LLM_OPENAI_API_KEY=
AUTH_USE_AUTH=false
SENTRY_ENABLED=false
```

Important:

- You do **not** need every LLM key for every workflow, but you need the providers required by the code paths you are testing
- If reasoning, summaries, or embeddings are enabled, missing provider keys will break those features
- `AUTH_USE_AUTH=false` is the easiest local-development default

### 5. Run migrations

```bash
uv run alembic upgrade head
```

### 6. Start the API server

In terminal 1:

```bash
uv run fastapi dev src/main.py
```

### 7. Start the deriver

In terminal 2:

```bash
uv run python -m src.deriver
```

Why two terminals?

- The API server accepts writes and reads
- The deriver processes queued reasoning tasks in the background
- If the deriver is not running, messages may be stored but representations, summaries, and dreams will not progress

## Verification checklist

After setup, verify these in order.

### Check 1: imports and environment

```bash
uv run python -c "import src.main; print('ok')"
```

### Check 2: database connectivity

```bash
uv run alembic current
```

### Check 3: API server health

Start the API server, then in another shell:

```bash
curl http://127.0.0.1:8000/docs
```

### Check 4: background processing

Create data through the API or SDK, then confirm queue activity via logs from the deriver process.

## Common development tasks

### Run the main test suite

```bash
uv run pytest tests/
```

### Run a single test

```bash
uv run pytest tests/path/to/test_file.py::test_name -v
```

### Run only tests matching a keyword

```bash
uv run pytest tests/ -k search
```

### Lint and format Python

```bash
uv run ruff check src/ tests/ scripts/ migrations/
uv run ruff format src/ tests/ scripts/ migrations/
```

### Type-check Python

```bash
uv run basedpyright
```

### Run pre-commit on all files

```bash
uv run pre-commit run --all-files
```

### Install pre-commit hooks

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

### Create a new migration

After changing SQLAlchemy models:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

Then apply it:

```bash
uv run alembic upgrade head
```

## Important repository-specific gotchas

### TypeScript SDK tests

Do **not** run `bun test` directly for the TypeScript SDK.

Use the root pytest workflow instead:

```bash
uv run pytest tests/ -k typescript
```

Type-checking the TypeScript SDK directly is fine:

```bash
cd sdks/typescript && bun run typecheck
```

### The database URI prefix matters

This is a very common failure.

Bad:

```env
DB_CONNECTION_URI=postgresql://localhost/honcho
```

Good:

```env
DB_CONNECTION_URI=postgresql+psycopg://localhost/honcho
```

If the prefix is wrong, SQLAlchemy setup will fail in confusing ways.

### The deriver is required for memory formation

If someone says:

- "Messages are being created but no representation appears"
- "Summaries never show up"
- "Dreams are not running"

check whether `uv run python -m src.deriver` is running.

### Missing API keys can look like random runtime failures

If the server starts but reasoning features fail later, inspect:

- `.env`
- `config.toml` if present
- which providers are configured for dialectic, summary, deriver, and embeddings

### Docker only starts the database by default

`docker compose up -d database` brings up Postgres, not the API server or deriver. Those still need to be started manually with `uv run`.

## Troubleshooting playbook

### Problem: `uv sync` fails

Check:

1. Python version is at least 3.10
2. `uv` is installed and on `PATH`
3. You are running from the repo root

Helpful commands:

```bash
python3 --version
uv --version
pwd
```

### Problem: `alembic upgrade head` fails

Check:

1. `DB_CONNECTION_URI` exists in `.env`
2. the URI uses `postgresql+psycopg`
3. Postgres is actually running

Helpful commands:

```bash
docker compose ps
uv run alembic current
```

### Problem: API server starts, but requests fail

Check:

1. `.env` was created from `.env.template`
2. required provider keys exist for the feature being exercised
3. migrations were applied
4. the database is reachable from the app process

### Problem: representations or summaries never update

Check:

1. the deriver is running in a separate terminal
2. reasoning is enabled in configuration
3. provider keys required by the deriver are present

### Problem: tests fail in unrelated places

Use narrower test scopes first:

```bash
uv run pytest tests/path/to/test_file.py::test_name -v
uv run pytest tests/ -k keyword
```

Then expand outward once the local failure is understood.

## How the agent should help

When using this skill, prefer this order:

1. Read the current repo state (`.env`, `config.toml`, `docker-compose.yml`, recent errors)
2. Pick the smallest setup path that solves the user's problem
3. Run one verification command after each major step
4. Explain why a command is needed, not just what it does
5. If the user hits an error, quote the failing command and diagnose from the exact output

## Definition of done

A local Honcho setup is working when all of these are true:

- `uv sync --group dev` completed successfully
- `.env` is configured with a valid database URI
- `uv run alembic upgrade head` succeeded
- `uv run fastapi dev src/main.py` starts cleanly
- `uv run python -m src.deriver` starts cleanly
- the user can run at least one targeted test or reach the API docs locally
