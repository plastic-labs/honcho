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

- [uv](https://docs.astral.sh/uv/) — Python package manager (handles Python version automatically)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or: brew install uv
  ```
- **Git**
- **Docker** with Compose v2 (for Docker setup) — use `docker compose`, not `docker-compose`

### Install Dependencies

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
uv sync
```

Activate the virtual environment (optional — `uv run` handles this automatically):

```bash
source .venv/bin/activate        # Unix/macOS
# .venv\Scripts\activate         # Windows CMD
# .venv\Scripts\activate.ps1     # Windows PowerShell
```

### Configure Environment Variables

```bash
cp .env.template .env
```

#### LLM Setup (required — server will not start without this)

Honcho uses LLMs for memory extraction, summarization, dialectic chat, and dreaming. Any OpenAI-compatible endpoint works — OpenRouter, Together, Fireworks, Ollama, vLLM, or a direct vendor API. Models must support tool calling.

**Quick start (recommended):**

1. Get a key from [openrouter.ai](https://openrouter.ai) (or any OpenAI-compatible provider)
2. Edit your `.env`:

```bash
LLM_OPENAI_COMPATIBLE_BASE_URL=https://openrouter.ai/api/v1
LLM_OPENAI_COMPATIBLE_API_KEY=your-api-key-here
LLM_EMBEDDING_PROVIDER=openrouter

DERIVER_PROVIDER=custom
DERIVER_MODEL=google/gemini-2.5-flash

SUMMARY_PROVIDER=custom
SUMMARY_MODEL=google/gemini-2.5-flash

DREAM_PROVIDER=custom
DREAM_MODEL=google/gemini-2.5-flash
DREAM_DEDUCTION_MODEL=google/gemini-2.5-flash
DREAM_INDUCTION_MODEL=google/gemini-2.5-flash

DIALECTIC_LEVELS__minimal__PROVIDER=custom
DIALECTIC_LEVELS__minimal__MODEL=google/gemini-2.5-flash
DIALECTIC_LEVELS__low__PROVIDER=custom
DIALECTIC_LEVELS__low__MODEL=google/gemini-2.5-flash
DIALECTIC_LEVELS__medium__PROVIDER=custom
DIALECTIC_LEVELS__medium__MODEL=google/gemini-2.5-flash
DIALECTIC_LEVELS__high__PROVIDER=custom
DIALECTIC_LEVELS__high__MODEL=google/gemini-2.5-flash
DIALECTIC_LEVELS__max__PROVIDER=custom
DIALECTIC_LEVELS__max__MODEL=google/gemini-2.5-flash
```

> **Tip:** Use find-and-replace to swap all `your-model-here` with your chosen model in one step.

**Alternative — direct vendor keys:**
```bash
# Uncomment only the key you want to use
# LLM_GEMINI_API_KEY=your-key
# LLM_ANTHROPIC_API_KEY=your-key
# LLM_OPENAI_API_KEY=your-key
# LLM_GROQ_API_KEY=your-key
```
Then set `DERIVER_PROVIDER=google` (or `anthropic`, `openai`, `groq`) and use the corresponding model name format.

#### Key environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_CONNECTION_URI` | `postgresql+psycopg://postgres:postgres@localhost:5432/postgres` | Must use `postgresql+psycopg` prefix |
| `LLM_OPENAI_COMPATIBLE_BASE_URL` | — | Base URL for OpenAI-compatible endpoint |
| `LLM_OPENAI_COMPATIBLE_API_KEY` | — | API key for your endpoint |
| `LLM_EMBEDDING_PROVIDER` | `openrouter` | Routes embeddings through your custom endpoint |
| `DERIVER_PROVIDER` | `custom` | LLM provider for the background worker |
| `DERIVER_MODEL` | — | Model name for the deriver |
| `AUTH_USE_AUTH` | `false` | Set to `true` in production; requires `AUTH_JWT_SECRET` |

---

## Phase 2: Running Honcho

### Option A: Docker (Recommended)

Docker Compose handles the database, Redis cache, API server, and deriver worker.

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d --build
```

The first build takes a few minutes. This starts four services:
- **api** — HTTP server on `localhost:8000`
- **deriver** — background worker (no HTTP port)
- **database** — PostgreSQL with pgvector on `localhost:5432`
- **redis** — cache on `localhost:6379`

Migrations run automatically on startup.

Verify everything is running:
```bash
docker compose ps
curl http://localhost:8000/health
# {"status":"ok"}
```

### Option B: Manual Setup

**1. Start PostgreSQL and Redis:**
```bash
docker run --name honcho-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d pgvector/pgvector:pg15

docker run --name honcho-redis \
  -p 6379:6379 \
  -d redis:alpine
```

Or use [Supabase](https://supabase.com) (free, enable pgvector in SQL editor: `CREATE EXTENSION IF NOT EXISTS vector;`).

**2. Run migrations:**
```bash
uv run alembic upgrade head
```

**3. Start the API server:**
```bash
uv run fastapi dev src/main.py
```

**4. Start the deriver** (separate terminal — required for memory processing):
```bash
uv run python -m src.deriver
```

---

## Phase 3: Verifying Your Setup

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok"}

# Smoke test — confirms database + migrations
curl -s -X POST http://localhost:8000/v3/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}' | python3 -m json.tool
# Should return a workspace object with an id
```

Interactive API docs: `http://localhost:8000/docs`

---

## Phase 4: Development Tasks

### Running Tests

```bash
uv run pytest
```

Run a specific test file:
```bash
uv run pytest tests/test_users.py -v
```

### Creating a New Migration

```bash
uv run alembic revision --autogenerate -m "describe your change"
```

Review the generated file in `migrations/versions/` before applying:
```bash
uv run alembic upgrade head
```

Check current migration status:
```bash
uv run alembic current
```

### Setting Up Pre-commit Hooks

```bash
uv run pre-commit install
```

Hooks run automatically on `git commit`. To run manually:
```bash
uv run pre-commit run --all-files
```

### Code Standards

- **Formatter:** Black (`uv run black .`)
- **Types:** Type hints required on all new functions
- **Docstrings:** Google style
- **Commits:** Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`)

---

## Troubleshooting

### Server fails to start — "LLM provider not configured"

The server requires at least one LLM provider. Set `LLM_OPENAI_COMPATIBLE_BASE_URL` and `LLM_OPENAI_COMPATIBLE_API_KEY`, then set `DERIVER_MODEL`, `SUMMARY_MODEL`, `DREAM_MODEL`, and all `DIALECTIC_LEVELS__*__MODEL` values. See [LLM Setup](#llm-setup) above.

### Database connection error

```
sqlalchemy.exc.OperationalError: connection refused
```

- Confirm PostgreSQL is running: `docker compose ps` or `brew services list | grep postgresql`
- Confirm `DB_CONNECTION_URI` uses the `postgresql+psycopg` prefix (not `postgresql://`)
- Default DB name is `postgres` (not `honcho`)

### Migrations not applied

```
sqlalchemy.exc.ProgrammingError: relation "..." does not exist
```

Run: `uv run alembic upgrade head`

### Deriver not processing messages

Check logs:
```bash
docker compose logs deriver --tail 20
```

- Confirm `DERIVER_PROVIDER` and `DERIVER_MODEL` are set in `.env`
- Without the deriver, messages are stored but no memory extraction occurs
- Look for "polling" or "processing" in the logs to confirm it's running

### Port already in use

```bash
lsof -i :8000   # find what's using port 8000
kill -9 <PID>
```

### Docker build fails

```bash
docker compose down --volumes
docker compose up -d --build
```

If BuildKit errors occur, ensure Docker Desktop is up to date.

---

## Quick Reference

| Task | Command |
|------|---------|
| Start all services (Docker) | `docker compose up -d --build` |
| Stop all services | `docker compose down` |
| Start API server (manual) | `uv run fastapi dev src/main.py` |
| Start deriver (manual) | `uv run python -m src.deriver` |
| Run migrations | `uv run alembic upgrade head` |
| New migration | `uv run alembic revision --autogenerate -m "description"` |
| Run tests | `uv run pytest` |
| Check health | `curl http://localhost:8000/health` |
| View API docs | `http://localhost:8000/docs` |
| View logs (Docker) | `docker compose logs api --tail 50` |
| Check container status | `docker compose ps` |

---

## Resources

- [Honcho Documentation](https://docs.honcho.dev)
- [Configuration Guide](https://docs.honcho.dev/contributing/configuration)
- [API Reference](https://docs.honcho.dev/api-reference/introduction)
- [Discord Community](https://discord.gg/honcho)
