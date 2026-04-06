---
name: local-development
description: Set up and run Honcho locally for development. Use when a contributor needs help with environment setup, running the API server or deriver, database configuration, running tests, pre-commit hooks, creating migrations, or troubleshooting common local development issues.
allowed-tools: Read, Glob, Grep, Bash(uv:*), Bash(docker:*), Bash(cp:*), Bash(alembic:*), Edit, Write
---

# Local Honcho Development

Guide for setting up, running, and developing on Honcho locally across macOS, Linux, Windows, and WSL2.

## Index

1. [Prerequisites](#prerequisites) — What you need before starting
2. [Environment Setup](#environment-setup) — Clone, install, configure, database
3. [Running Honcho](#running-honcho) — API server + deriver
4. [Development Tasks](#development-tasks) — Tests, linting, pre-commit hooks, migrations
5. [Contributing Conventions](#contributing-conventions) — Branches, commits, code style, PRs
6. [Platform-Specific Notes](#platform-specific-notes) — Windows, WSL2, macOS/Linux differences
7. [Troubleshooting](#troubleshooting) — Common issues and fixes
8. [Quick Reference](#quick-reference) — Command cheat sheet

---

## Prerequisites

- **Python**: 3.9+ required (tested through 3.13; newer versions like 3.14 may lack pre-built wheels for some dependencies)
- **uv**: 0.4.9+ (package manager). Install: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux/WSL2) or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows PowerShell)
- **Docker**: Required for the local Postgres + pgvector database (alternative: external Postgres via Supabase)
- **Git**: For cloning and branch management

---

## Environment Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
uv sync
```

This creates a virtual environment at `honcho/.venv` and installs all dependencies.

Activate the virtual environment:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Git Bash:**
```bash
source .venv/Scripts/activate
```

**WSL2:**
```bash
source .venv/bin/activate
```

> Note: `uv run` prefixes commands with the virtual environment automatically, so activation is optional when using `uv run`.

### 2. Configure Environment Variables

```bash
cp .env.template .env
```

Edit `.env` and set the required values. The method for setting environment variables differs by platform, but the `.env` file approach works everywhere since Honcho loads it via python-dotenv.

**Required variables in `.env`:**

```env
# Database connection (must use postgresql+psycopg prefix — this is a SQLAlchemy requirement)
DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@localhost:5432/postgres

# LLM provider keys — Anthropic and Gemini are the primary required keys
LLM_ANTHROPIC_API_KEY=your-key-here    # Used for dialectic (chat endpoint) and deriver
LLM_GEMINI_API_KEY=your-key-here       # Used for summary and deriver
LLM_GROQ_API_KEY=your-key-here         # Used for query generation
LLM_OPENAI_API_KEY=your-key-here       # Optional, for embeddings if EMBED_MESSAGES=true

# Recommended for local dev (both default to false, but setting explicitly avoids confusion)
AUTH_USE_AUTH=false
SENTRY_ENABLED=false
```

If you need to set environment variables directly in your shell (e.g., for one-off overrides):

**macOS / Linux / WSL2 / Git Bash:**
```bash
export DB_CONNECTION_URI="postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
```

**Windows PowerShell:**
```powershell
$env:DB_CONNECTION_URI="postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
```

### 3. Start the Database

**Option A: Docker (recommended)**

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d database redis
```

Both the API server and the deriver depend on Redis. Starting only `database` will cause Redis connection errors when you run the deriver.

Verify both are running:

```bash
docker compose ps
```

**WSL2 note:** If using Docker Desktop for Windows with WSL2 integration enabled, `docker compose` commands work directly inside your WSL2 terminal. Make sure "Use the WSL 2 based engine" is enabled in Docker Desktop settings.

**Option B: Supabase (no Docker required)**

Create a project at [supabase.com](https://supabase.com/) and use the connection string it provides. Update `DB_CONNECTION_URI` in your `.env` with the Supabase connection string, making sure to change the prefix to `postgresql+psycopg://` (Supabase gives you `postgresql://` by default).

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

This creates all tables: workspaces, peers, sessions, messages, and the queue system.

---

## Running Honcho

Honcho requires two processes running simultaneously: the API server and the background worker (deriver). Run these in separate terminals.

### Start the API Server

```bash
uv run fastapi dev src/main.py
```

This runs a development server with hot reload at `http://localhost:8000`. API docs are available at `http://localhost:8000/docs`.

### Start the Deriver (Background Worker)

In a separate terminal:

```bash
uv run python -m src.deriver
```

The deriver processes messages asynchronously to generate representations, summaries, peer cards, and dreaming tasks. You can run multiple deriver instances to improve throughput.

---

## Development Tasks

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/path/to/test_file.py

# Run a specific test function
uv run pytest tests/path/to/test_file.py::test_function
```

**TypeScript SDK tests**: Do NOT run `bun test` directly. The TS tests require a running Honcho server with database and Redis. Run them through pytest from the monorepo root:

```bash
uv run pytest tests/ -k typescript
```

TypeScript type-checking (safe to run standalone):

```bash
cd sdks/typescript && bun run tsc --noEmit
```

### Linting and Formatting

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run basedpyright
```

### Pre-commit Hooks

Install hooks (covers pre-commit, commit-msg, and pre-push stages):

```bash
uv add --dev pre-commit
uv run pre-commit install \
    --hook-type pre-commit \
    --hook-type commit-msg \
    --hook-type pre-push
```

Run hooks manually on all files:

```bash
uv run pre-commit run --all-files
```

Run a specific hook:

```bash
uv run pre-commit run ruff --all-files
uv run pre-commit run basedpyright --all-files
```

The hooks enforce: ruff linting/formatting, basedpyright type checking, bandit security scanning, markdown linting, license headers, trailing whitespace cleanup, and conventional commit messages.

### Creating New Migrations

When you change SQLAlchemy models in `src/models.py`:

```bash
uv run alembic revision --autogenerate -m "description of change"
uv run alembic upgrade head
```

Review the generated migration file before applying. Autogenerate does not catch every change (e.g., table renames, some constraint changes).

---

## Contributing Conventions

### Branch Naming

- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation updates
- `refactor/description` for code refactoring
- `test/description` for adding or updating tests

### Commit Messages

Follow conventional commit format:

```
type(scope): description
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:

```bash
git commit -m "feat(api): add new dialectic endpoint for user insights"
git commit -m "fix(db): resolve connection pool timeout issue"
git commit -m "docs(readme): update installation instructions"
```

### Code Style

- PEP 8 with Black-compatible 88-char line length
- Absolute imports preferred (isort conventions)
- Explicit type hints with SQLAlchemy `mapped_column` annotations
- `snake_case` for variables/functions, `PascalCase` for classes
- Google style docstrings
- Never hold a DB session open during external calls (LLM, embedding, HTTP). Compute the external result first, then pass it to the DB operation.
- Use `tracked_db` for short-lived DB-only operations

### Pull Requests

- Create PRs against `main`
- Include a clear description of changes and motivation
- Reference issues with "Closes #123" to auto-close them
- Ensure tests pass and linting is clean before requesting review

---

## Platform-Specific Notes

### macOS / Linux

This is the primary development platform for Honcho. All commands in the README work as documented.

On Linux, if `uv sync` fails building C extensions, install system dev packages:

```bash
# Debian/Ubuntu
sudo apt install python3-dev libpq-dev

# Fedora
sudo dnf install python3-devel libpq-devel
```

On macOS, if you hit compiler issues, make sure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

### Windows (PowerShell or Git Bash)

**psycopg on Windows:** Honcho uses `psycopg` (v3, not psycopg2). If `uv sync` fails with a "Microsoft Visual C++ 14.0 or greater is required" error, the C-optimized adapter could not compile. Fixes:

1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload. Then re-run `uv sync`.
2. Or force the binary or pure-Python fallback by setting an environment variable before install:
   - PowerShell: `$env:PSYCOPG_IMPL="binary"` then `uv sync`
   - Git Bash: `export PSYCOPG_IMPL="binary"` then `uv sync`

**Line endings:** Git on Windows defaults to `core.autocrlf=true`, which converts LF to CRLF on checkout. This can cause pre-commit hooks (trailing whitespace, end-of-file fixers) to flag every file. Configure git for this repo before your first commit:

```bash
git config core.autocrlf input
```

Or set it globally: `git config --global core.autocrlf input`.

**Path separators:** Use forward slashes (`/`) in commands even on Windows. Both PowerShell and Git Bash handle them. Avoid backslashes in arguments passed to `uv run` or Python.

**Docker Desktop:** Make sure Docker Desktop is running before using `docker compose` commands. Docker Desktop on Windows requires either the WSL2 backend or Hyper-V enabled.

**Python version compatibility:** If you are on a very new Python version (3.14+), some dependencies may not have pre-built wheels yet. If `uv sync` fails on specific packages, try pinning to a well-supported version:

```bash
uv python install 3.13
uv sync --python 3.13
```

### WSL2 (Windows Subsystem for Linux)

WSL2 is a strong option for Honcho development on Windows because it provides a native Linux environment where all commands work exactly as documented for macOS/Linux.

**Setup tips:**

- Install Docker Desktop for Windows and enable "WSL 2 based engine" plus integration with your WSL2 distro (Settings > Resources > WSL Integration).
- `docker compose` commands work directly in your WSL2 terminal once integration is enabled.
- Clone the repo inside the WSL2 filesystem (`~/honcho`), not on the Windows mount (`/mnt/c/...`). The Windows mount has significant I/O performance penalties that slow down installs, tests, and file watching.
- Your `.env` file works the same as on native Linux.
- If you edit files from both Windows (e.g., VS Code) and WSL2, use VS Code's "Remote - WSL" extension to avoid line ending and path issues.

**Common WSL2 issue:** If `docker compose up -d database redis` fails with connection errors, verify Docker Desktop's WSL2 integration is enabled for your specific distro. Run `docker info` inside WSL2 to check. If it errors, restart Docker Desktop.

---

## Troubleshooting

### Database Connection Issues

**"Connection refused" or "could not connect to server"**

1. Verify Docker is running: `docker compose ps`
2. Check the database container logs: `docker compose logs database`
3. Confirm your `DB_CONNECTION_URI` matches the Docker compose configuration
4. Make sure the URI uses `postgresql+psycopg` as the prefix (not `postgres://`, `postgresql://`, or `postgresql+psycopg2`)
5. On WSL2: confirm Docker Desktop WSL integration is enabled for your distro

**"relation does not exist" or missing tables**

Run migrations: `uv run alembic upgrade head`

**"password authentication failed"**

Check that the credentials in your `.env` match what is configured in `docker-compose.yml`.

### Missing API Keys

**"API key not found" or LLM provider errors**

Honcho uses different LLM providers for different subsystems:

- Anthropic: dialectic (chat endpoint) and deriver
- Gemini: summary and deriver
- Groq: query generation

Anthropic and Gemini are the primary required keys. Check which subsystem is failing from the error message and add the corresponding key to `.env`. Restart both the API server and the deriver after changing `.env`.

### Dependency Install Failures

**"Microsoft Visual C++ 14.0 or greater is required" (Windows)**

See the [Windows platform notes](#windows-powershell-or-git-bash) section for psycopg workarounds.

**"No matching distribution found" or missing wheels**

Your Python version may be too new for some dependencies. Try: `uv python install 3.13` then `uv sync --python 3.13`.

On Linux, install system dev packages: `sudo apt install python3-dev libpq-dev` (Debian/Ubuntu).

### Deriver Not Processing

**Messages are created but no representations appear**

1. Confirm the deriver is running in a separate terminal: `uv run python -m src.deriver`
2. Confirm Redis is running: `docker compose ps` — the deriver requires Redis
3. Check deriver logs for errors (connection issues, missing API keys)
4. Verify the database is accessible from the deriver process
5. Make sure you have the required LLM API key for the deriver (Anthropic or Gemini)

### Pre-commit Hook Failures

**Every file is flagged for whitespace/line ending changes**

This usually means Git is converting line endings. Fix with:

```bash
git config core.autocrlf input
git rm --cached -r .
git reset --hard
```

**Hook fails with "command not found"**

Make sure you run hooks with `uv run pre-commit run`, not just `pre-commit run`, so the virtual environment is active.

### Migration Errors

**"Target database is not up to date"**

```bash
uv run alembic upgrade head
```

**"Can't locate revision"**

Your local migrations may be out of sync. Pull latest from upstream and re-run:

```bash
git pull upstream main
uv run alembic upgrade head
```

### Port Already in Use

**"Address already in use" when starting the API server**

**macOS / Linux / WSL2:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Windows PowerShell:**
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Windows Git Bash:**
```bash
netstat -ano | grep :8000
# Note the PID, then:
taskkill //PID <PID> //F
```

### Docker Issues

**Containers won't start:**

```bash
docker compose down
docker compose up -d database redis
```

**Database needs a fresh start (deletes all data):**

```bash
docker compose down -v
docker compose up -d database redis
uv run alembic upgrade head
```

---

## Quick Reference

| Task | Command |
|---|---|
| Install dependencies | `uv sync` |
| Activate venv (macOS/Linux/WSL2) | `source .venv/bin/activate` |
| Activate venv (Windows PS) | `.\.venv\Scripts\Activate.ps1` |
| Activate venv (Windows Git Bash) | `source .venv/Scripts/activate` |
| Set up env | `cp .env.template .env` |
| Start database + Redis (Docker) | `docker compose up -d database redis` |
| Run migrations | `uv run alembic upgrade head` |
| Start API server | `uv run fastapi dev src/main.py` |
| Start deriver | `uv run python -m src.deriver` |
| Run all tests | `uv run pytest tests/` |
| Run TS SDK tests | `uv run pytest tests/ -k typescript` |
| Lint | `uv run ruff check src/` |
| Format | `uv run ruff format src/` |
| Type check | `uv run basedpyright` |
| Pre-commit (all files) | `uv run pre-commit run --all-files` |
| New migration | `uv run alembic revision --autogenerate -m "msg"` |
| Fix line endings (Windows) | `git config core.autocrlf input` |
