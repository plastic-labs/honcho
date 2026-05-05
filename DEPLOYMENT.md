# Self-Hosted Honcho + Athena (External Service Model)

## What This Guide Optimizes For

This guide assumes:
- you deploy Honcho separately from Athena
- Athena is then pointed to that running Honcho instance
- you prefer OpenRouter/OpenAI-compatible routing over direct OpenAI usage

This is the simplest operating model for a solo developer.

---

## 1) Deploy Honcho (Separate Service)

### Prerequisites

- Docker + Docker Compose
- One API key for a supported LLM route (recommended: `OPENROUTER_API_KEY`)

### Clone + branch

```bash
git clone https://github.com/TheophilusChinomona/honcho.git
cd honcho
git checkout stable
```

### Start infra

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d database redis
```

### Configure `.env` (OpenRouter-first)

```bash
# Database — use your ParadeDB host when running separately, or localhost if colocated
DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@<your-paradedb-host>:5432/postgres
# Schema isolation when sharing ParadeDB with other applications
DB_SCHEMA=honcho

# Keep auth off for private/internal deployments where Athena talks over trusted network
AUTH_USE_AUTH=false

# OpenRouter key (single source)
OPENROUTER_API_KEY=sk-or-v1-...

# Embeddings via OpenAI-compatible endpoint
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=text-embedding-3-small
EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY

# Deriver
DERIVER_MODEL_CONFIG__TRANSPORT=openai
DERIVER_MODEL_CONFIG__MODEL=openai/gpt-4.1-mini
DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
DERIVER_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY

# Dialectic (example level override)
DIALECTIC_LEVELS__low__MODEL_CONFIG__TRANSPORT=openai
DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL=openai/gpt-4.1-mini
DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY

# Summary
SUMMARY_MODEL_CONFIG__TRANSPORT=openai
SUMMARY_MODEL_CONFIG__MODEL=openai/gpt-4.1-mini
SUMMARY_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
SUMMARY_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY

# Dream
DREAM_DEDUCTION_MODEL_CONFIG__TRANSPORT=openai
DREAM_DEDUCTION_MODEL_CONFIG__MODEL=openai/gpt-4.1-mini
DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
DREAM_DEDUCTION_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY
DREAM_INDUCTION_MODEL_CONFIG__TRANSPORT=openai
DREAM_INDUCTION_MODEL_CONFIG__MODEL=openai/gpt-4.1-mini
DREAM_INDUCTION_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
DREAM_INDUCTION_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY

# Vector store + retrieval defaults
VECTOR_STORE_TYPE=pgvector
RETRIEVAL_HYBRID_ENABLED=true
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_TOP_K=20
RETRIEVAL_FULLTEXT_USE_WEBSEARCH=true
RETRIEVAL_MMR_ENABLED=false

# App
EMBED_MESSAGES=true
```

### Run migrations

```bash
uv run alembic upgrade head
```

### Start services

Terminal 1:
```bash
uv run fastapi dev src/main.py
```

Terminal 2:
```bash
uv run python -m src.deriver
```

### Verify Honcho

```bash
curl http://localhost:8000/health
```

---

## 2) Point Athena To Self-Hosted Honcho

### Manual path (deterministic)

```bash
hermes config set memory.provider honcho
hermes honcho setup
# choose: local
# base URL: http://<your-honcho-host>:8000
```

Then verify:

```bash
hermes memory status
hermes honcho status
```

### Agent bootstrap prompt path

Use this prompt in Athena after Honcho is up:

```text
Configure your memory provider to use my self-hosted Honcho instance.

Requirements:
1) Set memory.provider to honcho.
2) In the active HERMES_HOME profile, create or merge honcho.json (do not overwrite unrelated settings).
3) Set baseUrl to http://<your-honcho-host>:8000.
4) Ensure current host block has enabled=true and set:
   - peerName = <my-user-name>
   - aiPeer = <this-agent-name>
   - workspace = <my-workspace-id>
5) Preserve existing customized host blocks and non-Honcho providers unless they directly conflict.
6) Show exactly which files were changed and run:
   - hermes memory status
   - hermes honcho status
```

Replace placeholders before running.

---

## 3) Athena/Honcho Identity Mapping

Recommended multi-agent pattern:
- one shared user peer per human user
- one AI peer per Athena profile/agent
- shared workspace for related profiles

This preserves per-agent identity while sharing user understanding.

---

## 4) Notes On Providers

- Athena can use OpenCode Go/OpenRouter/etc for its own chat/delegation path.
- Honcho server-side model routing is configured independently.
- If OpenCode Go does not expose an OpenAI-compatible endpoint you want to target, use OpenRouter for Honcho.

---

## 5) Troubleshooting

- `password authentication failed for user postgres`
  - check `DB_CONNECTION_URI` credentials and host
- Athena shows built-in memory only
  - ensure `memory.provider` is `honcho`
- Athena not connecting to Honcho
  - verify `baseUrl` in `honcho.json` and `curl <baseUrl>/health`
- no observations being derived
  - ensure deriver process is running

---

## 6) Later CI/CD (not this step)

After the external-service workflow is stable:
- build/publish Athena + Honcho images from GitHub
- deploy by SSH-triggered host update (`deploy.sh update` style)
- keep Athena and Honcho as separate services for cleaner ops and rollback
