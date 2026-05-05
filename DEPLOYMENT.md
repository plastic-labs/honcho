# Self-Hosted Honcho + Athena Deployment Guide

## Fork Status

Your fork already exists at:
**https://github.com/TheophilusChinomona/honcho**

The `feat/hybrid-retrieval` branch contains the hybrid search enhancements.

---

## Quick Start: Self-Host Honcho

### Prerequisites

- Docker + Docker Compose
- OpenAI API key (or other supported LLM provider)

### 1. Clone Your Fork

```bash
git clone https://github.com/TheophilusChinomona/honcho.git
cd honcho
git checkout feat/hybrid-retrieval
```

### 2. Start Postgres + pgvector

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d db
```

This boots Postgres with pgvector on port `5432`.

### 3. Configure Environment

Create a `.env` file:

```bash
# Database
DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@localhost:5432/postgres

# LLM (required for deriver, dialectic, dreamer)
LLM_OPENAI_API_KEY=sk-...

# Embeddings
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=text-embedding-3-small

# Optional: Hybrid retrieval tuning
RETRIEVAL_HYBRID_ENABLED=true
RETRIEVAL_RRF_K=60
RETRIEVAL_MMR_ENABLED=false
RETRIEVAL_MMR_LAMBDA=0.5
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_TOP_K=20
RETRIEVAL_FULLTEXT_USE_WEBSEARCH=true

# App
APP_EMBED_MESSAGES=true
```

### 4. Run Migrations

```bash
uv run alembic upgrade head
```

### 5. Start Services

Terminal 1 — API server:
```bash
uv run fastapi dev src/main.py
```

Terminal 2 — Background deriver worker:
```bash
uv run python -m src.deriver
```

The API is now available at `http://localhost:8000`.

---

## Hybrid Retrieval Configuration

All retrieval knobs are environment variables prefixed with `RETRIEVAL_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_HYBRID_ENABLED` | `true` | Enable semantic + full-text fusion for documents |
| `RETRIEVAL_RRF_K` | `60` | RRF constant (lower = top ranks dominate more) |
| `RETRIEVAL_SCORE_THRESHOLD` | *(none)* | Minimum RRF score to keep a result |
| `RETRIEVAL_MMR_ENABLED` | `false` | Enable Maximal Marginal Relevance diversity |
| `RETRIEVAL_MMR_LAMBDA` | `0.5` | Relevance/diversity trade-off (0=diversity, 1=relevance) |
| `RETRIEVAL_RERANK_ENABLED` | `false` | Enable lexical reranking after fusion |
| `RETRIEVAL_RERANK_TOP_K` | `20` | How many top results to rerank |
| `RETRIEVAL_EXACT_MATCH_BOOST` | `2.0` | Multiplier for exact substring matches |
| `RETRIEVAL_FULLTEXT_USE_WEBSEARCH` | `true` | Use `websearch_to_tsquery` for quoted phrases |

### Recommended Starting Config

For Athena agents where **exact-name recall** and **long-history retrieval** matter:

```bash
RETRIEVAL_HYBRID_ENABLED=true
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_TOP_K=20
RETRIEVAL_FULLTEXT_USE_WEBSEARCH=true
```

If you see redundant observations in dialectic results, enable MMR:

```bash
RETRIEVAL_MMR_ENABLED=true
RETRIEVAL_MMR_LAMBDA=0.7
```

---

## Connecting Athena (Hermes) Agents

### 1. Athena Memory Provider Setup

In your Athena repo, the Honcho plugin lives at:
```
athena/plugins/memory/honcho/
```

Configure Athena to use your self-hosted Honcho instance:

```yaml
# athena/config.yaml or env equivalent
memory:
  provider: honcho
  honcho:
    base_url: http://localhost:8000   # or your deployed URL
    api_key: ""                        # if AUTH_USE_AUTH=true, set JWT here
```

### 2. Peer / Session Mapping

The Athena Honcho plugin maps Athena concepts to Honcho peers:

| Athena Concept | Honcho Concept |
|----------------|----------------|
| User | Peer (human peer) |
| Agent / Profile | Peer (AI peer) |
| Chat / Thread | Session |
| Workspace | Workspace |

Recommended Athena config shape:

```yaml
memory:
  provider: honcho
  honcho:
    base_url: http://localhost:8000
    recallMode: hybrid
    contextCadence: 1
    dialecticCadence: 2
    dialecticDepth: 1
    sessionStrategy: per-session
```

### 3. Multi-Agent Setup

If Athena runs multiple agent profiles for the same user:

- **One shared user peer** per real user
- **One AI peer per Athena profile/agent**
- **Shared workspace** for related profiles
- Observation defaults should preserve tenant boundaries

This gives each agent its own identity while sharing a unified user model.

### 4. Environment Variables for Athena

If Athena reads Honcho config via env:

```bash
HONCHO_BASE_URL=http://localhost:8000
HONCHO_API_KEY=                      # set if Honcho auth is enabled
```

---

## Deployment Options

### Option A: Local Development (Docker Compose)

Use the included `docker-compose.yml.example` for local testing.

```bash
docker compose up -d db
uv run alembic upgrade head
uv run fastapi dev src/main.py
```

### Option B: Production Deploy

For production, deploy:

1. **Postgres + pgvector** (managed or self-hosted)
2. **Honcho API** as a container/service
3. **Deriver worker** as a separate container/service
4. **Redis** (optional, for caching/queues)

Example `docker-compose.prod.yml` skeleton:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  honcho-api:
    build: .
    command: uv run fastapi run src/main.py --host 0.0.0.0 --port 8000
    environment:
      DB_CONNECTION_URI: postgresql+psycopg://postgres:${DB_PASSWORD}@db:5432/postgres
      LLM_OPENAI_API_KEY: ${OPENAI_API_KEY}
      RETRIEVAL_HYBRID_ENABLED: "true"
    ports:
      - "8000:8000"
    depends_on:
      - db

  deriver:
    build: .
    command: uv run python -m src.deriver
    environment:
      DB_CONNECTION_URI: postgresql+psycopg://postgres:${DB_PASSWORD}@db:5432/postgres
      LLM_OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - db

volumes:
  pgdata:
```

### Option C: Kubernetes / Cloud

- Use the same container image for both API and deriver
- Set `DERIVER_ENABLED=true` on deriver pods, `DERIVER_ENABLED=false` on API pods if you want to split them
- Use a managed Postgres with pgvector extension (e.g., Supabase, AWS RDS with pgvector)

---

## Verification Checklist

After deploying, verify:

1. **API health**: `curl http://localhost:8000/health`
2. **Hybrid retrieval active**: Query documents via API and confirm both semantic and FTS results appear
3. **Athena connection**: Send a message through Athena and check Honcho's `messages` table
4. **Deriver running**: Check that observations are created in the `documents` table after messages are ingested
5. **Dialectic working**: Use Athena's dialectic/memory recall and verify context is injected

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `password authentication failed for user "postgres"` | Check `DB_CONNECTION_URI` matches your Postgres credentials |
| No observations created after messages | Ensure deriver worker is running and `DERIVER_ENABLED=true` |
| Athena sees no memory context | Verify `EMBED_MESSAGES=true` and embedding API key is set |
| Hybrid search too slow | Disable MMR (`RETRIEVAL_MMR_ENABLED=false`) or reduce `RERANK_TOP_K` |
| Exact phrases not matching | Ensure `RETRIEVAL_FULLTEXT_USE_WEBSEARCH=true` |
| High latency on dialectic | Reduce `DIALECTIC_MAX_TOOL_ITERATIONS` or `DERIVER_WORKING_REPRESENTATION_MAX_OBSERVATIONS` |

---

## Next Steps

1. **Benchmark** the hybrid retrieval against your existing Athena-memory workloads
2. **Tune** `RRF_K` and `MMR_LAMBDA` based on real query patterns
3. **Monitor** retrieval quality via Athena's memory-context scrubber output
4. **Iterate** — if gaps remain, the next port candidates from the plan are:
   - pgvector/ParadeDB-first self-host defaults
   - Retrieval stats / traceability
   - Stricter scope filters for Athena multi-tenancy
