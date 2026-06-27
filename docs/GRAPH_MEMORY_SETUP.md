# Graph Memory + Promotion Worker — Setup & Operations Guide

**Status:** Operational (2026-06-27)
**Branch:** `local/ngram-graph-memory`
**Workspace:** `agentc`

---

## What This Is

Graph memory extends Honcho's vector-store memory with a **semantic network layer**:

1. **Promotion worker** — background process that evaluates each observation
   (conclusion) for durability and promotes worthy ones to L2 (graph memory).
   For each promoted observation, it finds semantically similar neighbors via
   pgvector cosine similarity and creates typed **edges** between them.

2. **Graph recall** — spreading-activation traversal across edges, starting
   from vector-search anchors. Returns ranked observations with activation
   scores and confidence decay.

3. **Context management** — named contexts for workstream isolation, with
   Redis-backed active-context state and thread-to-context bindings.

4. **Compaction scheduler** — prunes access-log events older than 5 half-lives
   (~5 days) every 24 hours.

---

## Prerequisites

- Docker + Docker Compose
- Ollama running on the host with these models loaded:
  - `nomic-embed-text:latest` (768-dim embeddings)
  - `qwen2.5:7b-instruct-ctx16k` (deriver + promotion LLM)
  - `qwen3.5` (summary, dialectic)
- pgvector extension (provided by `pgvector/pgvector:pg15` image)

---

## Installation (From Scratch)

### 1. Clone and checkout

```bash
cd /home/claw/honcho-selfhost
git checkout local/ngram-graph-memory
```

### 2. Configure `.env`

The `.env` file provides all model routing and feature flags. Key settings for
graph memory:

```bash
# Promotion — uses heuristic test (no cloud LLM needed)
PROMOTION_ENABLED=false          # LLM promotion test off (uses heuristic)
# PROMOTION_PROCESSING_ENABLED defaults to True in config — do NOT set to false

# Embeddings — 768-dim nomic-embed-text via local Ollama
EMBEDDING_VECTOR_DIMENSIONS=768
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=nomic-embed-text:latest
EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://host.docker.internal:11434/v1
EMBEDDING_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=LLM_OPENAI_API_KEY

# LLM key (dummy — Ollama ignores it but the OpenAI client requires one)
LLM_OPENAI_API_KEY=ollama

# Deriver
DERIVER_ENABLED=true
DERIVER_WORKERS=1

# Vector store
VECTOR_STORE_TYPE=postgres
VECTOR_STORE_MIGRATED=true
```

**Critical:** `EMBEDDING_VECTOR_DIMENSIONS=768` must match the model actually
loaded in Ollama. If you rebuild the Docker image without this env var, it
defaults to 1536 dimensions and the deriver will crash with a dimension
mismatch against existing vectors in the database.

### 3. Run database migrations

The graph memory tables (`edges`, `access_log`, `context_index`,
`thread_binding_registry`) and the `promotion_failed`, `promotion_attempts`,
`promotion_error`, `promoted_at` columns on `documents` are created by
Alembic migrations.

```bash
# Start database first
docker compose up -d database redis

# Run migrations through the API container
docker compose run --rm api sh -c 'cd /app && .venv/bin/alembic upgrade head'
```

### 4. Build and start all services

```bash
docker compose build
docker compose up -d
```

### 5. Verify the deriver is processing

```bash
docker logs honcho-selfhost-deriver-1 2>&1 | tail -20
```

You should see:
- `Starting promotion scheduler (interval: 60s)`
- `N observations await graph promotion`
- `Processing promotion for observation ...`
- `Created N edges for observation ...`

If you see `ValueError: Invalid task type in work_unit_key: promotion`,
the `work_unit.py` fix is missing — see **Bug Fixes** below.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Docker Compose Network                  │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
│  │ Database  │   │  Redis   │   │    API       │    │
│  │ (pgvector)│   │ (cache)  │   │ :8088→:8000  │    │
│  └──────────┘   └──────────┘   └──────────────┘    │
│       ▲                              │               │
│       │                              │               │
│  ┌────┴───────────────────────────────┘              │
│  │                                                   │
│  │  ┌──────────────────────────────────────────┐    │
│  │  │             Deriver Worker                │    │
│  │  │                                            │    │
│  │  │  • Queue consumer (representation,        │    │
│  │  │    summary, promotion, webhook, dream)    │    │
│  │  │  • Promotion scheduler (60s interval)     │    │
│  │  │  • Compaction scheduler (24h interval)    │    │
│  │  │  • Reconciler scheduler                   │    │
│  │  │  • Prometheus metrics :9090               │    │
│  │  └──────────────────────────────────────────┘    │
│  └───────────────────────────────────────────────────│
│                                                      │
└─────────────────────────────────────────────────────┘
           │
           │ host.docker.internal:11434
           ▼
    ┌──────────────┐
    │    Ollama     │
    │  (host GPU)   │
    └──────────────┘
```

### Data Flow

```
Messages → API → Queue → Deriver worker
                              │
                    ┌─────────┴──────────┐
                    │                    │
              Representation       Promotion Scheduler
              (extract obs)         (every 60s)
                    │                    │
                    ▼                    ▼
              Documents table      Enqueue promotion tasks
              (+ embeddings)            │
                                        ▼
                                  Promotion worker
                                  (process_promotion)
                                        │
                           ┌────────────┴────────────┐
                           │                         │
                    Heuristic/LLM           Vector similarity
                    promotion test           (cosine dist ≤ 0.3)
                           │                         │
                           ▼                         ▼
                    Promoted to L2          Create edges to
                    (access_log)            related observations
                                                   │
                                                   ▼
                                            edges table
                                                   │
                                                   ▼
                                        Graph recall
                                   (spreading activation CTE)
```

---

## Configuration Reference

### Promotion Settings (`config.toml` `[promotion]` or env vars)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `enabled` | `PROMOTION_ENABLED` | `true` | If `false`, uses heuristic test instead of LLM |
| `processing_enabled` | `PROMOTION_PROCESSING_ENABLED` | `true` | Master switch — if `false`, scheduler scans but doesn't enqueue |
| `model_config` | `PROMOTION_MODEL_CONFIG__*` | qwen2.5:7b | LLM for promotion test (only used if `enabled=true`) |
| `max_tokens` | `PROMOTION_MAX_TOKENS` | 8 | Max output tokens for LLM promotion test |
| `max_input_tokens` | `PROMOTION_MAX_INPUT_TOKENS` | 2000 | Max input tokens for LLM promotion test |
| `max_outer_retries` | `PROMOTION_MAX_OUTER_RETRIES` | 3 | Retries for LLM promotion test |

### Embedding Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `vector_dimensions` | `EMBEDDING_VECTOR_DIMENSIONS` | 1536 | **Must match Ollama model** — nomic-embed-text = 768 |
| `model_config.model` | `EMBEDDING_MODEL_CONFIG__MODEL` | - | Ollama model name |
| `model_config.transport` | `EMBEDDING_MODEL_CONFIG__TRANSPORT` | - | `openai` for Ollama |
| `max_input_tokens` | `EMBEDDING_MAX_INPUT_TOKENS` | 2048 | Max tokens per embedding request |

### Promotion Worker Parameters (in `src/deriver/promotion.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_PROMOTION_EDGE_COSINE_DISTANCE` | 0.3 | Two observations must be closer than this (cosine similarity > 0.7) to get an edge |
| `MAX_PROMOTION_ATTEMPTS` | 3 | After this many failures, observation is marked `promotion_failed=True` |
| `MAX_TOKENS_PER_OBSERVATION_EMBEDDING` | 90% of model max | Chunk threshold for oversized observations |
| `_get_related_observation_ids limit` | 20 | Max edges per promotion |

---

## Operational Procedures

### Rebuild the Deriver After Code Changes

The deriver Docker image has code **baked in** (no volume mount for `src/`).
After any code change to the deriver source:

```bash
cd /home/claw/honcho-selfhost

# Rebuild the image
docker compose build deriver

# Remove old container and start fresh
docker rm -f honcho-selfhost-deriver-1
docker compose up -d --no-deps deriver

# Verify it's running
docker logs honcho-selfhost-deriver-1 2>&1 | tail -20
```

**Do NOT use `docker compose up` without first removing the old container** —
you'll get a container name conflict.

### Rebuild the API After Code Changes

Same pattern:

```bash
docker compose build api
docker rm -f honcho-selfhost-api-1
docker compose up -d --no-deps api
```

### Check Graph Health

```bash
# Edge count
docker exec honcho-selfhost-deriver-1 python3 -c "
import asyncio
from src.dependencies import tracked_db
from sqlalchemy import text
async def check():
    async with tracked_db('check') as db:
        r = await db.execute(text('SELECT count(*) FROM edges'))
        print(f'Edges: {r.scalar()}')
        r2 = await db.execute(text(\"SELECT count(*) FROM queue WHERE task_type='promotion' AND processed=False\"))
        print(f'Pending promotions: {r2.scalar()}')
asyncio.run(check())
"
```

### Clear Stuck Queue Sessions

If the deriver crashes and leaves orphaned active queue sessions:

```bash
docker exec honcho-selfhost-deriver-1 python3 -c "
import asyncio
from src.dependencies import tracked_db
from sqlalchemy import text
async def cleanup():
    async with tracked_db('cleanup') as db:
        await db.execute(text('DELETE FROM active_queue_sessions'))
        await db.commit()
        print('Cleared all active_queue_sessions')
asyncio.run(cleanup())
"
```

### Test Recall Quality

```bash
# Direct API test
curl -s http://localhost:8088/v3/workspaces/agentc/conclusions/query \
  -X POST -H 'Content-Type: application/json' \
  -d '{"query": "development workflow", "top_k": 5, "filters": {"observer": "andrew", "observed": "andrew"}}'

# Graph recall (via Hermes gateway — uses spreading activation)
# Use the honcho_recall tool from the Hermes agent
```

---

## Bug Fixes Applied (2026-06-27)

Three bugs prevented graph edge creation. All are fixed in commit `435f619`
on branch `local/ngram-graph-memory`.

### Bug 1: `parse_work_unit_key` did not recognize `promotion` task type

**File:** `src/utils/work_unit.py`

**Symptom:** Deriver crashes with `ValueError: Invalid task type in work_unit_key: promotion` whenever it claims a promotion work unit from the queue. Docker auto-restarts the container, but it crashes again on the next promotion item. Representation tasks work fine because they're processed first.

**Root cause:** The promotion scheduler creates queue items with `work_unit_key` format `promotion:{workspace}:{observed}:{obs_id}`, but `parse_work_unit_key()` and `construct_work_unit_key()` had no handler for the `promotion` task type.

**Fix:** Added `promotion` support to both functions. Key format: `promotion:{workspace_name}:{observed}:{obs_id}` (4 colon-separated parts).

### Bug 2: `create_edge` could not adapt Python dict to JSONB

**File:** `src/crud/graph_memory.py`

**Symptom:** Promotion worker logs "Created 0 edges" for every observation, even when vector similarity finds 20 related neighbors. The error is caught at debug level and silently swallowed.

**Root cause:** `create_edge()` passes `edge_metadata or {}` (a Python dict) as the `:metadata` parameter to raw SQL `text()`. psycopg cannot adapt Python dicts to PostgreSQL JSONB. Additionally, `:metadata::jsonb` syntax conflicts with SQLAlchemy's `:param` naming — the `::jsonb` cast gets parsed as part of the parameter name.

**Fix:**
- Serialize metadata with `json.dumps()` before passing
- Use `CAST(:metadata AS jsonb)` instead of `:metadata::jsonb`
- Use `CAST(:created_by AS text)` instead of `:created_by::text`

### Bug 3: Duplicate queue items from promotion scheduler

**Symptom:** The promotion scheduler enqueues the same observations every 60s scan cycle, creating hundreds of duplicate queue items. The deriver processes them all (creating duplicate edges that get upserted), wasting cycles.

**Root cause:** The scheduler's `_scan_and_enqueue()` queries for observations without a `promote` event in `access_log`. If the deriver is slow or crashed, the same observations appear in every scan.

**Mitigation:** The `ON CONFLICT` upsert on edges prevents duplicate edge rows, and the `access_log` promote event prevents re-promotion after success. But the queue items themselves accumulate. This is a known issue — a deduplication guard or `work_unit_key` uniqueness constraint on the queue would fix it.

---

## File Manifest

### Graph Memory Core

| File | Purpose |
|------|---------|
| `src/utils/types.py` | `EdgeType`, `AccessLogEventType` type literals |
| `src/models.py` | `Edge`, `AccessLogEntry`, `ContextIndex`, `ThreadBinding` SQLAlchemy models |
| `src/schemas/graph_memory.py` | Pydantic request/response schemas |
| `src/crud/graph_memory.py` | CRUD: edges, contexts, thread bindings, pinning, verify, recall CTE |
| `src/routers/graph_memory.py` | FastAPI router (18 endpoints) |
| `src/routers/GRAPH_MEMORY_README.md` | API reference for graph memory endpoints |

### Promotion Worker

| File | Purpose |
|------|---------|
| `src/deriver/promotion.py` | `process_promotion()`, heuristic/LLM test, vector similarity, edge creation |
| `src/deriver/promotion_scheduler.py` | Scans for un-promoted observations every 60s, enqueues tasks |
| `src/deriver/compaction_scheduler.py` | Compacts access log every 24h (GC protocol) |

### Queue Infrastructure

| File | Purpose |
|------|---------|
| `src/utils/work_unit.py` | Work unit key construction and parsing (all task types) |
| `src/utils/queue_payload.py` | Pydantic payloads for each task type (`PromotionPayload`, etc.) |
| `src/deriver/queue_manager.py` | Queue polling, claiming, batch processing |
| `src/deriver/consumer.py` | Task dispatch — routes queue items to handlers |

### Hermes Agent Integration

| File | Purpose |
|------|---------|
| `~/.hermes/hermes-agent/plugins/memory/honcho/__init__.py` | `honcho_recall`, `honcho_recall_context`, `honcho_thread_bind` tools |

### Migrations

| File | Purpose |
|------|---------|
| `migrations/versions/2a3b4c5d6e7f_add_graph_memory_tables.py` | Creates `edges`, `access_log`, `context_index`, `thread_binding_registry` tables |
| (later migration) | Adds `promotion_failed`, `promotion_attempts`, `promotion_error`, `promoted_at` columns to `documents` |

### Tests

| File | Purpose |
|------|---------|
| `tests/unit/validate_phase1.py` | Schema + CRUD logic validation (26 tests) |
| `tests/unit/verify_migration.py` | Migration verification (tables, indexes, FKs, rollback) |

---

## Troubleshooting

### Deriver crashes with `ValueError: Invalid task type in work_unit_key: promotion`

The `work_unit.py` fix is missing. Apply commit `435f619` or manually add
`promotion` support to `construct_work_unit_key()` and `parse_work_unit_key()`.

### Deriver crashes with embedding dimension mismatch

`EMBEDDING_VECTOR_DIMENSIONS` env var doesn't match the Ollama model.
Check: `docker exec honcho-selfhost-deriver-1 python3 -c "from src.config import settings; print(settings.EMBEDDING_MODEL.VECTOR_DIMENSIONS)"`
and compare with the model loaded in Ollama.

### Promotion worker logs "Created 0 edges" for all observations

The `create_edge` JSONB adaptation fix is missing. Apply commit `435f619`
or fix `src/crud/graph_memory.py` to use `json.dumps()` + `CAST(:metadata AS jsonb)`.

### No promotion tasks in queue

Check that `PROMOTION_PROCESSING_ENABLED` is `True` (it defaults to `True`
in config). If `PROMOTION_ENABLED` is `False`, that's OK — it just means the
heuristic test is used instead of the LLM test. Promotion still runs.

### Recall returns results with `confidence: 0.0`

Confidence is derived from verification events in the access log. If no
observations have been verified (no `verify` events), confidence is 0.0
for all results. This is expected behavior — confidence decays from the
last verification event. Activation scores will still be non-zero if
promotion or recall events exist.

### Container name conflict on restart

```bash
docker rm -f honcho-selfhost-deriver-1
docker compose up -d --no-deps deriver
```

---

## Hermes Agent Integration

The Hermes agent (Aime) accesses graph memory through three tools in
`plugins/memory/honcho/__init__.py`:

- **`honcho_recall`** — Spreading-activation recall. Returns observations
  ranked by activation × confidence, traversing graph edges from vector-search
  anchors.
- **`honcho_recall_context`** — Manage named recall contexts (create, switch,
  activate, evict, list members).
- **`honcho_thread_bind`** — Bind Slack thread IDs to named contexts for
  automatic context routing.

These tools were merged in PR #4 on `hermes-tmw/hermes-agent` (squash-merged
2026-06-27). The gateway must be restarted after merging to pick up the new
tools.

### Restarting the Gateway

```bash
# Find the gateway process
ps aux | grep 'hermes.*gateway\|hermes.*serve' | grep -v grep

# Restart (use at-job to avoid self-kill)
echo "kill <PID> && hermes serve &" | at now
```