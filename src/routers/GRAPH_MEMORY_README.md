# Graph Memory Module — Operational Guide

**Part of:** Honcho + ngram Integration (Approach A)
**Phase:** 1 — New tables, API endpoints, SQL CTE recall, Redis context state, auth/authz
**Location:** `src/routers/graph_memory.py`, `src/crud/graph_memory.py`, `src/schemas/graph_memory.py`, `src/models.py`
**Migration:** `migrations/versions/2a3b4c5d6e7f_add_graph_memory_tables.py`

---

## Overview

The graph memory module adds semantic-network capabilities to Honcho's existing vector-store memory. It enables:

- **Typed edges** between observations (related, refines, supersedes, contradicts, etc.)
- **Spreading-activation recall** via SQL recursive CTE (not in-process BFS)
- **Named contexts** for workstream isolation (1:many thread→context mapping)
- **Two-axis decay** (activation + confidence) derived from an append-only access log
- **Per-pin verify cadence** (null default, confidence-threshold backstop)
- **Source-diversity weighting** (prevents self-reinforcement loops)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Honcho (PostgreSQL)                            │
│                                                                    │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐  │
│  │   Documents             │  │   Edges                         │  │
│  │   (existing)            │  │   (new)                        │  │
│  └────────────────────────┘  └─────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │   Access Log (new) — append-only, derived activation/conf   │   │
│  └────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────┐  ┌──────────────────────────────┐   │
│  │   Context Index (new)  │  │   Thread Binding Reg (new)   │   │
│  └────────────────────────┘  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
         ▲
         │ REST API (all endpoints authenticated)
         │
  ┌──────┴──────┐
  │  API Server │
  └─────────────┘
         │
  ┌──────┴──────┐
  │  Redis      │  ← Active context state (ephemeral, TTL-extended)
  └─────────────┘
```

---

## API Endpoints

All endpoints are under `/v3/workspaces/{workspace_id}/graph-memory/` and require JWT authentication.

### Edges

| Method | Path | Description |
|---|---|---|
| POST | `/edges` | Create edge (convergence-upsert via SQL ON CONFLICT) |
| POST | `/edges/list` | List edges with optional filters |
| DELETE | `/edges/{edge_id}` | Delete an edge |

### Recall

| Method | Path | Description |
|---|---|---|
| POST | `/recall` | Spreading-activation recall (SQL recursive CTE) |

**Request body:**
```json
{
  "query": "memory retrieval performance",
  "collection_name": "my-collection",
  "max_depth": 3,
  "frontier_cap": 10,
  "token_budget": 2000,
  "context": "humane-economy",
  "include_pinned": true
}
```

### Contexts

| Method | Path | Description |
|---|---|---|
| POST | `/contexts` | Create a named context |
| POST | `/contexts/{name}/members` | Add observation to context |
| DELETE | `/contexts/{name}/members/{obs_id}` | Remove observation from context |
| GET | `/contexts/{name}/members` | List context members |

### Context Switching (Redis-backed)

| Method | Path | Description |
|---|---|---|
| POST | `/peers/{peer_id}/context-switch` | True swap: page-in + page-out |
| POST | `/peers/{peer_id}/context-activate` | Additive page-in |
| POST | `/peers/{peer_id}/context-evict` | Explicit page-out |

### Thread Bindings

| Method | Path | Description |
|---|---|---|
| POST | `/thread-bindings` | Bind thread to context (rebinding denied) |
| GET | `/thread-bindings/{thread_id}` | Resolve thread → context |

### Pinning & Verification

| Method | Path | Description |
|---|---|---|
| POST | `/observations/{obs_id}/pin` | Pin observation (quota: 100/persona) |
| DELETE | `/observations/{obs_id}/pin` | Unpin observation |
| POST | `/observations/{obs_id}/verify` | Record verification event |
| GET | `/observations/verify-due` | List observations needing verification |

### Administration

| Method | Path | Description |
|---|---|---|
| POST | `/access-log` | Append access log event |
| POST | `/access-log/compact` | Compact access log (prune events >5 half-lives) |
| POST | `/evict-stale` | Evict stale unpinned observations |

---

## Security Model

### Authentication
All endpoints require a valid JWT. Use Honcho's existing key management API to create tokens.

### Authorization
- **Workspace-scoped:** JWT must include the workspace name. Cross-workspace queries are denied.
- **Peer-scoped:** Peer-scoped tokens can only access their own resources.
- **Admin-scoped:** Admin tokens can access all workspaces.

### Rate Limiting
| Endpoint Group | Limit | Window |
|---|---|---|
| Edge creation | 100 requests | 60 seconds |
| Recall queries | 60 requests | 60 seconds |
| Context operations | 50 requests | 60 seconds |
| Access log writes | 1000 events | 60 seconds |
| Pin toggles | 10 operations | 3600 seconds |

### Resource Quotas
| Resource | Default Quota |
|---|---|
| Pins per persona | 100 |
| Edges per persona | 10,000 (planned) |

### Input Validation
- `edge_type` must be one of: `related`, `composes-with`, `see-also`, `refines`, `supersedes`, `contradicts`
- `context_name` must match `^[a-zA-Z0-9_-]{1,64}$`
- `thread_id` must match Slack thread_ts format: `^[0-9]{10,}\.[0-9]+$`
- `verify_cadence_days` must be between 1 and 3650
- `created_by` is NEVER user-supplied — derived from authenticated JWT identity

---

## Decay Model

### Activation
Derived at query time from the access log:

```
activation(obs, now) = Σ(distinct_sources) Σ(events from that source)
                       weight(event) * exp(-Δt / half_life)
```

| Event Type | Weight |
|---|---|
| `access` | 0.3 |
| `verify` | 1.0 |
| `recall` | 0.5 |
| `promote` | 1.0 |
| `rehydrate` | 1.0 |
| `evict` | 0.0 |

Same-source repeats get diminishing returns: `repeat_factor = 1 / (1 + ln(1 + n))`

### Confidence
Pure function of last_verify and now — NO compounding:

```
confidence(obs, now) = exp(-(now - last_verify) / verify_half_life)
```

- Half-life: 30 days
- Threshold: 0.3 (confidence below this → flagged as verify-due)

### Pinned Floor
Pinned observations get `activation = max(computed, 0.85)`. Confidence still decays (pins remain falsifiable).

---

## Verify-Due Triggers

1. **Explicit cadence** (pins only, activation-independent): fires when `now - last_verify ≥ verify_cadence_days`
2. **Confidence threshold** (all observations, always active): fires when confidence < 0.3

Default cadence is null (no explicit cadence — confidence threshold alone handles it).

---

## Log Compaction

The access log is append-only. Events older than 5 activation half-lives (~5 days) are pruned by periodic compaction. Their contribution to activation is `exp(-5) ≈ 0.007` — negligible.

### Manual Compaction
```bash
POST /v3/workspaces/{id}/graph-memory/access-log/compact
```

Returns a gap-note style report:
```json
{
  "pruned_events": 46,
  "retention_policy": {
    "half_lives": 5,
    "activation_half_life_hours": 24,
    "cutoff_age_hours": 120,
    "cutoff_timestamp": "2026-06-18T17:45:00+00:00"
  },
  "pre_compaction": {
    "total_events": 100,
    "oldest_event": "2026-06-10T12:00:00+00:00",
    "newest_event": "2026-06-23T17:45:00+00:00"
  },
  "post_compaction": {
    "remaining_events": 54,
    "pruned_percentage": 46.0
  },
  "health": "healthy",
  "note": "Pruned 46 events older than 5 activation half-lives (~5 days). Their contribution to activation was exp(-5) ≈ 0.0067 — negligible."
}
```

### Automatic Compaction
The compaction scheduler runs as a background task in the deriver process (sibling to the reconciler and promotion schedulers). It compacts the access log every 24 hours.

**GC protocol alignment:** The compaction follows the graceful-compact (GC) protocol pattern from the agentc conventions:
- **Proactive, not reactive** — runs on a fixed schedule, not waiting for forced compaction
- **Gap-note style report** — logs what was pruned, what survived, and why
- **Version-anchored** — the retention policy version is included in every report
- **Post-compaction health check** — verifies the log is in a healthy state after pruning

---

## Running Tests

### Prerequisites
- Docker containers running (PostgreSQL, Redis, Honcho API)
- Test files copied into the container

### Run All Phase 1 Validation
```bash
docker exec honcho-selfhost-api-1 sh -c 'cd /app && .venv/bin/python3 tests/unit/validate_phase1.py'
```

This runs 26 tests covering:
- Schema validation (edge types, context names, thread IDs, pin cadence, recall bounds)
- CRUD logic (activation decay, confidence decay, source-diversity, pinned floor)

### Run Migration Verification
```bash
docker exec honcho-selfhost-api-1 sh -c 'cd /app && .venv/bin/python3 tests/unit/verify_migration.py'
```

This verifies:
- All 4 new tables exist with correct columns
- Indexes and foreign keys are in place
- Migration can be rolled back and re-applied

### Run Simulation Regression
```bash
cd /home/claw/agentc && python3 workshop/experiments/ngram-honcho-bridge/sim_v3.py
```

This runs the simulation with 10 invariants and concurrent access test.

### Copy Updated Files to Container
After making changes to any graph memory files:
```bash
docker cp src/models.py honcho-selfhost-api-1:/app/src/models.py
docker cp src/schemas/graph_memory.py honcho-selfhost-api-1:/app/src/schemas/graph_memory.py
docker cp src/crud/graph_memory.py honcho-selfhost-api-1:/app/src/crud/graph_memory.py
docker cp src/routers/graph_memory.py honcho-selfhost-api-1:/app/src/routers/graph_memory.py
docker cp src/utils/types.py honcho-selfhost-api-1:/app/src/utils/types.py
docker cp src/main.py honcho-selfhost-api-1:/app/src/main.py
docker cp migrations/versions/2a3b4c5d6e7f_add_graph_memory_tables.py honcho-selfhost-api-1:/app/migrations/versions/
docker cp tests/unit/validate_phase1.py honcho-selfhost-api-1:/app/tests/unit/validate_phase1.py
docker exec -u root honcho-selfhost-api-1 sh -c 'chown 100:101 /app/src/models.py /app/src/schemas/graph_memory.py /app/src/crud/graph_memory.py /app/src/routers/graph_memory.py /app/src/utils/types.py /app/src/main.py /app/migrations/versions/2a3b4c5d6e7f_add_graph_memory_tables.py /app/tests/unit/validate_phase1.py && chmod 644 /app/src/models.py /app/src/schemas/graph_memory.py /app/src/crud/graph_memory.py /app/src/routers/graph_memory.py /app/src/utils/types.py /app/src/main.py /app/migrations/versions/2a3b4c5d6e7f_add_graph_memory_tables.py /app/tests/unit/validate_phase1.py'
```

---

## File Manifest

### Graph Memory Core

| File | Purpose |
|---|---|
| `src/utils/types.py` | `EdgeType`, `AccessLogEventType` type literals |
| `src/models.py` | `Edge`, `AccessLogEntry`, `ContextIndex`, `ThreadBinding` SQLAlchemy models |
| `src/schemas/graph_memory.py` | Pydantic request/response schemas |
| `src/crud/graph_memory.py` | CRUD functions (activation/confidence derivation, edges, contexts, thread bindings, pinning, verify, eviction) |
| `src/routers/graph_memory.py` | FastAPI router with 18 endpoints |
| `src/main.py` | Router wired into app |

### Promotion Worker

| File | Purpose |
|---|---|
| `src/deriver/promotion.py` | Promotion worker: heuristic/LLM test, vector similarity (`_get_related_observation_ids`), edge creation, intent-aware chunking (`_embed_observation`) |
| `src/deriver/promotion_scheduler.py` | Scans for un-promoted observations every 60s, enqueues promotion tasks |
| `src/deriver/compaction_scheduler.py` | Compaction scheduler (compacts access log every 24h, GC protocol aligned) |
| `src/utils/work_unit.py` | Work unit key construction/parsing — supports `promotion` task type (format: `promotion:{workspace}:{observed}:{obs_id}`) |
| `src/utils/queue_payload.py` | `PromotionPayload` and other task payloads |

### Migrations

| File | Purpose |
|---|---|
| `migrations/versions/2a3b4c5d6e7f_*.py` | Creates graph memory tables |
| (later migration) | Adds `promotion_failed`, `promotion_attempts`, `promotion_error`, `promoted_at` columns to `documents` |

### Hermes Agent Integration

| File | Purpose |
|---|---|
| `~/.hermes/hermes-agent/plugins/memory/honcho/__init__.py` | `honcho_recall`, `honcho_recall_context`, `honcho_thread_bind` tools (PR #4, merged 2026-06-27) |

### Tests & Simulation

| File | Purpose |
|---|---|
| `tests/unit/validate_phase1.py` | Schema + CRUD logic validation (26 tests) |
| `tests/unit/verify_migration.py` | Migration verification (tables, indexes, FKs, rollback) |
| `tests/unit/verify_reapply.py` | Quick re-apply check |
| `workshop/experiments/ngram-honcho-bridge/sim_v3.py` | Simulation (10 invariants, concurrent access test) |
| `workshop/experiments/ngram-honcho-bridge/concrete-spec.md` | Full specification |
| `workshop/experiments/ngram-honcho-bridge/phase1-validation-strategy.md` | Test plan |
| `workshop/experiments/ngram-honcho-bridge/process-template.md` | Implementation process template |

---

## Full Setup Guide

See **[docs/GRAPH_MEMORY_SETUP.md](../../docs/GRAPH_MEMORY_SETUP.md)** for:
- Complete installation from scratch
- `.env` configuration reference
- Docker rebuild procedures
- Bug fixes applied (2026-06-27)
- Troubleshooting guide
