# Queue Item Schema and State Transitions (Phase 4 Step 1 reference)

> Companion to `2026-06-15-rust-full-migration.md`. This is the authoritative
> extraction of the Python queue contract that `worker-rs` must reproduce
> byte-for-byte before any worker cutover. Everything here is derived from the
> Python sources cited inline; no behavior is invented. Where Python has a quirk,
> the quirk is documented as the contract.

**Sources of truth:**

- `src/models.py` — `QueueItem` (`queue` table), `ActiveQueueSession`
  (`active_queue_sessions` table).
- `src/deriver/enqueue.py` — the producer (queue-row construction).
- `src/utils/queue_payload.py` — per-task payload Pydantic models + JSON dump rules.
- `src/utils/work_unit.py` — `construct_work_unit_key` / `parse_work_unit_key`.
- `src/deriver/queue_manager.py` — the consumer state machine.
- `src/schemas/internal.py` (`ReconcilerType`), `src/schemas/configuration.py` (`DreamType`).

---

## 1. Tables

### 1.1 `queue` (`QueueItem`, `src/models.py:477`)

| Column | Type | Null | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | bigint identity | no | identity | PK, autoincrement. Ordering key for FIFO within a work unit. |
| `session_id` | text | yes | — | FK → `sessions.id` (the internal nanoid `id`, **not** `name`). Indexed. |
| `work_unit_key` | text | no | — | Logical processing lane. See §3. |
| `task_type` | text | no | — | One of the task types in §2. Stored as plain text. |
| `payload` | jsonb | no | — | Task payload. See §4. |
| `processed` | bool | no | `false` | The **only** completion flag. Indexed. |
| `error` | text | yes | — | Set (with `processed=true`) when an item terminally errors. |
| `created_at` | timestamptz | no | `now()` | Indexed. |
| `workspace_name` | text | yes | — | FK → `workspaces.name`. Indexed. |
| `message_id` | bigint | yes | — | FK → `messages.id` (the bigint PK, **not** `public_id`). |

**Indexes (`src/models.py:503`):**

- `ix_queue_message_id_not_null` — partial on `message_id WHERE message_id IS NOT NULL`.
- `ix_queue_work_unit_key_processed_id` — `(work_unit_key, processed, id)`. This is
  the index the consumer's "next item" and "claimable work units" queries ride.
- `uq_queue_reconciler_pending_work_unit_key` — **unique** on `work_unit_key`
  `WHERE task_type = 'reconciler' AND processed = false`. Dedups pending reconciler tasks.
- `uq_queue_dream_pending_work_unit_key` — **unique** on `work_unit_key`
  `WHERE task_type = 'dream' AND processed = false`. Dedups pending dream tasks.

There is **no `status`/`attempts`/`locked_at` column.** All "state" is the pair
`(processed, error)` plus the existence of an `active_queue_sessions` row for the
work unit. The Rust worker must not add columns; it must reproduce this model.

### 1.2 `active_queue_sessions` (`ActiveQueueSession`, `src/models.py:536`)

| Column | Type | Null | Default | Notes |
| --- | --- | --- | --- | --- |
| `id` | text | no | `generate_nanoid` | PK. The "claim token" (`aqs_id`) a worker holds. |
| `work_unit_key` | text | no | — | **UNIQUE**. This is the work-unit mutex. |
| `last_updated` | timestamptz | no | `now()`, `onupdate=now()` | Heartbeat; drives stale recovery. |

One row per work unit currently being processed. Its uniqueness on
`work_unit_key` is the entire concurrency-control mechanism: claiming = inserting
a row, releasing = deleting it, stale recovery = deleting rows whose
`last_updated` is older than the timeout.

---

## 2. Task types

`task_type` is the `TaskType` literal union (`src/utils/types.py:236`):
`"webhook" | "summary" | "representation" | "dream" | "deletion" | "reconciler"`,
stored as plain `TEXT`. The producer/consumer behavior per type:

| `task_type` | Producer | Consumer path | Batched? |
| --- | --- | --- | --- |
| `representation` | `enqueue()` on message create | `process_representation_batch` | **Yes** (token-capped batch) |
| `summary` | `enqueue()` on message create | `process_item` | No (one at a time) |
| `deletion` | `enqueue_deletion()` / HTTP session-delete | `process_item` | No |
| `dream` | `enqueue_dream()` / `DreamScheduler` | `process_item` | No |
| `reconciler` | `ReconcilerScheduler` | `process_item` | No |
| `webhook` | webhook delivery path | `process_item` | No |

Enums for payload sub-types:

- `ReconcilerType` (`src/schemas/internal.py:16`): `sync_vectors`, `cleanup_queue`.
- `DreamType` (`src/schemas/configuration.py:16`): `omni`.

---

## 3. `work_unit_key` formats (`src/utils/work_unit.py`)

The key is a `:`-delimited string. **It is parsed by splitting on `:`** — so no
component may contain `:` or parsing breaks (a real constraint the producer
relies on workspace/peer/session names not violating). Formats:

| `task_type` | `construct` format | Parse arity |
| --- | --- | --- |
| `representation` | `representation:{workspace}:{session}:{observed}` | 4 parts (legacy 5-part `…:{observer}:{observed}` also parsed) |
| `summary` | `summary:{workspace}:{session}:{observer}:{observed}` | exactly 5 |
| `dream` | `dream:{dream_type}:{workspace}:{observer}:{observed}` | exactly 5 |
| `webhook` | `webhook:{workspace}` | exactly 2 |
| `deletion` | `deletion:{workspace}:{deletion_type}:{resource_id}` | exactly 4 |
| `reconciler` | `reconciler:{reconciler_type}` | exactly 2 |

Notes:
- For `representation` and `summary`, missing `observer`/`observed`/`session_name`
  default to the literal string `"None"` (`construct_work_unit_key`,
  `src/utils/work_unit.py:45`). Representation deliberately omits `observer` from
  the key because one representation pass writes to multiple observer collections.
- `summary`'s key uses `observer`/`observed` slots that the summary producer does
  not populate, so in practice they serialize as `"None"` → key shape
  `summary:{workspace}:{session}:None:None`.
- The session-delete HTTP fast path already implemented in `api-rs`
  (`db::delete_session`) writes `deletion:{workspace}:session:{session}` — this
  matches `construct_work_unit_key` for `deletion_type=session, resource_id=session`.

---

## 4. Payload shapes (`src/utils/queue_payload.py`)

All payloads are Pydantic models with `extra="forbid"`, dumped with
**`model_dump(mode="json", exclude_none=True)`**. Two consequences the Rust
producer must match exactly:

1. **`mode="json"`** → `datetime` fields serialize as ISO-8601 strings.
2. **`exclude_none=True`** → any field whose value is `None` is **omitted** from
   the stored JSON (not stored as `null`). This matters for round-trip parity of
   stored rows.

Every payload carries its own `task_type` discriminator field (redundant with the
column, but present in the JSON).

### `representation` (`RepresentationPayload`)
```jsonc
{
  "task_type": "representation",
  "session_name": "<str>",
  "content": "<str>",          // message content
  "observers": ["<peer>", ...], // non-empty
  "observed": "<peer>",         // the sender
  "created_at": "<iso8601>",
  "configuration": { ...ResolvedConfiguration... }
}
```
`workspace_name` and `message_id` are **not** in the payload — they live in the
dedicated `queue` columns (`create_payload` docstring, `queue_payload.py:143`).

### `summary` (`SummaryPayload`)
```jsonc
{
  "task_type": "summary",
  "session_name": "<str>",
  "message_seq_in_session": <int>,
  "configuration": { ...ResolvedConfiguration... },
  "message_public_id": "<str|omitted>"  // omitted when None (exclude_none)
}
```

### `dream` (`DreamPayload`)
```jsonc
{
  "task_type": "dream",
  "dream_type": "omni",
  "observer": "<peer>",
  "observed": "<peer>",
  "session_name": "<str|omitted>",
  "trigger_reason": "<str|omitted>",
  "delay_reason": "<str|omitted>",
  "documents_since_last_dream_at_schedule": <int|omitted>,
  "document_threshold": <int|omitted>
}
```

### `deletion` (`DeletionPayload`)
```jsonc
{
  "task_type": "deletion",
  "deletion_type": "session" | "observation" | "workspace",
  "resource_id": "<str>"
}
```
Matches the payload `api-rs` already writes on session soft-delete.

### `reconciler` (`ReconcilerPayload`)
```jsonc
{ "task_type": "reconciler", "reconciler_type": "sync_vectors" | "cleanup_queue" }
```

### `webhook` (`WebhookPayload`)
```jsonc
{ "task_type": "webhook", "event_type": "<str>", "data": { ... } }
```

### The full queue **row** (what `enqueue.py` inserts)
Each `create_*_record` returns a dict inserted as a `queue` row:
```jsonc
{
  "work_unit_key": "<§3>",
  "payload": { ...§4... },
  "session_id": "<session.id|null>",   // representation/summary: session.id; dream/deletion: null
  "task_type": "<§2>",
  "workspace_name": "<str|null>",
  "message_id": <int|null>             // representation/summary: messages.id; dream/deletion: null
}
```

---

## 5. Producer flow (message create → queue rows)

`enqueue(payloads)` (`src/deriver/enqueue.py:25`), called as a FastAPI
background task after `create_messages`:

1. **Dream cancellation:** for each payload with `workspace_name`+`peer_name`,
   cancel pending dreams for `observed=peer_name` via the in-process
   `DreamScheduler` (best-effort; no DB queue effect). The Rust worker/API must
   reproduce this cancellation when it owns dreams.
2. Open a `tracked_db("message_enqueue")` session. Empty payload list → return.
3. `handle_session` (`enqueue.py:81`):
   - `get_or_create_session` (idempotent).
   - Load workspace; resolve session-level config (`get_configuration`).
   - Load per-peer config + active flags (`get_peers_with_configuration`).
   - For each message, resolve message-level config (falls back to session-level
     when the message carries no `configuration`), then `generate_queue_records`.
4. `generate_queue_records` (`enqueue.py:293`) per message:
   - Resolve `message_seq_in_session` (prefer the value captured at create; else
     query).
   - **Summary record** iff `conf.summary.enabled` **and**
     (`seq % messages_per_short_summary == 0` **or** `seq % messages_per_long_summary == 0`).
   - **Representation record** iff `conf.reasoning.enabled` and there is ≥1
     observer. Observers are computed as:
     - sender self-observes iff `get_effective_observe_me(sender)` (session-peer
       `observe_me` overrides peer `observe_me`; default `True`);
     - plus every *active* other peer whose session-peer `observe_others` is true.
     - If no observers → no representation record (summary may still exist).
5. Bulk `INSERT ... RETURNING` all queue records, commit. Failures are swallowed
   (logged + optional Sentry) — enqueue never raises into the request path.

**Implication for the Rust message-write slice:** producing the queue rows
requires `ResolvedConfiguration` resolution (workspace→session→message hierarchy),
the observer-selection logic above, and the summary-threshold arithmetic — none of
which exist in `api-rs` yet. This is why message-write is gated behind this phase.

---

## 6. State model

There is no explicit status. Derive logical state as:

| Logical state | Condition |
| --- | --- |
| **PENDING** | `queue.processed = false` |
| **CLAIMED / "processing"** | a row exists in `active_queue_sessions` for the item's `work_unit_key` |
| **DONE (completed)** | `queue.processed = true AND error IS NULL` |
| **DONE (failed, terminal)** | `queue.processed = true AND error IS NOT NULL` |

Key truth: **claiming is per work unit, not per item.** Many `queue` rows can
share a `work_unit_key`; one `active_queue_sessions` row covers all of them. A
work unit is "available" when it has ≥1 unprocessed item and **no**
`active_queue_sessions` row.

---

## 7. State transitions (consumer, `queue_manager.py`)

```
                 enqueue (INSERT processed=false)
                              │
                              ▼
                        ┌───────────┐
                        │  PENDING  │  (no AQS row for work_unit_key)
                        └─────┬─────┘
            claim_work_units  │  INSERT active_queue_sessions(work_unit_key)
            ON CONFLICT DO    │  ON CONFLICT DO NOTHING  → returns aqs_id
            NOTHING (mutex)   ▼
                        ┌───────────┐
                        │  CLAIMED  │  (AQS row exists; worker holds aqs_id)
                        └─────┬─────┘
        per item / per batch  │  process_item / process_representation_batch
                  ┌───────────┼────────────────────┐
        success   │           │ error              │ worker crash / stall
                  ▼           ▼                     ▼
        processed=true   processed=true,       AQS.last_updated ages past
        error=NULL       error=msg (FIRST      STALE_SESSION_TIMEOUT_MINUTES
        + AQS heartbeat  item only) +          → cleanup_stale_work_units
                         AQS heartbeat         DELETEs AQS (FOR UPDATE SKIP
                  │           │                LOCKED) → item still
                  └─────┬─────┘                processed=false → re-PENDING
                        ▼
              loop: next unprocessed item for work_unit_key?
                        │ yes → back to "process"
                        │ no  → finally: DELETE AQS by (id, work_unit_key)
                        ▼
                   work unit RELEASED (queue.empty webhook for
                   representation/summary if rows were processed)
```

### 7.1 Claim (`get_and_claim_work_units` → `claim_work_units`, `queue_manager.py:330`)
- Capacity = `WORKERS − owned_work_units`; 0 → skip.
- Candidate query: distinct `work_unit_key` of unprocessed items with **no**
  `active_queue_sessions` row, `LIMIT capacity`.
- Representation throttle: unless `FLUSH_ENABLED`, a `representation:` work unit is
  only eligible when its summed `messages.token_count` (over unprocessed items)
  `>= REPRESENTATION_BATCH_MAX_TOKENS`. Non-representation work units always pass.
- Claim = `INSERT active_queue_sessions(work_unit_key) ON CONFLICT DO NOTHING
  RETURNING work_unit_key, id`. Only rows that inserted are "won" (the unique
  constraint makes this the cross-worker, cross-process mutex). Returns
  `{work_unit_key: aqs_id}`.

### 7.2 Process loop (`process_work_unit`, `queue_manager.py:571`)
- Bounded by an asyncio `Semaphore(WORKERS)`.
- Re-verifies in-memory ownership each iteration; stops if lost.
- `representation`: `get_queue_item_batch` (token-capped context window, §8); call
  `process_representation_batch`; on success mark the whole batch processed.
- other types: `get_next_queue_item` (single, `ORDER BY id LIMIT 1`, joined to the
  AQS row to confirm ownership); `process_item`; mark processed.
- Loops until no unprocessed items remain for the work unit or shutdown.

### 7.3 Complete (`mark_queue_items_as_processed`, `queue_manager.py:1017`)
- `UPDATE queue SET processed=true WHERE id IN (...) AND work_unit_key=...`.
- `UPDATE active_queue_sessions SET last_updated=now() WHERE work_unit_key=...`
  (heartbeat — keeps a long-running work unit from being reaped as stale).

### 7.4 Error (`mark_queue_item_as_errored`, `queue_manager.py:1049`)
- **Only the first item** of the failing batch is marked
  `processed=true, error=msg[:65535]`. The rest stay `processed=false`.
- Heartbeats AQS like the success path.
- **There is no retry counter and no automatic re-queue of the errored item.** The
  errored item is terminal (it is now `processed=true`). "Retry" only means: the
  remaining unprocessed items in the work unit are attempted again on the next
  loop / next claim. This intentional design lets a poison message be skipped
  while the rest of the work unit still makes progress. The Rust worker MUST
  replicate "mark only the first, leave the rest pending" — not "fail the batch".

### 7.5 Release (`_cleanup_work_unit`, `queue_manager.py:1069`)
- On loop exit, `DELETE active_queue_sessions WHERE id=aqs_id AND work_unit_key=...`.
- If a row was actually deleted **and** ≥1 item was processed, and the task is
  `representation`/`summary`, publish a `queue.empty` webhook event.

### 7.6 Stale recovery (`cleanup_stale_work_units`, `queue_manager.py:301`)
- `cutoff = now() − STALE_SESSION_TIMEOUT_MINUTES`.
- `SELECT id FROM active_queue_sessions WHERE last_updated < cutoff ORDER BY
  last_updated FOR UPDATE SKIP LOCKED`, then `DELETE` those ids.
- Deleting the AQS row (without touching `queue.processed`) returns the work unit
  to PENDING so another worker can re-claim it. This is the crash-recovery path.
- Gated to run at most once per `STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS`
  (jittered) per instance; safe to run concurrently across instances via
  `SKIP LOCKED`.

---

## 8. Representation batching (`get_queue_item_batch`, `queue_manager.py:756`)

For representation work units, the consumer builds a **token-bounded context
window** rather than processing one message at a time:

- Re-verify ownership (`active_queue_sessions.id = aqs_id`).
- Find `min_unprocessed_message_id` for the work unit; optionally prepend the
  immediately-preceding message **iff it is from a different peer than `observed`**
  (conversational context).
- A CTE computes a running `cumulative_token_count` ordered by `message.id` and a
  window flag `cap_exceeded = bool_or(cumulative > REPRESENTATION_BATCH_MAX_TOKENS)`.
- Include messages while `cumulative <= cap`, **always** including the first
  unprocessed message even if it alone exceeds the cap.
- Trim the batch to a homogeneous leading run of identical `configuration`
  (`_resolve_batch_configuration`) — a config change boundary ends the batch.
- `hit_batch_token_cap` is reported true only when the config filter did **not**
  move the queue-item boundary AND the CTE saw a message past the cap.

This SQL is intricate; the Rust port must reproduce window-function semantics and
the "always include the first unprocessed" exception exactly, or representation
batches will diverge.

---

## 9. Dedup rules

- **Reconciler / dream**: partial unique indexes (§1.1) prevent two *pending* rows
  with the same `work_unit_key`. The Rust producer must rely on the same
  `ON CONFLICT DO NOTHING` semantics, not application-side checks alone.
- **Dream enqueue** (`enqueue_dream`, `enqueue.py:445`) additionally pre-checks
  both an existing `active_queue_sessions` row (in-progress) and an existing
  unprocessed `queue` row (pending) before inserting, and skips if either exists.

---

## 10. Concurrency model summary

- **Work-unit mutex:** uniqueness of `active_queue_sessions.work_unit_key` +
  `INSERT ... ON CONFLICT DO NOTHING`. No advisory locks in the queue path.
- **Per-instance parallelism:** `Semaphore(DERIVER.WORKERS)`; capacity feeds the
  claim `LIMIT`.
- **Stale recovery concurrency:** `FOR UPDATE SKIP LOCKED` on the stale select.
- **Message creation ordering (separate, in `crud.create_messages`):** takes
  `pg_advisory_xact_lock(hashtext(workspace), hashtext(session))` + `SET LOCAL
  lock_timeout='5s'` before computing `seq_in_session = last_seq + offset`. The
  Rust message-write slice must take the **same** advisory lock to preserve
  gap-free per-session sequencing under concurrency.

---

## 11. Config defaults (resolved in the test/dev env)

| Setting (`DERIVER.*`) | Default | Role |
| --- | --- | --- |
| `WORKERS` | `1` | Parallel work units per instance. |
| `POLLING_SLEEP_INTERVAL_SECONDS` | `1.0` | Base idle poll interval. |
| `POLLING_SLEEP_MAX_INTERVAL_SECONDS` | `30.0` | Backoff ceiling. |
| `POLLING_BACKOFF_ENABLED` | `True` | Grow interval while idle. |
| `POLLING_BACKOFF_MULTIPLIER` | `2.0` | Backoff growth factor. |
| `POLLING_JITTER_RATIO` | `0.5` | ± jitter on sleeps. |
| `POLLING_STARTUP_JITTER_SECONDS` | `30.0` | Pre-first-poll random delay. |
| `STALE_SESSION_TIMEOUT_MINUTES` | `5` | AQS heartbeat age → stale. |
| `STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS` | `60.0` | Stale-sweep gate (jittered). |
| `REPRESENTATION_BATCH_MAX_TOKENS` | `1024` | Representation batch token cap / claim threshold. |
| `FLUSH_ENABLED` | `False` | Bypass the batch-token claim threshold. |

These are behavior-affecting and must be parsed by `worker-rs` config (Rust does
not parse them yet).

---

## 12. Mapping to Phase 4 Step 2 test states

The plan's Step 2 lists transitions to cover. Mapped to this schema:

| Plan transition | Concrete meaning here |
| --- | --- |
| `pending -> processing` | Insert `active_queue_sessions(work_unit_key)` via `ON CONFLICT DO NOTHING`; item still `processed=false`. |
| `processing -> completed` | `queue.processed=true, error=NULL`; AQS heartbeat; AQS deleted on work-unit drain. |
| `processing -> failed with retry` | First failing item → `processed=true, error=msg`; **remaining** items stay `processed=false` and are retried (the errored item itself is terminal — there is no per-item retry counter). |
| `stale processing -> pending` | `cleanup_stale_work_units` DELETEs the aged AQS row; items remain `processed=false`, so the work unit is re-claimable. |
| workspace/session ordering | Items ordered by `queue.id` within a work unit; representation batches add the token-window + same-config constraints in §8; message sequencing uses the advisory lock in §10. |

---

## 13. Port gotchas checklist (for `worker-rs`)

- [ ] No new columns: model state as `(processed, error)` + AQS existence only.
- [ ] `exclude_none` payload semantics — omit, don't null, optional fields.
- [ ] `datetime` payload fields as ISO-8601 strings (`mode="json"`).
- [ ] `session_id` / `message_id` are the **internal** ids (`sessions.id`,
      `messages.id`), not `name` / `public_id`.
- [ ] `work_unit_key` components must never contain `:`; missing parts serialize as
      the literal `"None"`.
- [ ] Claim mutex = unique `work_unit_key` + `ON CONFLICT DO NOTHING`.
- [ ] Error path marks **only the first** batch item; the rest stay pending.
- [ ] Errored items are terminal (no retry counter); "retry" = remaining items.
- [ ] Heartbeat `active_queue_sessions.last_updated` on every processed/errored mark.
- [ ] Stale recovery DELETEs AQS only (never flips `processed`), `FOR UPDATE SKIP LOCKED`.
- [ ] Representation: token-capped window, always include first unprocessed, stop at
      config-change boundary.
- [ ] Reconciler/dream pending-dedup partial unique indexes are load-bearing.
- [ ] Message-write producer needs full `ResolvedConfiguration` resolution +
      observer selection + summary thresholds, and the per-session advisory lock.
