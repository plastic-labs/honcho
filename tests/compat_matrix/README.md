# Compatibility matrix

Rolling Honcho back is an image swap. It never runs a downgrade. So the previous
release has to keep working on the schema a candidate migrates to. That has to
hold for data the candidate wrote, and while the candidate's backfills are still
running. This harness checks it. CI runs it from
`.github/workflows/compat-matrix.yml` on pull requests that touch `migrations/`,
`src/models.py`, `src/crud/`, `src/reconciler/`, `src/deriver/` or the queue
payload schemas, on the same paths on `main`, and on demand.

## What it does

Both images run side by side in Docker Compose (`compose.yml`) against one
Postgres and one Redis. A stub provider (`llm_stub.py`) answers every LLM and
embedding call deterministically, so the deriver, summarizer and dreamer run for
real without API keys.

1. **Bootstrap.** The old image builds its schema with `scripts/provision_db.py`,
   the same command the deployment's init container runs.
2. **Seed.** The old API and worker write data. Rows in the legacy JSONB
   provenance shapes are then inserted directly, and work is left queued with no
   worker running.
3. **Migrate.** The new image runs `python -m src.migrate`.
4. **Round A, backfills incomplete.** The new image runs one half-size batch of
   every registered backfill, so the database holds both shapes. Then every
   pairing of writer API (old, new) and worker (old, new, both at once) does a
   full pass:
   - create workspaces, peers, sessions, messages and conclusions, with nested,
     empty and null-valued metadata;
   - queue representation, summary and dream work for the worker to process;
   - delete a conclusion, and delete a session whose derived memory must go too.
5. **Round B.** The new image drains every backfill with the reconciler's own
   cycle, and the pairings run again.

Every write is recorded in a ledger (`ledger.py`). After every pass, the whole
ledger is read back through both APIs. Two things must hold: both images read
each row the same way it was written, and deleted rows stay gone. The run also
fails if a queue item fails, if a backfill is still pending after the drain, or
if the schema revision moves during the run.

The reconciler is held off with a day-long interval, so backfills move only when
the driver moves them.

## Running it locally

```bash
docker build -t honcho-candidate:local .
```

```bash
uv run python -m tests.compat_matrix.run --old-image ghcr.io/plastic-labs/honcho:v3.3.0 --new-image honcho-candidate:local
```

`--lineage` picks how the database is laid out, and `--keep` leaves the stack up
for inspection. On failure, service logs go to `compat-matrix-logs/`. A pass
takes about four minutes.

## Lineages

- **`dedicated`**: one database per tenant. The whole migration chain is replayed
  into a named schema (`DB_SCHEMA=<app_name>` in deployment, `cmx_tenant` here).
- **`self-hosted`**: the same chain, replayed into `public`.
- **`shared`**: not runnable yet, and the driver exits with the reasons. The
  shared pool is a hash-partitioned schema built by a bootstrap and then stamped,
  rather than replayed. Running it here needs three things that don't exist yet:
  the bootstrap and its row-level-security policies checked in, the revision it
  was stamped at, and a released image with the multi-tenant code to act as the
  old side.

## Adding scenarios

Add writes to `run_combo` in `scenarios.py`. Each write needs a matching
expectation in the ledger; a write with no expectation tests nothing. Suppress
and restore don't exist in the API yet. When they land, their scenarios go next
to the deletes.
