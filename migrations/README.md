# Migrations

Honcho changes quickly, and every release has to upgrade self-hosted and managed instances
with as little friction as possible. Every revision in `versions/` follows these rules.

- **Expand, then contract.** A release can add schema or remove it, never both. Code
  that ships with a new column, table or index reads from and writes to both the old
  and new locations. Dropping or tightening the old location waits for a later release.
  `make_column_non_nullable_safe` in `migrations/utils.py` follows this pattern.
- **Online DDL only.** Add nullable columns or columns with a constant default, create
  tables, and create indexes `CONCURRENTLY`. A migration that builds an index has to
  check that the index is valid, not only that it exists. Never rewrite a large table.
  A table rename is allowed only as an expand step: the same transaction creates a view
  under the old name, and the rename gets a revision of its own. `migrations/env.py`
  runs the upgrade under a one-second `lock_timeout` and retries with jittered backoff
  (`run_with_lock_retry` in `src/migrate.py`), so you don't need to add your own.
- **Backfills are reconciler tasks.** The migration adds the column and stops. To copy
  data, register a `Backfill` in `src/reconciler/backfill.py`. You supply a
  `has_pending` predicate, a `count_pending` query and an idempotent `run_batch`, and
  state the `retire_after` condition. The reconciler enqueues it while the predicate
  holds, commits each batch in its own session, and exports
  `backfill_pending{task=<name>}`. `backfill_document_sources.py` is the reference
  implementation.
- **One lock for the whole run.** `python -m src.migrate [revision]` is the migrate
  entry point, and `scripts/migrate_db.py` wraps it. `migrations/env.py` holds a
  session-level advisory lock for the entire upgrade, so plain `alembic upgrade` is
  serialized too. A second migrator waits up to `DB_MIGRATION_LOCK_WAIT_SECONDS`
  (default 300), then fails and reports the pid that holds the lock.
- **One head.** `scripts/check_alembic_heads.py` fails when the migration graph has more
  than one head. It runs as a pre-push hook and in the Static Analysis workflow against
  the PR's merge ref. That ref is only as fresh as the last CI run, so rebase onto
  current `main` and let the check pass again before a migration PR merges.
- **The previous release has to run on the new schema.** Every revision needs a test in
  `tests/alembic/` (`uv run python scripts/run_alembic_tests.py`). The Compatibility
  Matrix workflow then runs the previous release image and the candidate side by side on
  the candidate's schema, with its backfills both half done and finished, and fails if
  either image cannot read what the other wrote. When you add a write path or a
  backfill, add a scenario for it in `tests/compat_matrix/` (see its README).

Maintainers take care of batching migrations into releases, declaring the compatibility
floor in `migrations/compat.toml`, and applying the revision graph to the shared and
dedicated databases. Don't edit `compat.toml` in a migration PR. Expect a migration PR to
wait for the next migration release.
