# Plan: normalize `source_ids` into a `document_sources` table

Branch: `abigail/source-ids-table` (on top of `abigail/dev-2219`)

## Goal

Replace the JSONB `documents.source_ids` column with a proper edge table so
reasoning-tree linkage is queryable, indexed, and has one source of truth.
API shape is unchanged: `Conclusion.source_ids` stays `list[str] | None`.

## Schema

```
document_sources
  derived_id     TEXT PK, FK -> documents.id ON DELETE CASCADE
  source_id      TEXT PK  (NOT an FK: dreamer emits unresolvable IDs)
  position       INT  (preserves premise order)
  workspace_name TEXT FK -> workspaces.name
  INDEX (source_id, workspace_name)   -- reverse traversal, replaces GIN
  CHECK length(source_id)=21, nanoid format
```

## Changes

1. **models.py** — add `DocumentSource`; on `Document` drop the JSONB column +
   GIN index, add `source_links` relationship (`lazy="selectin"` — required,
   async lazy-load raises) and `source_ids` / `resolved_source_ids` properties.
2. **Migration** (one revision after `e4eba9cfaa6f`) — create table; backfill
   from `source_ids` column AND legacy `internal_metadata->'source_ids'`,
   dropping entries that don't match the nanoid regex; drop GIN index.
   The old column is KEPT (unwritten) for one release as a rollback net —
   follow-up migration drops it.
3. **crud/document.py** — insert sites build `DocumentSource` rows via
   `build_source_links()` helper (dedupes + drops malformed IDs);
   `get_child_observations` becomes a join instead of JSONB containment.
4. **utils/filter.py** — remove `source_ids` from `JSONB_COLUMNS`; special-case
   it (and new alias `parent_id`) to an EXISTS subquery. Semantics parity:
   scalar = membership, list = all present, `contains` = membership,
   `in` = any present. Existing filter tests are the spec.
5. **utils/representation.py, utils/agent_tools.py** — delete
   `internal_metadata.get("source_ids")` fallbacks (backfill retires them).

## Decision points

- **`/derived` endpoint**: `POST /conclusions/list` with
  `{"filters": {"parent_id": "<id>"}}` now covers it. Options:
  (a) keep both, (b) drop `/derived` before it ships in a release.
  Leaning (b) — one less route, avoids the `{conclusion_id}/derived`
  path-capture footgun. SDK `derived()` helpers can wrap the filter.
- **Garbage IDs**: backfill and write path silently drop malformed entries.
  They were already invisible to traversal; dry-run the backfill count on
  real data before merging.
- **Old column retention**: kept unmapped for one release (see #2).

## Costs

- Every Document query gains one batched selectin SELECT.
- Write amplification: N link rows per conclusion.
- Backfill migration over all deductive/inductive documents.

## Test plan

- Existing `tests/routes/test_conclusions.py` filter + `/derived` tests pass
  unchanged (parity spec).
- New: migration backfill test (pattern: `test_f1a2b3c4d5e6`), link
  dedupe/malformed-drop unit tests, `parent_id` filter tests.
