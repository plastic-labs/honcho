---
name: property-repro
description: Reproduce a reported Honcho API bug (a 5xx, a wrong status code, data that doesn't round-trip) as a failing Hypothesis/Schemathesis test, then fix it. Use when an issue reports an unexpected API error, when hunting for 5xx bugs in a route, or before changing request validation or error handling.
---

# Reproducing API bugs with property tests

`tests/property/` fuzzes the `/v3` API in-process against a real Postgres test
database. Every test asserts that no response is a 5xx and that no unhandled
exception escapes (`TestClient` re-raises them). Start here when an issue says
"I got a 500 when…": the shortest path is usually to get one of these tests to
fail on the reported input, then fix the cause.

## The harnesses

| File | What it drives | Reach for it when |
|---|---|---|
| `test_api_no_server_errors.py` | Schemathesis: every operation, from the app's own OpenAPI schema, in positive and negative mode. Path and body IDs point at seeded resources (`seeded_ids`) most of the time. | The report is one request. |
| `test_api_inputs.py` | Hand-written strategies for what the schema doesn't describe: multipart uploads, the filter DSL, odd JSON in metadata/configuration, store-then-read round trips. | The input is a file, a filter, or a JSON value. |
| `test_api_stateful.py` | A `RuleBasedStateMachine` over workspaces, peers, sessions, messages, conclusions, including use after delete. | The bug needs a history ("after I deleted the session…"). |
| `test_api_auth.py` | The per-operation fuzzer with auth on and a random token per request (none, forged, expired, admin, workspace/peer/session/named-scope keys across two workspaces), checked against an oracle built from the route policy. | The report involves a scoped key, a 401/403, or data visible to the wrong caller. |
| `test_api_concurrency.py` | Batches of requests on colliding names fired at once over httpx's ASGI transport, each on its own DB connection; also checks get-or-create leaves exactly one row. | The bug is intermittent, a spurious 409, or only shows up under load. |

They are skipped unless pytest gets `--fuzz` (CI passes it; the pre-push hook
doesn't, to keep it fast), so every command below includes it. All of them
use `fuzz_client` (one DB session per request, as in production) from
`tests/property/conftest.py`.

## Workflow

1. **Try the broad fuzzer on the route first.** Select by operation label and
   raise the example count:

   ```bash
   HONCHO_FUZZ_EXAMPLES=1000 uv run pytest tests/property/test_api_no_server_errors.py \
     --fuzz -k "messages and POST and not list and not upload" -n 0 -p no:cacheprovider \
     --tb=line --show-capture=no
   ```

   A failure prints the exception type and a curl command for the shrunk request.

2. **If it passes, check that the route is actually being reached.** A route that
   only ever gets 404/422 is only being fuzzed on its error paths:

   ```bash
   HONCHO_FUZZ_STATS=/tmp/stats.txt uv run pytest --fuzz tests/property/test_api_no_server_errors.py
   awk '{k=$1" "$2" "$3; t[k]++; if ($4<300) ok[k]++} END {for (k in t) if (!ok[k]) print k}' /tmp/stats.txt
   ```

   Fix coverage before concluding there is no bug: add the ID field to
   `BODY_ID_FIELDS`, seed the resource the route needs, or write a targeted
   strategy in `test_api_inputs.py`.

3. **Pin the reported input.** Put the issue's exact input in a test, either as
   an `@example(...)` on a property or as a plain test, and confirm it fails
   before you touch `src/`.

4. **Fix the cause, not the status code.** Follow the existing conventions:
   - NUL (`\x00`) is *stripped* at the schema boundary (`NulStripped`,
     `strip_nul` in `src/utils/sanitization.py`), not rejected. A new free-form
     text or JSON field needs the same treatment.
   - A handler that builds a create schema from a path ID (get-or-create of a
     parent workspace or session) goes through `schema_from_path`
     (`src/utils/path_params.py`), which turns a bad ID into a 422 at
     `path.<param>`. A `ValidationError` anywhere else is a server bug and stays
     a 500.
   - Values Postgres can't store are a 422 only for an explicit allowlist of
     SQLSTATEs (`_is_input_data_error` in `src/main.py`). Widen it only for an
     input class, and match narrowly when the SQLSTATE also covers server bugs
     (see the 22P02 surrogate case).
   - Library parse failures on uploads become a `ValidationException`, caught
     around the library call only.
   - Never add a blanket `except Exception` to turn a 500 into a 4xx.

5. **Leave a deterministic regression test** in `tests/routes/` (most land in
   `test_validation_api.py`) that sends the shrunk input. Hypothesis runs with
   `database=None`, so a property failure is not replayed on the next run
   unless you pin it.

## Gotchas

- Background work does not run under test (the deriver, deletion cleanup, the
  reconciler). Only the "deletion pending" state is reachable; LLM and
  embedding calls are mocked.
- DB state persists across Hypothesis examples within one test function. That
  helps (more collisions) but means a failure can depend on earlier examples;
  re-run with `-n 0` and a fixed `--hypothesis-seed` to replay it.
- Schemathesis' positive mode respects the OpenAPI schema. When the server
  enforces a constraint the schema doesn't declare (e.g. `peer_id` must match
  the name pattern), positive-mode requests get 422s and never reach the
  handler. That gap is a schema bug in its own right.
- Default example counts are small so the suite stays fast. Raise
  `HONCHO_FUZZ_EXAMPLES` (and `HONCHO_FUZZ_STEPS` for the state machine)
  locally when hunting.
