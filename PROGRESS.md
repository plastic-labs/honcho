# Progress

## Pin

Upstream remote: `https://github.com/plastic-labs/honcho.git` (`upstream`).

Pinned release: **v3.3.0**

Commit: `608aec8a928da13c540262ed980c569448fb53ab`

The fork's `main` was three commits behind that tag (changelog, trigram
index, contributing docs) plus the local goals commit. M0 merged the tag
in. `git describe` on the pin is `v3.3.0`.

## M0. Baseline and map

Built:

- Tracked `docker-compose.yml`: API, deriver, Postgres `pgvector/pgvector:pg15`, Redis 8.2.
- `docs/MAP.md` for the pinned tree.
- `NOTICE` and `FORK_CHANGES.md`.
- GitHub Actions runs `uv run pytest` on every pull request and every push to `main`.
- Acceptance tests in `tests/sentica/test_m0_baseline.py`.

Decisions:

- Local Compose still uses the upstream development database password and
  `AUTH_USE_AUTH=false`. Removing the password and requiring a client
  certificate is M1. Doing it in M0 would leave the pinned stack unable to
  boot, because the certificate proxy does not exist yet.
- Representation queue payloads still include message text
  (`src/utils/queue_payload.py:28`). The pinned deriver reads that field.
  Changing it here would diverge from the release and break the worker.
  Recorded in `docs/MAP.md`. A later milestone removes content from queue rows.
- Redis persistence is off and `/data` is tmpfs. API and deriver `/tmp` are
  tmpfs. Postgres data stays in the `pgdata` volume because the database is
  the store. The default vector store is pgvector, so the compose file does
  not mount a LanceDB volume.
- GCP, Fly, AWS, and live-provider workflows no longer run on push or pull
  request, and their external jobs are `if: false`. The upstream unit-test
  workflow still runs on GitHub-hosted runners with a local Postgres service.
- `docker-compose.yml.example` is unchanged so the upstream example stays
  diffable. `docker compose up` uses the tracked `docker-compose.yml`.
- Host ports default to 8000, 5432, and 6379 and can be moved with
  `SENTICA_API_PORT`, `SENTICA_POSTGRES_PORT`, and `SENTICA_REDIS_PORT`.
  This machine already had an unrelated Postgres container on 5432. The
  local proof used port 5433 and did not connect to that container.

Deviations from GOALS.md: none in the M0 deliverables. The two items above
(database password, queue message text) are left as upstream behavior until
the milestone that replaces them.

Test counts: `uv run pytest` on 2026-10-11, against the local Postgres
on host port 5433: **3140 passed, 53 skipped, 0 failed** (132.83s). The
53 skips are upstream (live provider and optional store tests). M0 adds
no skips. M0 acceptance: `tests/sentica/test_m0_baseline.py`, 7 passed.

Open questions: none for M0.
