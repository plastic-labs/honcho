# Fork changes

Every change from plastic-labs/honcho is listed here. The pin is upstream
release v3.3.0 (`608aec8a928da13c540262ed980c569448fb53ab`).

| File | Why | Goal |
| --- | --- | --- |
| `NOTICE` | State that this is a modified AGPL fork and keep upstream attribution. | Ground rule 1 |
| `FORK_CHANGES.md` | Record divergences from upstream. | Ground rule 1 |
| `PROGRESS.md` | Milestone log. | Ground rule 8 |
| `docs/MAP.md` | Map of the pinned runtime: database, auth, storage, queue, model calls, prompts. | M0 |
| `docker-compose.yml` | Tracked local stack so `docker compose up` starts the API, deriver, Postgres with pgvector, and Redis. Redis persistence is off and its data directory is tmpfs. API and deriver `/tmp` are tmpfs. No LanceDB volume. Host ports default to 8000, 5432, and 6379 and can be moved with `SENTICA_API_PORT`, `SENTICA_POSTGRES_PORT`, and `SENTICA_REDIS_PORT`. | M0, ground rule 5 |
| `.gitignore` | Stop ignoring `/docker-compose.yml` and `/compose.yml` so the local stack can be committed. | M0 |
| `.github/workflows/unittest.yml` | Run the upstream test suite on every pull request and every push to `main`. Removed path filters and the changes gate. | M0 |
| `.github/workflows/push-gcp-registry-prod.yml` | Manual dispatch only, and both jobs are `if: false`. Do not push to Plastic Labs GCP or call its production webhook. | Ground rule 2 |
| `.github/workflows/push-gcp-registry-staging.yml` | Same as production, for the staging registry and webhook. | Ground rule 2 |
| `.github/workflows/unified-tests.yml` | Manual dispatch only. Fly and AWS jobs are `if: false`. | Ground rule 2 |
| `.github/workflows/live-llm-tests.yml` | Manual dispatch only. The live provider job is `if: false` so CI does not call external models or AWS. | Ground rule 2 |
| `.github/workflows/start-fly-runner.yml` | The Fly runner job is `if: false`. | Ground rule 2 |
| `tests/sentica/test_m0_baseline.py` | Acceptance checks for the pin, the map citations, the compose stack, and CI. | M0, ground rule 6 |
