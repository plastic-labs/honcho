# sentica-memory: build goals

sentica-memory is Sentica's memory service: a public fork of plastic-labs/honcho (AGPL-3.0), changed only where Sentica's isolation and safety rules require it. Sentica (a separate, private repository) calls it over HTTP through its own memory broker. This file is the source of truth for what to build. Work through the milestones in order. Each milestone lists its goal, what must be true when it is done, and how that is proven.

## Ground rules (apply to every milestone)

1. **Stay close to upstream.** Change Honcho only where a goal requires it. Prefer configuration, new modules and small hooks over edits to upstream files. Record every divergence from upstream in `FORK_CHANGES.md` (file, why, which goal). Keep the license, upstream copyright notices and attribution intact; add a `NOTICE` stating this is a modified fork.
2. **Local only.** Run everything with Docker Compose on this machine. Never connect to any real Sentica, DigitalOcean, AWS or production system. Never commit secrets, real customer data or real keys; `.env.example` holds variable names only. This repository is public.
3. **Fail closed.** Any missing, unknown or ambiguous tenant, workspace, token or label is refused with a clear error. Never fall back to a default tenant, workspace, provider or model.
4. **No content in logs.** Logs and metrics never contain message text, conclusions or names; log ids, counts and timings only.
5. **Nothing at rest on the host.** The database is the only place memory is stored. The instance writes no memory content to its host's disk: no Redis persistence, temporary files only in memory-backed storage (tmpfs), queue entries carry ids not content, and caches live in memory only. A snapshot of the host's disk must contain no memory content.
6. **Tests prove each goal.** Every acceptance item has an automated test that would fail if the behavior regressed. No skipped tests, no tests that assert on mocks the code no longer calls. The upstream test suite stays green.
7. **Commits and PRs.** One branch and one pull request per milestone (`m0-baseline`, `m1-single-tenant`, ...). Plain commit messages, no co-author, Claude or Cursor trailer of any kind. Docs in plain English, no em dashes.
8. **Progress log.** Keep `PROGRESS.md`: per milestone, what was built, decisions taken, deviations from this file and why, test counts, and open questions.

## How you work without waiting on Jose

- After finishing a milestone: run the full test suite and the milestone's acceptance tests, complete the self-review checklist below in the PR description, wait for CI to pass, then **merge it yourself** with `gh pr merge --merge --delete-branch` and start the next milestone.
- **Two milestones stop for Jose's review before merge: M1 and M2.** Open the PR, write `.review/M<n>-summary.md` (what changed, where isolation is enforced, the tests that prove it, file:line for each), then stop and report. Do not start the next milestone until Jose says the PR is merged.
- **When you hit an unknown in upstream code:** read the code, document what you found in `PROGRESS.md`, choose the least invasive option that meets the goal, and continue. Stop and ask only if a goal is impossible as written, if you need an account or secret, or if two goals conflict.

**Self-review checklist (in every PR description):** every acceptance item mapped to the test that proves it; upstream suite green; `FORK_CHANGES.md` updated; no secrets or content in logs; fail-closed paths tested; `PROGRESS.md` updated.

---

## M0. Baseline and map

**Goal:** a pinned, running, understood fork.

- Pin to the latest upstream release tag (record it in `PROGRESS.md`); keep an `upstream` remote.
- `docker compose up` starts the API, the deriver worker, Postgres with pgvector, and Redis locally; the upstream test suite passes; GitHub Actions runs it on every PR.
- Write `docs/MAP.md`: where the database engine and sessions are created, how auth and tokens work, how workspaces, peers, sessions, scopes, conclusions, representations, summaries and dreams are stored and queried, how the deriver queue works, where every LLM call is made, and where every prompt lives. Cite file:line.

## M1. Single-tenant instances (SECURITY: stop for review)

**Goal:** each running instance (one API plus its deriver workers) serves exactly one tenant, using that tenant's own Postgres database, and refuses everything else. Sentica runs one instance per tenant as a container beside that tenant's agent, on the tenant's own VM, under a separate Linux user. The instance never holds a database password: it reaches the database only through a proxy in Sentica's control plane, authenticated by a short-lived TLS client certificate.

- An instance is configured with exactly one `SENTICA_TENANT_ID`, one proxy address, one client certificate and key (paths in memory-backed storage), the proxy's CA, and one token verification key. Missing any of them: the instance refuses to start. No database password appears in configuration, environment, code or logs.
- Database connections use TLS with full verification (`verify-full`) and the client certificate. Certificates rotate with overlap (valid 48 hours, replaced every 24): the instance picks up a new certificate file without restarting, uses it for new connections, and retires pooled connections after at most 1 hour so old certificates drain naturally.
- Revocation: when the proxy drops the instance's connections, in-flight transactions roll back with no partial writes; API calls return a clear "memory unavailable" error (never a hang); deriver and dream tasks back off and resume idempotently once connections succeed again.
- Every request's signed token must carry `tenant_id` equal to the instance's tenant; every deriver task must carry the same tenant. Missing, different or forged tenant: refused and counted.
- Honcho's tables live in a dedicated Postgres schema (`memory`) inside the tenant database, owned by a role that can reach only that schema. `migrate` runs the schema migrations for the configured tenant.
- Redis runs inside the instance's own container set, memory-only (persistence off), reachable only by that instance.
- Admin (all-access) tokens are disabled when `SENTICA_MODE=production`.
- In production mode, tokens are verified with a configured public key (Ed25519 or RS256); only the issuer (Sentica's memory broker) holds the private key. Tokens carry an issuer, an expiry of at most 5 minutes, and a request id; expired, wrongly signed or wrong-issuer tokens are refused.
- The API listens only on the private interface Sentica's broker uses; document that it must never be exposed to the internet or to the agent container.
- For local testing, run PgBouncer (or an equivalent proxy) with certificate authentication in Docker Compose, with a local test CA. Never use real certificates.
- **Acceptance tests (two instances, A and B, each with its own database and certificate):** a token for tenant B is refused by instance A on every endpoint; a deriver task tagged for B is refused by A's worker; A's certificate cannot reach B's database through the proxy; A's database role cannot read anything outside the `memory` schema; an instance with a missing tenant, certificate or key refuses to start; missing, expired, forged or wrong-issuer tokens are refused; a certificate swap during continuous traffic causes zero failed requests; a revocation during a write leaves no partial rows and the API reports memory unavailable; after the instance stops, nothing with memory content remains on the host's disk.

## M2. Workspaces as security partitions (SECURITY: stop for review)

**Goal:** Sentica maps each access partition (general, each role, each restricted role, each user's personal space) to its own workspace. Nothing derived in one workspace may ever appear in another.

- Tokens carry the tenant and an explicit **set** of allowed workspaces. Every endpoint checks membership; a workspace not in the set is refused, not silently skipped.
- Search, context, chat (dialectic), representation and card endpoints accept a set of workspaces, query each, and merge the results. Every returned item carries its workspace id.
- Representations, peer cards, conclusions, summaries, dialectic answers and dreams are computed from one workspace only. A peer that exists in two workspaces has two separate representations.
- **Acceptance tests:** a fact written only to workspace R never appears, directly or paraphrased, in any response to a token without R. That includes search, context, chat answers, representations, cards, summaries and dream output. Include an adversarial chat test that asks for the content of R in ten different ways: zero leaks. Removing a workspace from a token takes effect on the next request.

## M3. All model calls through one gateway

**Goal:** every model call (deriver, summary, dialectic, dream, embeddings) goes to one OpenAI-compatible base URL, so Sentica can route it to US-only inference and log it.

- The base URL and model name are configured per function. With gateway mode on, no provider SDK is called directly.
- Every call sends metadata headers: tenant, workspace, purpose (`deriver`, `summary`, `dialectic`, `dream`, `embed`) and a request id.
- Per-tenant token counters per purpose, with an optional per-tenant daily budget. When the budget is exhausted, background work pauses for that tenant and the API returns a clear error; nothing switches to another model.
- Retries with backoff; a failed call never falls back to another provider.
- **Acceptance tests (against a mock gateway server):** every function calls the gateway with the correct headers; no direct provider calls when gateway mode is on; a budget stop works; a gateway outage pauses work without data loss.

## M4. Safe inference and provenance

**Goal:** memory never invents sensitive facts, and every conclusion can be traced to its sources.

- Edit the reasoning, summary and dream prompts so the model never infers or stores: health conditions or medical reasons, sexual orientation or activity, religion, political views, immigration status, criminal history, or an individual's personal finances or pay. Leave or absence is stored as dates only.
- A post-filter checks every new conclusion against those categories (rules first, an optional model check second) and drops violations, counted per tenant. The "asked about low-sugar meals plus insulin pumps, so likely diabetic" pattern is a required test fixture.
- Every conclusion stores the ids of the messages it came from, and the API returns them.
- Messages carry a `source_trust` value (`owner`, `staff`, `external`, `system`). Conclusions derived only from `external` content are marked so, and are never returned in a field Sentica treats as an instruction.
- **Acceptance tests:** each banned category has fixtures that must produce zero stored conclusions; provenance resolves for every conclusion; external-only conclusions are flagged.

## M5. Freshness: stale flags, superseding, fading, deletion

**Goal:** stale or replaced facts stop steering answers.

- Conclusions can carry a `subject_ref` (for example `customer:123`, `job:J-2291`, `staff:45`). `POST /subjects/{ref}/changed` marks every linked conclusion stale.
- On write, compare the new conclusion with the most similar existing ones for the same peer in the same workspace and classify it: new, update or contradiction. An update or a contradiction marks the old conclusion `superseded_by` the new one; nothing is hard-deleted by this step.
- Retrieval excludes superseded conclusions and down-ranks stale ones.
- Usage counters (retrieved, used). A scheduled job decays the weight of unused conclusions (for example halving every 30 days, configurable) and soft-deletes below a floor; soft-deleted items are purged after a configurable retention.
- A deletion API removes everything for a peer or a `subject_ref` (hard delete, including embeddings and derived summaries) and writes a deletion record with counts and no content.
- **Acceptance tests:** a changed subject flags its conclusions; an update supersedes; superseded items never appear in retrieval; decay and pruning follow the schedule; deletion leaves nothing searchable.

## M6. Retrieval quality and preloading

**Goal:** what is stored is found, fast enough for a phone call.

- Hybrid retrieval: keyword plus vector, with filters for workspace set, peer, `subject_ref` and recency, and a merged ranking.
- A `context` endpoint returns a bounded package for a peer (token budget parameter): card, top conclusions, recent summary, open stale flags. Sentica calls it from caller ID before a call is answered.
- **Targets on local hardware with 10,000 conclusions in a tenant:** context p95 under 300 ms warm; report cold and warm separately.
- **Acceptance tests:** a query using different words than the stored fact still finds it (synonym fixtures); filters are honored; latency is measured in CI and reported.

## M7. Dreaming, inside each partition

**Goal:** daily and weekly synthesis that finds patterns without crossing partitions.

- Dreams run per workspace only, on a configurable schedule: a daily digest, then a weekly pass that reads the daily digests (not the raw messages).
- Every insight links to the conclusions and digests it came from, carries its workspace, and passes the M4 filter.
- Cost per tenant per run is counted (M3 counters).
- Synthesis is on by default for every workspace, including restricted ones, and can be switched off per workspace through the API (Sentica exposes this to the owner per role). Switching it off stops new dreams for that workspace without deleting earlier ones.
- **Acceptance tests:** a weekly insight cites daily digests; no dream reads from or writes to another workspace; banned inferences are blocked; a workspace with synthesis off produces no new dreams.

## M8. Evaluation harness

**Goal:** proof that memory works and stays sealed.

- Retrieval-only evaluation on seeded synthetic businesses, modeled on LongMemEval's categories: information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention (correctly finding nothing). Scored deterministically, without a model judging.
- An isolation red-team suite across tenants and workspaces. It must report zero leaks.
- Reports recall at 5 and 10 per category, plus cold and warm latency. After the first full run, record a baseline in `eval/BASELINE.md`. CI fails on any isolation leak, or on a drop of more than 2 points in any category.

## M9. Packaging

**Goal:** ready for Sentica to deploy.

- Production Docker images for the API and the deriver; health and readiness endpoints; Prometheus metrics (requests, latency, queue depth, model tokens by purpose, filtered conclusions, decay and prune counts).
- The instance runs beside Sentica's agent container on the tenant's VM, so images run hardened: a dedicated non-root Linux user distinct from the agent's, a read-only root filesystem with tmpfs for temporary files and certificates, all Linux capabilities dropped, a seccomp profile, no volumes shared with any other container, and verified to run under the gVisor runtime (`runsc`). Document the exact run flags in `docs/DEPLOY.md`.
- Network: outbound only to the database proxy and the model gateway; inbound only from the broker's connection on the private interface; no route from the agent container to the instance or to the database proxy. Document the rules Sentica must apply.
- Measure and record in `docs/DEPLOY.md`: memory and CPU for the instance at idle and under a load test of 1,000 messages per hour, so Sentica can size tenant VMs.
- `docs/CONFIG.md` (every setting), `SECURITY.md` (the isolation model and how to report a vulnerability), `docs/UPSTREAM_SYNC.md` (how to merge a new upstream release with the fork changes).
- Tag `v0.1.0-sentica` and write release notes.

## Definition of done

M0 to M9 merged; every acceptance test green in CI; the isolation red-team at zero leaks; `FORK_CHANGES.md`, `PROGRESS.md`, `docs/MAP.md` and `eval/BASELINE.md` current; tag pushed.
