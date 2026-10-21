# DEV-2773 persisted-Redis regression evidence

Local run on 2026-10-21, Compose project `del2773_cache_1021`, mock provider only.
API ports 19100/19101; Postgres 16532; Redis 17479. No production or staging data.
Commands and scenario contracts are in [the sandbox README](README.md#erasure-slice-runner).

## Images

Built from unchanged application sources; this regression modifies only the harness/docs.

| Role | Source | Local image |
| --- | --- | --- |
| Prototype | `07924c696f5aea26a804f8d3ff1c4efe731e0bcc` | `del2773/honcho:prototype-07924c69` |
| **UNSAFE negative control** | v3.4.0, `1781bfb23b5c976ecd84324376a0527a7a5b6a36` | `del2773/honcho:unsafe-v3.4.0` |

Actual container image IDs, confirmed on both replicas in each phase:

```text
prototype: sha256:e861827d408018ff91b2d8b664926cdfc2472e64a10843333af2b1c668926cbc
v3.4.0:    sha256:3b4724d351affb5b91a9415ecf5194690f96e03699cea0a66ec24e90279d08d3
```

## Results

| Scenario / phase | Expected | Actual |
| --- | --- | --- |
| Original: prototype after delete | Clean | 38/38 checks |
| Original: prototype after release | Clean | 39/39 checks |
| Original: v3.4.0 starts while Redis is down | Clean via this sequence only | 39/39 checks; startup falls back to local memory |
| Persisted: prototype after delete + release | Clean | 77/77 checks |
| Persisted: v3.4.0 starts after Redis returns | **Expose stale cache** on both replicas | 39/39 expectations; exactly 12 expected exposures |
| Persisted: prototype on the same persisted dataset | Clean on both replicas | 39/39 checks; zero exposures |

**Original: 116/116 checks + 15/15 steps, 73 seconds.**
**Persisted regression: 155/155 checks + 49/49 steps, 144 seconds.**
The 12 expected unsafe exposures are three observer cards, short summary, long summary,
and session context, each on two replicas. Every exposure required HTTP 200. These are
successful negative-control assertions, not ignored failures. All authoritative DB checks
remained clean. Both workers were stopped during the two-image persisted-cache comparison;
the original scenario separately restarted both workers on the old image.

The same four target entries survived all six cache snapshots (before outage, after restart,
and before/after each image's reads), with TTL `-1` and unchanged SHA-256 and byte counts:

| Target key suffix (prefix `{honcho}:v2:workspace:slice-ws:`) | Bytes | Canary occurrences | SHA-256 |
| --- | --- | --- | --- |
| `peer:alice` | 334 | 1 | `30db9f43cd2af0cf78808de573cca5ac585c457e7f81fe7be1d6a04ebc36e422` |
| `peer:bob` | 338 | 1 | `cd6278f769d01bd2002843e4d7415c8651ef01db9400ef50100de7e35d49d444` |
| `peer:carol` | 340 | 1 | `62e05184bbef33e41147f60860c20219dd56bc25298876750c941d93ab1128a5` |
| `session:slice-session` | 631 | 2 | `0eff1daf87466708148c3f3d5f44b13bc0ce8df3d17eecadf1f199f709774416` |

Redis's process ID changed while its container ID stayed constant; AOF remained enabled.
Both images had successful Redis startup logs **and** live `CLIENT LIST` connections from
each API container. No target-key deletion, rewrite, cache flush or expiry occurred during
comparison. Hashes are fixture-specific and will differ on subsequent runs.

## Logs and tests

The run used `--results-dir` pointing at the local coordination directory's `2773/results/`.
Evidence filenames:

- `persisted-cache-final.json` — complete checks, steps, image identities and six cache snapshots.
- `persisted-cache-final.{after-delete,after-release,unsafe-negative-control,prototype-persisted-cache}.compose.log`.
- `original-final.json` and `original-final.{after-delete,after-release,after-swap}.compose.log`.
- `final-verification.log` — full runner output, lint and targeted tests.

Targeted runner tests: **56 passed**, ruff lint/format clean. They cover missing/changed/expiring
entries, AOF/restart proof, Redis-client/fallback/worker-state guards, HTTP-error rejection,
expected exposure versus prototype cleanliness, original ordering, and failed-step verdicts.
The standalone test environment emitted two harmless pytest-asyncio configuration warnings
because it installed only the harness's dependencies, not the application test plugins.

An earlier attempt correctly reported `blocked` when old-image API-2 stalled during import
with an empty tokenizer cache, after a large wall-clock/suspend gap. It did not reach the
prototype comparison and is not counted as a pass. Its `persisted-cache.*` evidence is retained.
The complete fresh rerun above passed without a production-code change.

## Boundary

This is the hard-delete sizing prototype, not production lifecycle acceptance. Cards and
summaries are injected canary fixtures; the paused deriver and message embedding are real
mock-backed paths. No claim is made for production tombstones, full lineage, every producer
or primitive, external-vector recovery, durable purge, shared-tenant isolation, suppression,
or activation-to-expand rollback. **v3.4.0 remains unsafe and is not a supported rollback floor.**
Production work remains DEV-2777/DEV-2781; integrated verification remains DEV-3141.
