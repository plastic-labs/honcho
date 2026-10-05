# Honcho sandbox

A local Honcho you can wipe and rebuild in seconds, so harness testing stops depending on
whatever state your laptop happens to be in.

It comes up **already seeded** — a known workspace, peers, session, messages, conclusions the
deriver has actually produced, and committed conclusions at every reasoning level — and `reset`
returns it to that exact state without re-deriving.

```bash
sandbox/sandbox.sh up       # start + seed (first run also builds/pulls)
sandbox/sandbox.sh reset    # back to the seeded state, under a second
sandbox/sandbox.sh status   # what's running, which templates exist
sandbox/sandbox.sh down     # stop and delete the volumes
```

The API lands on <http://127.0.0.1:18000>. Postgres is on 15432 and Redis on 16379 — all
deliberately off the defaults, so the sandbox coexists with a normal local stack instead of
fighting it for ports and volumes.

## Provider modes

The deriver calls a model provider, which is what made a sandbox neither free nor deterministic.
Both ways of resolving that ship here; `mock` is the default.

|                       | `mock` (default)                                                 | `real`                                               |
| --------------------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| **Determinism**       | Same conclusions every run                                         | Non-deterministic                                     |
| **Cost**              | Zero, and no network egress                                        | Real spend                                            |
| **Seed time**         | ~15s                                                               | ~6 min                                                |
| **Reset time**        | 0.86s                                                              | 1.29s                                                 |
| **Vector recall**     | **Untestable** — see below                                         | Testable                                              |
| **Derived**           | 4 conclusions, all `explicit`, synthetic text                      | 22 conclusions — 16 `explicit`, 4 `deductive`, 2 `inductive` |
| **Seeded**            | 7 — 4 `explicit`, 2 `deductive`, 1 `inductive`, committed text      | the same 7, identical bytes                          |
| **Use it for**        | CI, harness smoke tests, anything that needs a repeatable answer   | Recall quality, validating against real model output  |

The derived counts are from the committed fixture, measured on both paths. They are the sharpest
illustration of the difference: mock gives you four copies of `[mock] mock dummy placeholder …`,
while real mode reasons its way from "my cat Marzipan is named after the pylon's colour" to the
deductive conclusion *"alice's cat is named after a feature of alice's pedestrian bridge project."*
Anything asserting on a model's *own* inference needs real mode.

The seeded row is why you may not need to. Seeded conclusions are written straight into the
collection rather than derived, so their text and level are identical in both modes — see
[Seeded conclusions](#seeded-conclusions).

```bash
sandbox/sandbox.sh up                    # mock
sandbox/sandbox.sh up --provider real    # real
```

Each mode keeps its own seeded template, so both can exist at once and switching between them
costs nothing — about 10 seconds, and no re-derivation, because `up` restores that mode's existing
template instead of reseeding.

### Two things that will otherwise cost you a day

**Mock embeddings carry no semantic similarity.** They are hash-derived, so two paraphrases are
exactly as far apart as two unrelated strings. Any recall assertion built on mock mode must use
lexical or full-text search. A vector-ranking assertion will fail there for reasons that have
nothing to do with the code you are testing — that is what `real` mode is for.

**Mock *derived* conclusions are synthetic.** The text comes from the request, not from the meaning
of your messages, and the level is always `explicit` (the Dreamer's specialists write via tool
calls, which the mock deliberately never emits). Assert that derived conclusions *exist*; do not
assert on what they say or on the level mix, or your test will pass in one mode and fail in the
other. Assert against the seeded conclusions instead — that is what they are for.

### Real mode

```bash
cp sandbox/real.env.example sandbox/real.env   # then add a key
sandbox/sandbox.sh up --provider real
```

Credentials come from that one gitignored file, never from your ambient shell environment, so a
real-mode run has a single auditable input and the sandbox still doesn't inherit machine state.
`sandbox.sh` refuses to start if the file is missing or still carries the placeholder key — a
real-mode stack with no key boots perfectly well and then derives nothing, which is indistinguishable
from "the deriver found nothing".

Real mode gets an unexpected benefit from the reset design: derivation is slow and costs money, and
the snapshot means you pay for it **once per seed** and reset for free after that. Measured: 361s to
seed, 1.29s to reset back to that exact state with no provider calls at all.

## How reset is fast

Seeding runs the deriver, and a naive reset would have to run it again. Instead, `seed` snapshots
the finished database as a Postgres template:

```
CREATE DATABASE honcho_sandbox_seeded_mock TEMPLATE honcho_sandbox
```

and `reset` drops the live database and re-creates it from that template, then flushes Redis. No
migrations, no re-derivation, no LLM calls — **measured at 0.86s**, and byte-identical every time.

Nothing is stopped or restarted. `DROP DATABASE ... WITH (FORCE)` evicts the api and deriver
connection pools, and both reconnect by themselves — the api on its next request, the deriver on its
next poll a quarter-second later. Each logs one `OperationalError` as its in-flight connection dies;
that is expected, and it is the price of not paying ~9 seconds of container restart on every reset.

The cost of a snapshot is that it can go stale. Three guards:

- **Per-mode template names**, so a mock-seeded template can never be restored into a real-mode run.
- **A fingerprint recorded inside each template** — the Alembic revision it was seeded at, a hash of
  `fixture.json` + `seed.py`, and the provider mode. `reset` compares and refuses with a "reseed"
  message rather than silently restoring a state that predates a migration.
- **A provider check against the running stack.** Neither `seed` nor `reset` recreates containers, so
  `--provider` on either cannot change what the stack actually talks to — only what gets recorded.
  `up` writes the provider it created the stack with to `sandbox/.state.env`, and `seed`/`reset`
  refuse a mismatch and tell you to run `up --provider ...` first. Without it,
  `seed --provider mock` against a running real stack would spend money on non-deterministic
  conclusions and label them `mock`, and every later `reset` would restore those as the
  deterministic baseline.

If you hit the staleness refusal: `sandbox/sandbox.sh seed`.

## Seeded conclusions

`fixture.json` can carry pre-made conclusions per peer, at any of the three reasoning levels.
All three keys are optional; with none of them the sandbox behaves exactly as it would without
this feature.

```json
{
  "id": "alice",
  "observe_me": true,
  "observe_others": false,

  "explicit": [
    "alice is a structural engineer based in Rotterdam",
    "alice has a cat called Marzipan, named after the colour of the bridge's pylon"
  ],
  "deductive": [
    { "content": "alice named her cat after a feature of the bridge she is designing",
      "premises": [0, 1] }
  ],
  "inductive": ["alice optimises for low lifetime maintenance cost"]
}
```

**The peer carrying the keys is the one being observed.** Conclusions are keyed by an
`(observer, observed)` pair, and the observer is inferred: every fixture peer with
`observe_others` set, excluding the observed itself. For the standard harness shape that is
exactly `assistant -> alice` — the same pair the dream uses and the one a harness reads from. A
peer nobody else observes is an **error**, not a silent no-op; add an explicit `"observer"` to
that peer to override the inference, which is also how you reach self-representation
(`alice -> alice`).

**Items are a bare string, or `{content, premises}`.** Each premise is an index into the same
peer's `explicit` list. Premises are optional, but a derived conclusion without them is not
something the Dreamer would ever write, so the committed fixture always supplies them.

**Contents must be unique.** Honcho collapses a conclusion whose content matches something
already stored — case- and whitespace-insensitively — and that cannot be switched off. The
seeder asserts exact counts, so a near-duplicate fails the seed rather than quietly vanishing.

### What premises buy you, and what they don't

The premise indices become real `Document.source_ids` pointing at the actual rows from the
`explicit` pass, so the reasoning tree genuinely traverses in both directions. Two caveats on
where that is visible:

- **Premise text renders in the representation.** `peer.representation(target=…)` prints each
  premise indented under its conclusion, which is the practical payoff and what `seed.py verify`
  asserts.
- **Premise *links* are not on the API.** The conclusion response carries `level` but not
  `source_ids`, and `get_reasoning_chain` is a Dialectic tool rather than a route. So a test can
  only reach the links through the Dialectic, or through SQL. Because of that, a broken
  reasoning tree is invisible from outside — so the seeder checks its own links before the
  snapshot is taken, resolving every premise id and confirming each is reachable from its
  children. A dangling or mis-filed premise fails the seed.

### Why this is not done over the API

The public create-conclusions endpoint always writes `level="explicit"` with no premises, and
neither the schema nor the SDKs have a field for either. The columns exist; only an in-process
caller can set them. So `sandbox.sh` runs `inject_conclusions.py` **inside the api container**,
which already is Honcho's venv with the api's settings and a live embedding client — the script
goes in on stdin and the fixture through the environment, so nothing is mounted and nothing is
left behind.

The cost is that the injector calls Honcho *internals*, which carry no stability contract, out
of the image pinned in `image.env`. It checks the signatures it depends on before writing and
fails with a "bump `image.env` and update this script together" message, rather than half-seeding
a database. Widening the public API to accept a level was the alternative and was deliberately
rejected: it would let any client assert a conclusion is `deductive` with premises it invented.

## What's in the fixture

`fixture.json` — two peers with **explicitly stated** observation topology (`alice` is observed and
does not observe; `assistant` observes and is not observed), one session, six messages carrying
distinctive lexically-searchable facts, and one scheduled cross-peer dream.

Topology is written down rather than defaulted because it is the thing that silently breaks, and
`seed.py` re-reads it back from the server to confirm it took.

Edit `fixture.json`, then `sandbox/sandbox.sh seed` to rebuild the template. `seed` always starts
from an empty, freshly migrated database, so it means the same thing whatever state you run it
from — running it twice gives the same result as running it once.

Dreams never fire on their own here — the document threshold is 50 and the minimum gap between
dreams is 8 hours — so the seed schedules one directly.

## Layout

| File                | Purpose                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `sandbox.sh`        | The entry point. Owns Docker and Postgres.                              |
| `compose.yml`       | Provider-agnostic base. **Not a working stack alone.**                  |
| `compose.mock.yml`  | Provider overlay: adds the mock-provider service and wiring.            |
| `compose.real.yml`  | Provider overlay: reads `real.env`, no mock service.                    |
| `image.env`         | The pinned image digest. One line, bump deliberately.                   |
| `init.sql`          | Creates the `honcho_sandbox` database.                                  |
| `fixture.json`      | The seeded conversation.                                                |
| `seed.py`           | Populates and verifies. Talks only to the API.                          |
| `inject_conclusions.py` | Writes the seeded conclusions. Runs inside the api container.       |
| `real.env.example`  | Template for real-mode credentials.                                     |

Exactly one provider overlay is always composed on top of the base, so the choice is visible in the
`-f` list rather than buried in a default. That is also why `compose.yml` alone does not run: the
`depends_on` edge to `mock-provider` lives in the mock overlay, because Compose merges `depends_on`
additively and an override file cannot remove one.

## Configuration

The sandbox is configured **only** by what Compose injects. `PYTHON_DOTENV_DISABLED` and
`HONCHO_CONFIG_TOML_DISABLED` are both set, because `src/config.py` calls `load_dotenv(override=True)`
at import and the Dockerfile's `COPY config.toml* /app/` bakes a local `config.toml` into any
locally built image. Without those two flags your machine's leftovers would quietly win.

`compose.yml` also pins the deriver's scheduling so derivation happens *now*. On stock settings a
sandbox seeded with a handful of messages produces zero conclusions and fails silently: work units
aren't claimed until a batch reaches 512 tokens or 30 minutes pass, and startup jitter delays the
first poll by up to 30 seconds. Those are turned off here — see the comments in `compose.yml`.

## Building from the working tree

```bash
sandbox/sandbox.sh up --build
```

Builds the repo instead of pulling the pinned digest, and points every service at the result so
they stay in sync. Use it when testing a change to Honcho itself. It trades reproducibility for
currency — you are running your tree, not the pinned bytes, which is right while iterating and
wrong when reproducing someone else's result.
