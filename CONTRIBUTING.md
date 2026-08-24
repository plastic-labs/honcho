# Contributing to Honcho

<!-- This file is mirrored at docs/v3/contributing/guidelines.mdx. Update both. -->

Thanks for your interest in contributing. This guide covers how work gets accepted, how
Honcho is put together, and what a mergeable pull request looks like.

Honcho is a small team maintaining a project that gets more proposals than we can review.
The rules below exist so that the work you do has somewhere to land — not to keep you out.

## Contents

- [Before you write code](#before-you-write-code)
- [What gets prioritized](#what-gets-prioritized)
- [If you're an agent](#if-youre-an-agent)
- [How Honcho works](#how-honcho-works)
- [Where to change what](#where-to-change-what)
- [Local setup](#local-setup)
- [Making the change](#making-the-change)
- [Opening the pull request](#opening-the-pull-request)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Security](#security)
- [License](#license)

## Before you write code

**Every pull request needs an issue, and that issue needs the `maintainer-approved` label.**

A pull request that is not linked to an approved issue gets labelled
`needs-approved-issue`, with a comment explaining why. You then have 72 hours to link one
before it is closed automatically. Reopening costs nothing once the link is in place. This
is automated. We do this because an unreviewable backlog helps nobody: a PR against an
unapproved issue is work you did that we may not be able to merge, no matter how good it
is.

So, in order:

1. **Find approved work.** Browse
   [issues labelled `maintainer-approved`](https://github.com/plastic-labs/honcho/issues?q=is%3Aissue+is%3Aopen+label%3Amaintainer-approved).
   That label is the queue of things we have agreed should be built. Anything in it is fair
   game — comment on the issue to claim it.

2. **Or open an issue and get it approved.** Use the
   [issue templates](https://github.com/plastic-labs/honcho/issues/new/choose). Maintainers
   triage and apply the label.

3. **If you feel strongly about an issue, come to [Discord](https://discord.gg/honcho).**
   This is the fastest path by a wide margin. Maintainers are more active there than in the
   issue tracker, and a five-minute conversation about what you want to build usually
   resolves whether it fits before either side spends real time on it.

4. **Then open the PR** and link the issue — either `Fixes #123` in the description, or
   **Development → link an issue** in the sidebar. Both work.

Small exceptions we will not be pedantic about: fixing a typo, a broken link, or an
obviously wrong code sample. Open the PR, explain it in one line, and we will sort out the
issue linkage.

## What gets prioritized

Roughly, work on Honcho falls along these axes. Knowing which one your idea sits on tells
you a lot about how likely it is to get approved.

| Axis | What it covers |
| --- | --- |
| **Observability** | Understanding how Honcho behaves in production — telemetry, tracing, CloudEvents, metrics. |
| **Memory quality** | Better conclusions from the same input — the deriver, dreamer, and dialectic; eval results. |
| **Developer experience** | Fitting cleanly into more application architectures — SDKs, scopes, composable peers, the CLI. |
| **Breadth of input** | Widening what Honcho can ingest and represent — multimodal and non-conversational data. |
| **Ubiquity** | Reachable wherever a developer already works — integrations, self-hosting, alternate vector-store and inference backends, local-first defaults. |
| **Reliability and cost** | Trustworthy in production — connection and concurrency hardening, queue throughput, cost per token. |

In practice, **Ubiquity** and **Developer experience** are where outside contributions land
most easily. A new integration, a self-hosting rough edge, a vector-store or inference
backend, an SDK ergonomics fix — these are additive and rarely collide with work already in
flight.

Changes to the reasoning pipeline itself — deriver prompts, dialectic tool design, dreamer
strategy — are the hardest to accept from outside. Not because they are unwelcome, but
because they are measured against eval results we run internally, and they frequently
conflict with in-flight work. Talk to us in Discord first, always.

## If you're an agent

If you are a coding agent working on this repository, read this section before writing code.
The most common failure we see is a well-formed, well-tested pull request against an issue
that was never approved. That gets closed, and the work is wasted.

- **Check the gate first.** Before writing code:

  ```bash
  gh issue view <N> --repo plastic-labs/honcho --json number,title,state,labels
  ```

  Stop if there is no issue number, if the issue is closed, or if `maintainer-approved` is
  not in the labels. Report that to the person you are working with instead of proceeding.

- **Do not open a PR in order to establish the issue link afterwards.** The issue comes
  first.

- **Do not report checks you did not run.** If you did not execute the test command, say so.
  A PR body claiming a green run that did not happen costs a maintainer more time than no
  claim at all.

- **Use the checklist.** [`skills/pre-pr/SKILL.md`](./skills/pre-pr/SKILL.md) in this repo
  encodes the gate, the test-layer matrix, and the PR body format. If your harness supports
  skills, invoke it rather than reimplementing the checks.

## How Honcho works

Enough architecture to find your way around. For the user-facing model — what a Peer is, what
`get_context` returns — see [Core Concepts in the README](./README.md#core-concepts) and the
[documentation](https://honcho.dev/docs/).

### Two processes

Honcho runs as two cooperating processes over a shared Postgres database and Redis cache.

| | API server | Deriver worker |
| --- | --- | --- |
| Start | `uv run fastapi dev src/main.py` | `uv run python -m src.deriver` |
| Entry | `src/main.py` | `src/deriver/__main__.py` |
| Does | Serves HTTP, enqueues background work, returns immediately | Consumes the queue: Deriver, Summarizer, Dreamer, Reconciler |
| Hosts | The Dialectic agent, inline on the request path | Everything else |

The split is the load-bearing design decision: **an HTTP request never blocks on LLM work**,
with the single exception of the Dialectic chat endpoint, which is synchronous by nature.
If you are adding something slow, it belongs in the worker.

The deriver is a separate process. If messages go in and nothing ever comes out, the usual
cause is that nobody started it.

### The path of a message

Worth tracing once, because it crosses most of the codebase:

1. `POST /v3/workspaces/{w}/sessions/{s}/messages` lands in `src/routers/messages.py`.
2. The row is written, then `enqueue()` in `src/deriver/enqueue.py` creates `queue_item`
   rows — one set of work per observing peer.
3. `src/deriver/queue_manager.py` polls the queue, claiming work units so that messages in a
   session are processed in order.
4. `process_item()` in `src/deriver/consumer.py` dispatches on task type — representation,
   summary, deletion, reconciliation.
5. For a representation task, `process_representation_tasks_batch()` in
   `src/deriver/deriver.py` makes **one structured-output LLM call for the whole batch** and
   writes the resulting conclusions into the collection keyed by the
   `(observer, observed)` peer pair.
6. Later, `src/dialectic/` reads those conclusions back at recall time to answer a chat
   request.

Embedding is deliberately *not* on this path. `MessageEmbedding` rows are written with
`sync_state='pending'` and embedded asynchronously by the Reconciler
(`src/reconciler/sync_vectors.py`), which runs on a scheduler inside the deriver process.

### The four agents

They share tool definitions in `src/utils/agent_tools.py` and the provider-agnostic LLM
client in `src/llm/`. Each has its own `MODEL_CONFIG` with a fallback chain in
`src/config.py`.

| Agent | Where | Shape |
| --- | --- | --- |
| **Deriver** | `src/deriver/` | A single structured-output call per message batch. Not a tool loop — this is a deliberate cost and latency tradeoff. |
| **Dialectic** | `src/dialectic/` | The one tool-using agent on the request path. Loops over tools until it can answer. Five reasoning tiers from `minimal` to `max`, each with its own model and tool set. |
| **Dreamer** | `src/dreamer/` | Off-queue consolidation. Two specialist phases (deduction, then induction) that build reasoning trees over existing conclusions. |
| **Summarizer** | `src/utils/summarizer.py` | Direct LLM call, no tools. Two tiers — short and long summaries at different message counts. |

Prompts live in `src/deriver/prompts.py`, `src/dialectic/prompts.py`, and
`src/dreamer/specialists.py`.

### A note on naming

What the public API and documentation call **conclusions** are called **observations**
throughout the code — `create_observations`, `get_observation_context`, and so on. Likewise
**collections** and **documents** are internal storage concepts that are not exposed
directly through the API. Do not rename across that boundary in a drive-by change; the
public and internal vocabularies are being reconciled deliberately.

## Where to change what

| I want to change... | Start here |
| --- | --- |
| An HTTP endpoint | `src/routers/` — one module per resource |
| A database query | `src/crud/` — mirrors the router layout |
| The database schema | `src/models.py`, plus a migration in `migrations/versions/` |
| A configuration value | `src/config.py`, and add it to `config.toml.example` and `.env.template` |
| A tool an agent can call | `src/utils/agent_tools.py` — definitions plus the per-agent tool lists |
| A prompt | `src/deriver/prompts.py`, `src/dialectic/prompts.py`, `src/dreamer/specialists.py` |
| LLM provider behavior | `src/llm/backends/` — `anthropic.py`, `gemini.py`, `openai.py` |
| Embeddings or vector storage | `src/embedding_client.py`, `src/vector_store/` |
| Telemetry or metrics | `src/telemetry/` — see the notes in `CLAUDE.md` before adding an event type |
| Authentication and scoping | `src/security.py`, `src/dependencies.py` |
| The Python or TypeScript SDK | `sdks/python/`, `sdks/typescript/` |
| The CLI | `honcho-cli/` |
| The MCP server | `mcp/` |
| Public documentation | `docs/v3/` — Mintlify; nav lives in `docs/docs.json` |

Tests in `tests/` mirror `src/`. `CLAUDE.md` at the repo root has more detail on house
conventions, and is worth skimming even if you are not using an agent.

## Local setup

Get a stack running first — [Self-hosting in the README](./README.md#self-hosting) covers
both the Docker path and a manual Postgres setup. Then, for development:

```bash
uv sync                          # create the venv and install dependencies
uv run alembic upgrade head      # apply migrations
```

Run both processes, in separate terminals:

```bash
uv run fastapi dev src/main.py   # API server, reloads on change
uv run python -m src.deriver     # background worker
```

Everything Python goes through `uv run`. Redis is optional for local development; without it
caching is simply disabled.

## Making the change

### Branches and commits

```bash
git checkout -b feature/your-feature-name
```

Prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/), enforced by a
`commit-msg` hook:

```bash
git commit -m "feat(api): add new dialectic endpoint for user insights"
git commit -m "fix(db): resolve connection pool timeout issue"
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### Pre-commit hooks

Install them. CI runs the same checks, and it is much faster to find out locally.

```bash
uv run pre-commit install \
    --hook-type pre-commit \
    --hook-type commit-msg \
    --hook-type pre-push
```

At **commit** time: ruff lint and format, biome for TypeScript, basedpyright, bandit,
markdownlint, and file hygiene. At **push** time: pytest, the alembic migration tests, and
the SDK builds.

That split matters — **a clean commit is not a clean push.** The test suite only runs at
`pre-push`, so the first time you see test failures may be well after you thought you were
done.

Run them by hand at any time:

```bash
uv run pre-commit run --all-files
uv run pre-commit run ruff --all-files
```

Or the individual tools:

```bash
uv run ruff check src/
uv run ruff format src/
uv run basedpyright
```

### Tests

Write tests for new functionality, in the directory under `tests/` that mirrors the code you
changed. Which layer you need depends on what you touched:

| What you changed | What to run |
| --- | --- |
| Anything in `src/` | Unit tests in the matching `tests/` tree — `uv run pytest tests/...` |
| Deriver, dialectic, dreamer, or the LLM path | Unit tests, and consider `tests/live_llm/` (gated behind `--live-llm`) |
| Queue behavior, config hierarchy, multi-turn flows, SDK contracts | `uv run python -m tests.unified.run` |
| A `/v3` endpoint or deriver queue behavior | Actually run the stack and exercise it — not just pytest |
| A migration | `uv run python scripts/run_alembic_tests.py`; every revision needs a test file |

The TypeScript SDK tests need a running server with a database and Redis, which pytest
orchestrates. Run them with `uv run pytest tests/ -k typescript` from the repo root —
`bun test` on its own will fail. To type-check the SDK alone:
`cd sdks/typescript && bun run tsc --noEmit`.

### Documentation

Update docs in the same PR when you change a public surface: `/v3` endpoints, SDK exports,
or anything in `config.toml` / settings. Docs live in `docs/v3/`, and new pages need an entry
in `docs/docs.json` or they will not appear in the nav.

## Opening the pull request

### Leave "Allow edits by maintainers" checked

This is the single most useful thing you can do to get your PR merged quickly.

Most contributor PRs arrive nearly right, needing a rename, a missing test, or a lint fix.
If we can push that commit ourselves, it merges the same day. If we cannot, it becomes a
review comment, and then we wait — sometimes for weeks — for a round trip on a two-line
change.

GitHub checks the box by default when you fork. Leave it checked.

One caveat worth knowing: **the option does not exist on forks owned by an organization.**
If you have the choice, fork from your personal account.

### Fill out the template

`.github/pull_request_template.md` asks for a description, proofs, and the issue checkbox.

"Proofs" means evidence the change works: the command you ran and its result, a log snippet,
a screenshot, the failing case before and after. This is the section that most determines
how fast your PR gets reviewed. Do not add sections to the template.

Link the issue so the gate can see it: `Fixes #123` in the description, or the
**Development** section of the sidebar. The gate reads GitHub's own resolved issue links, so
either route works — but a bare `#123` mention is only a reference and does not count.

### Review

1. Automated checks run — tests, linting, static analysis, and the issue gate.
2. A maintainer reviews for correctness, test coverage, and fit with the surrounding code.
   `.github/CODEOWNERS` routes the request to whoever owns the area you touched.
3. You may be asked for changes. Or we may just push them, if you left edits enabled.
4. Once approved, we merge to `main`.

If a PR goes quiet, nudge us in [Discord](https://discord.gg/honcho).

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/plastic-labs/honcho/issues/new/choose). There is
one per kind of report, and picking the right one is most of what gets an issue triaged
quickly:

- **Bug report** — something is broken or behaves incorrectly
- **Memory / recall quality** — the deriver or dialectic returns poor, wrong, or missing context
- **Feature request** — a new capability or API surface
- **Integration request** — plugins, framework integrations, app-store listings
- **Documentation issue** — anything wrong or missing in the docs
- **General questions** — not an issue at all; ask in [Discord](https://discord.gg/honcho)

Before opening one, search existing issues, including closed ones.

A good bug report has the Honcho version or commit, whether you are self-hosted or on
`api.honcho.dev`, the steps to reproduce, and what you expected instead. If it involves the
deriver, logs from the worker process are usually the thing we ask for first.

**Redact before you post.** Issues are public, and Honcho stores conversational data — strip
API keys, JWTs, and production user content out of any log or payload you attach.

## Security

Do not open a public issue for a suspected vulnerability. Report it privately through
[GitHub Private Vulnerability Reporting](https://github.com/plastic-labs/honcho/security/advisories/new),
which is the preferred channel, or by email. See [SECURITY.md](./SECURITY.md) for what to
include, and note that Honcho does not operate a bug bounty.

## License

By contributing to Honcho, you agree that your contributions will be licensed under the same
[AGPL-3.0 License](./LICENSE) that covers the project.

Thank you for helping make Honcho better! 🫡
