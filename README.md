<!-- markdownlint-disable MD033 -->
<div align="center">
  <a href="https://app.honcho.dev" target="_blank">
    <img src="assets/honcho.svg" alt="Honcho" width="400">
  </a>
</div>
<!-- markdownlint-enable MD033 -->

---

![Static Badge](https://img.shields.io/badge/Server-3.0.7-blue)
[![PyPI version](https://img.shields.io/pypi/v/honcho-ai.svg)](https://pypi.org/project/honcho-ai/)
[![NPM version](https://img.shields.io/npm/v/@honcho-ai/sdk.svg)](https://npmjs.org/package/@honcho-ai/sdk)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/honcho)

**Honcho is memory infrastructure for building stateful agents that understand changing people, agents, groups, projects, and ideas over time.**

Store messages and events, let Honcho reason in the background, then query peer representations, session context, search results, or natural-language insights from any model or framework. Use it managed at [api.honcho.dev](https://api.honcho.dev) or self-host the FastAPI server yourself.

Using Honcho as your memory system will earn your agents higher retention, more trust, and help you build data moats to out-compete incumbents.

> Honcho has defined the Pareto Frontier of Agent Memory. Watch the [video](https://x.com/honchodotdev/status/2002090546521911703?s=20), check out our [evals page](https://honcho.dev/evals/), and read the [blog post](https://blog.plasticlabs.ai/research/Benchmarking-Honcho) for more detail.

## Contents

- [Start Here](#start-here)
- [Why Honcho](#why-honcho)
- [The Honcho Loop](#the-honcho-loop)
- [Quickstart](#quickstart)
- [What Honcho Gives You](#what-honcho-gives-you)
- [Integrations](#integrations)
- [Core Concepts](#core-concepts)
- [Benchmarks & Evals](#benchmarks--evals)
- [Self-hosting](#self-hosting)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [SDKs](#sdks)
- [Learn More](#learn-more)
- [Contributing](#contributing)
- [License](#license)

The Honcho project is split between several repositories, with this one hosting the core service logic — implemented as a FastAPI server. Client SDKs for Python and TypeScript live in the [`sdks/`](./sdks) directory.

## Start Here

| I want to...                           | Path                                                       | Get started                   |
| -------------------------------------- | ---------------------------------------------------------- | ----------------------------- |
| Give my coding agent persistent memory | Claude Code, OpenCode, OpenClaw, Hermes, or any MCP client | [Integrations](#integrations) |
| Add memory to my product               | Python or TypeScript SDK                                   | [Quickstart](#quickstart)     |
| Self-host Honcho                       | Docker / local development                                 | [Self-hosting](#self-hosting) |

## Why Honcho

| Capability              | What it means                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------ |
| Reasoning-first memory  | Extracts conclusions from conversations and events, not just matching chunks.        |
| Peer-centric model      | Tracks users, agents, groups, projects, and ideas as entities that change over time. |
| Multi-peer perspective  | Models what one peer knows about another when configured.                            |
| Managed or self-hosted  | Use `api.honcho.dev` or run the FastAPI server yourself.                             |
| Agent-tool integrations | MCP, Claude Code, OpenCode, OpenClaw, Hermes, Cursor-compatible clients.             |

## The Honcho Loop

1. **Store** conversations, events, documents, or tool traces as messages on a session.
2. **Reason** — Honcho processes the queue in the background and updates peer representations.
3. **Query** — ask Honcho for context, search results, peer representations, or a natural-language answer.
4. **Inject** — drop the result into any LLM call or agent framework.

Concretely: workspaces hold peers, peers participate in sessions, messages live on sessions, and Honcho builds a per-peer representation that you query through the [Chat Endpoint](https://honcho.dev/docs/v3/documentation/features/chat) or directly.

## Quickstart

Get an API key at [app.honcho.dev](https://app.honcho.dev) — when you sign up you'll be prompted to join an organization, which gets its own dedicated Honcho instance and $100 free credits. Or [self-host](#self-hosting) and run against `http://localhost:8000`.

### Python

```bash
pip install honcho-ai
# or: uv add honcho-ai
# or: poetry add honcho-ai
```

```python
import os
from honcho import Honcho

# Managed service uses api.honcho.dev by default. For self-hosted, pass
# base_url="http://localhost:8000" or set HONCHO_URL.
honcho = Honcho(
    workspace_id="my-app-testing",
    api_key=os.environ["HONCHO_API_KEY"],
)

# 1. Store: peers and messages on a session
alice = honcho.peer("alice")
tutor = honcho.peer("tutor")
session = honcho.session("session-1")
session.add_messages([
    alice.message("Hey there — can you help me with my math homework?"),
    tutor.message("Absolutely. Send me your first problem!"),
])

# 2. Reason: happens asynchronously in the background.

# 3. Query: ask Honcho what it knows, or pull prompt-ready context.
answer = alice.chat("What learning styles does the user respond to best?")
context = session.context(summary=True, tokens=10_000)

# 4. Inject: hand the context to your model of choice.
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    messages=context.to_openai(assistant=tutor),
)
```

### TypeScript

```bash
npm install @honcho-ai/sdk
# or: bun add @honcho-ai/sdk
```

```typescript
import { Honcho } from "@honcho-ai/sdk";
import OpenAI from "openai";

const honcho = new Honcho({
  workspaceId: "my-app-testing",
  apiKey: process.env.HONCHO_API_KEY,
});

const alice = await honcho.peer("alice");
const tutor = await honcho.peer("tutor");
const session = await honcho.session("session-1");
await session.addMessages([
  alice.message("Hey there — can you help me with my math homework?"),
  tutor.message("Absolutely. Send me your first problem!"),
]);

const answer = await alice.chat(
  "What learning styles does the user respond to best?",
);
const context = await session.context({ summary: true, tokens: 10_000 });

const openai = new OpenAI();
const completion = await openai.chat.completions.create({
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  messages: context.toOpenAI({ assistant: tutor }),
});
```

> **Note:** background reasoning is asynchronous. Newly-added messages may take a moment to be reflected in chat/representation responses; for low-latency reads, use the [`representation`](https://honcho.dev/docs/v3/documentation/features/representation) endpoint.

## What Honcho Gives You

| Need                               | API                                                             |
| ---------------------------------- | --------------------------------------------------------------- |
| Save interaction history           | `session.add_messages(...)`                                     |
| Ask what Honcho knows about a peer | `peer.chat(...)`                                                |
| Get prompt-ready context           | `session.context(...).to_openai(...)` / `.to_anthropic(...)`    |
| Hybrid search (BM25 + vector)      | `peer.search(...)`, `session.search(...)`, `honcho.search(...)` |
| Low-latency static representations | `peer.representation(...)`, `session.representation(...)`       |
| Import documents                   | `session.upload_file(...)`                                      |
| Inspect background processing      | `honcho.queue_status(...)`                                      |

See the full [SDK Reference](https://honcho.dev/docs/v3/documentation/reference/sdk) and [API Reference](https://honcho.dev/docs/v3/api-reference/introduction).

## Integrations

### Claude Code

Two ways, depending on how deep you want to go:

**Plugin (richer integration — recommended for Claude Code users):**

```text
/plugin marketplace add plastic-labs/claude-honcho
/plugin install honcho@honcho
```

**Raw MCP (works in any MCP client — Cursor, Cline, Windsurf, etc.):**

```bash
claude mcp add honcho \
  --transport http \
  --url "https://mcp.honcho.dev" \
  --header "Authorization: Bearer hch-your-key-here" \
  --header "X-Honcho-User-Name: YourName"
```

Details: [Claude Code guide](https://honcho.dev/docs/v3/guides/integrations/claude-code) · [MCP guide](https://honcho.dev/docs/v3/guides/integrations/mcp).

### OpenCode

```bash
opencode plugin "@honcho-ai/opencode-honcho" --global
```

Details: [OpenCode guide](https://honcho.dev/docs/v3/guides/integrations/opencode).

### OpenClaw

```bash
openclaw plugins install @honcho-ai/openclaw-honcho
openclaw honcho setup
openclaw gateway --force
```

`openclaw honcho setup` prompts for your API key, writes the config, and optionally migrates legacy `MEMORY.md` / `USER.md` / `IDENTITY.md` files into Honcho (non-destructive — originals are never deleted). Details: [OpenClaw guide](https://honcho.dev/docs/v3/guides/integrations/openclaw).

### Hermes

```bash
hermes memory setup   # select "honcho", point at api.honcho.dev or your local server
```

Details: [Hermes guide](https://honcho.dev/docs/v3/guides/integrations/hermes).

### Add Honcho to your own codebase (agent skill)

For wiring the Honcho SDK into an existing application, install the integration skill — it explores your codebase, asks about integration preferences, generates the SDK setup, and verifies it works:

```bash
npx skills add plastic-labs/honcho
```

Then invoke `/honcho-integration` in Claude Code (or `/honcho-dev:integrate` via the plugin marketplace). Details: [agentic development guide](https://honcho.dev/docs/v3/documentation/introduction/vibecoding).

### Other MCP clients

The same `claude mcp add` form (or its client-specific equivalent) works in any MCP-compatible client. See [MCP guide](https://honcho.dev/docs/v3/guides/integrations/mcp).

## Core Concepts

Honcho organises everything around **peers** — humans and AI agents alike are first-class entities. The peer model enables:

- Multi-participant sessions with mixed human and AI agents
- Configurable observation settings (which peers observe which others)
- Flexible identity management for all participants
- Support for complex multi-agent interactions

Peers exchange messages within sessions; Honcho reasons over those messages to build a representation of each peer that you can query.

- **Workspace** (formerly App): top-level container; isolates data between use cases.
- **Peer** (formerly User): any participant — human user or AI agent.
- **Session**: a conversation context; many-to-many with peers.
- **Message**: an atomic data unit (peer-to-peer communication or ingested document chunk).

What you query out of Honcho:

- **Conclusions** — what Honcho has extracted about a peer (deductive and inductive). Exposed via the [conclusions API](https://honcho.dev/docs/v3/api-reference/introduction).
- **Representations** — static, low-latency snapshots of what Honcho knows about a peer (optionally session-scoped).
- **Peer Cards** — compact identity summaries.
- **Session context / summaries** — prompt-ready bundles for long-running conversations.

<!-- markdownlint-disable MD033 -->
<details>
<summary>Internal storage (Collections &amp; Documents)</summary>

Internally, Honcho stores peer-related observations in **collections** of vector-embedded **documents**. Collections are keyed by `(observer, observed)` peer pairs — the same mechanism powers self-representation (`observer == observed`) and cross-peer modelling (peer X's understanding of peer Y). These primitives are not exposed directly; the Conclusions API is the public surface.

</details>
<!-- markdownlint-enable MD033 -->

<!-- TODO(vineeth/marketing): write the "Honcho vs RAG / vector DB / memory-only" comparison.
     Audit recommendation referenced; copy intentionally deferred to avoid inventing
     positioning claims unsupported by primary sources. -->

## Benchmarks &amp; Evals

Honcho's evals span LongMemEval, LoCoMo, and other long-conversation benchmarks. See the [evals page](https://honcho.dev/evals/), the [research blog post](https://blog.plasticlabs.ai/research/Benchmarking-Honcho), and the [Pareto-frontier announcement video](https://x.com/honchodotdev/status/2002090546521911703?s=20) for methodology and reproducible results.

## Self-hosting

Honcho is open source under AGPL-3.0. You can run the full server locally with Docker, then point the SDKs at `http://localhost:8000`.

### Quick start (Docker)

```bash
git clone https://github.com/plastic-labs/honcho.git
cd honcho
cp docker-compose.yml.example docker-compose.yml
cp .env.template .env       # fill in LLM_GEMINI_API_KEY / LLM_ANTHROPIC_API_KEY / LLM_OPENAI_API_KEY
docker compose up
```

Then point the SDKs at it:

```python
honcho = Honcho(workspace_id="my-app-testing", base_url="http://localhost:8000")
# or: export HONCHO_URL=http://localhost:8000
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Local development without Docker</summary>

Below is a guide on setting up a local environment for running the Honcho Server without Docker.

#### Prerequisites and Dependencies

Honcho is developed using [python](https://www.python.org/) and [uv](https://docs.astral.sh/uv/).

The minimum python version is `3.10`
The minimum uv version is `0.5.0`

#### Setup

Once the dependencies are installed on the system run the following steps to get
the local project setup.

1. **Clone the repository**

```bash
git clone https://github.com/plastic-labs/honcho.git
```

2. **Enter the repository and install the python dependencies**

We recommend using a virtual environment to isolate the dependencies for Honcho
from other projects on the same system. `uv` will create a virtual environment
when you sync your dependencies in the project.

```bash
cd honcho
uv sync
```

This will create a virtual environment and install the dependencies for Honcho.
The default virtual environment will be located at `honcho/.venv`. Activate the
virtual environment via:

```bash
source honcho/.venv/bin/activate
```

3. **Set up a database**

Honcho utilizes [Postgres](https://www.postgresql.org/) for its database with
pgvector. An easy way to get started with a postgres database is to create a project
with [Supabase](https://supabase.com/)

Alternatively, a `docker-compose` template is available with a sample database configuration.
To use Docker:

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d database
```

4. **Edit the environment variables**

Honcho uses a `.env` file for managing runtime environment variables. A
`.env.template` file is included for convenience. Several of the configurations
are not required and are only necessary for additional logging, monitoring, and
security.

Below are the required configurations:

```env
DB_CONNECTION_URI= # Connection uri for a postgres database (with postgresql+psycopg prefix)

# LLM Provider API Keys
LLM_GEMINI_API_KEY= # API Key for Google Gemini (used for deriver, summary, and dialectic minimal/low by default)
LLM_ANTHROPIC_API_KEY= # API Key for Anthropic (used for dialectic medium/high/max and dream by default)
LLM_OPENAI_API_KEY= # API Key for OpenAI (used for embeddings when EMBED_MESSAGES=true)
```

> Note that the `DB_CONNECTION_URI` must have the prefix `postgresql+psycopg` to
> function properly. This is a requirement brought by `sqlalchemy`

The template has the additional functionality disabled by default. To ensure
that they are disabled you can verify the following environment variables are
set to false:

```env
AUTH_USE_AUTH=false
SENTRY_ENABLED=false
```

If you set `AUTH_USE_AUTH` to true you will need to generate a JWT secret. You can
do this with the following command:

```bash
python scripts/generate_jwt_secret.py
```

This will generate a JWT secret and print it to the console. You can then set
the `AUTH_JWT_SECRET` environment variable. This is required for `AUTH_USE_AUTH`:

```env
AUTH_JWT_SECRET=<generated_secret>
```

5. **Run database migrations**

With the database set up and environment variables configured, run the migrations
to create the necessary tables:

```bash
uv run alembic upgrade head
```

This will create all tables for Honcho including workspaces, peers, sessions,
messages, and the queue system.

6. **Launch Honcho**

With everything set up, you can now launch a local instance of Honcho. In addition to the database, two
components need to be running:

**Start the API server:**

```bash
uv run fastapi dev src/main.py
```

This is a development server that will reload whenever code is changed.

**Start a background worker (deriver):**

In a separate terminal, run:

```bash
uv run python -m src.deriver
```

The deriver generates representations, summaries, peer cards, and manages dreaming tasks. You can increase the number of derivers to improve runtime efficiency.

</details>
<!-- markdownlint-enable MD033 -->

Contributors: see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for pre-commit setup. Deploying to Fly.io: see [Self-hosting docs → Deploying on Fly.io](https://honcho.dev/docs/v3/contributing/self-hosting#deploying-on-fly-io).

## Configuration

Honcho uses a flexible configuration system that supports both TOML files and environment variables. Configuration values are loaded in priority order: **environment variables > `.env` file > `config.toml` > defaults**.

<!-- markdownlint-disable MD033 -->
<details>
<summary>Full configuration reference</summary>

### Using config.toml

Copy the example configuration file to get started:

```bash
cp config.toml.example config.toml
```

Then modify the values as needed. The TOML file is organized into sections:

- `[app]` - Application-level settings (log level, session limits, embedding settings, namespace)
- `[db]` - Database connection and pool settings
- `[auth]` - Authentication configuration
- `[cache]` - Redis cache configuration
- `[llm]` - LLM provider API keys and general settings
- `[deriver]` - Background worker settings and representation configuration
- `[peer_card]` - Peer card generation settings
- `[dialectic]` - Chat Endpoint configuration with per-level reasoning settings
- `[summary]` - Session summarization settings
- `[dream]` - Dream processing configuration (including specialist models and surprisal settings)
- `[webhook]` - Webhook configuration
- `[metrics]` - Prometheus pull-based metrics
- `[telemetry]` - CloudEvents telemetry for analytics
- `[vector_store]` - Vector store configuration (pgvector, turbopuffer, or lancedb)
- `[sentry]` - Error tracking and monitoring settings

### Using Environment Variables

All configuration values can be overridden using environment variables. The environment variable names follow this pattern:

- `{SECTION}_{KEY}` for top-level section settings
- Use `__` inside `{KEY}` for nested settings
- Just `{KEY}` for app-level settings

Examples:

- `DB_CONNECTION_URI` - Database connection string
- `AUTH_JWT_SECRET` - JWT secret key
- `DERIVER_MODEL_CONFIG__TRANSPORT` - Transport for the background deriver
- `SUMMARY_MODEL_CONFIG__MODEL` - Summary model override
- `DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL` - Model for low reasoning level
- `LOG_LEVEL` - Application log level
- `METRICS_ENABLED` - Enable Prometheus metrics
- `TELEMETRY_ENABLED` - Enable CloudEvents telemetry

### Example

If you have this in `config.toml`:

```toml
[db]
CONNECTION_URI = "postgresql+psycopg://localhost/honcho_dev"
POOL_SIZE = 10
```

You can override just the connection URI in production:

```bash
export DB_CONNECTION_URI="postgresql+psycopg://prod-server/honcho_prod"
```

The application will use the production connection URI while keeping the pool size from config.toml.

</details>
<!-- markdownlint-enable MD033 -->

## Architecture

Honcho splits into two services: **Storage** (workspaces, peers, sessions, messages, internal collections) and **Insights** (reasoning, conclusions, representations, summaries, the chat endpoint). Storage is synchronous via the API; Insights is asynchronous via a background queue consumed by the deriver worker process.

**Key features:**

- **Rich Reasoning System** — multiple implementation methods that extract conclusions from interactions and build comprehensive representations of peers
- **Chat Endpoint** — reasoning-informed responses that integrate conclusions with current context
- **Background Processing** — asynchronous processing pipeline for expensive operations like representation updates and session summarization
- **Multi-Provider Support** — configurable LLM providers for different use cases

<!-- markdownlint-disable MD033 MD001 -->
<details>
<summary>Storage primitives in detail</summary>

Honcho contains several different primitives used for storing application and
peer data. This data is used for managing conversations, modeling peer
identity, building RAG applications, and more.

The philosophy behind Honcho is to provide a platform that is peer-centric and
easily scalable from a single user to a million.

Below is a mapping of the different primitives and their relationships.

```
Workspaces
├── Peers ←──────────────────┐
│   ├── Sessions             │
│   └── (internal collections, keyed by observer/observed peer pair)
│                            │
│                            │
└── Sessions ←───────────────┤ (many-to-many)
    ├── Peers ───────────────┘
    └── Messages (session-level)
```

**Relationship Details:**

- A **Workspace** contains multiple **Peers**.
- **Peers** and **Sessions** have a many-to-many relationship (peers can participate in multiple sessions, sessions can have multiple peers).
- **Messages** belong to a session and are labelled by their source peer.
- **Internal collections** of vector-embedded **documents** are keyed by `(observer, observed)` peer pairs. They are not directly exposed via the API; the observations stored in them are exposed as **Conclusions**.

Users familiar with APIs such as the OpenAI Assistants API will be familiar with
much of the mapping here.

#### Workspaces

This is the top level construct of Honcho. Developers can register different
`Workspaces` for different assistants, agents, AI enabled features, etc. It is a way to
isolate data between use cases and provide multi-tenant capabilities.

#### Peers

Within a `Workspace` everything revolves around a `Peer`. The `Peer` object
represents any participant in the system — whether human users or AI agents.
This unified model enables complex multi-participant interactions.

#### Sessions

The `Session` object represents a set of interactions between `Peers` within a
`Workspace`. Other applications may refer to this as a thread or conversation.
Sessions can involve multiple peers with configurable observation settings.

#### Messages

The `Message` represents an atomic data unit that exists at the session level:
communication between peers within a session context. All messages are labelled
by their source peer and can be processed asynchronously to update their
representations. This flexible design allows for both conversational interactions
and broader data ingestion for personality modelling.

</details>
<!-- markdownlint-enable MD033 MD001 -->

<!-- markdownlint-disable MD033 -->
<details>
<summary>Reasoning pipeline</summary>

The reasoning functionality of Honcho is built on top of the Storage service. As
`Messages` and `Sessions` are created for `Peers`, Honcho will asynchronously
reason about peer psychology to derive facts about them and store them
in reserved internal collections.

A high level summary of the pipeline is as follows:

1. Messages are created via the API.
2. Derivation tasks are enqueued for background processing, including:
   - `representation`: update representations of `Peers`.
   - `summary`: create summaries of `Sessions`.
3. Session-based queue processing ensures proper ordering.
4. Results are stored internally and surfaced via the Conclusions API, Representations, Peer Cards, and the Chat Endpoint.

</details>
<!-- markdownlint-enable MD033 -->

<!-- markdownlint-disable MD033 MD001 -->
<details>
<summary>Retrieving data and insights</summary>

Honcho exposes several different ways to retrieve data from the system to best
serve the needs of any given application.

#### Get Context

In long-running conversations with an LLM, the context window can fill up
quickly. To address this, Honcho provides a `context`
endpoint that returns a combination of messages, conclusions, summaries from a
session up to a provided token limit.

Use this to keep sessions going indefinitely. If you'd like to see this in action, try out [Honcho Chat](https://honcho.chat).

#### Search

There are several search endpoints that let developers query messages at the
`Workspace`, `Session`, or `Peer` level using a hybrid search strategy.

Requests can include advanced filters to further refine
the results.

#### Chat API

The flagship interface for using these insights is the [Chat Endpoint](https://honcho.dev/docs/v3/documentation/features/chat) (`POST /peers/{peer_id}/chat`). It takes natural-language requests to get data about a peer and returns reasoning-grounded responses. Examples:

- Asking Honcho for a generic or specific insight about the peer.
- Asking Honcho to hydrate a prompt with data about the peer's behaviour.
- Asking Honcho for a second opinion on how to respond.
- Getting personalised responses that incorporate long-term facts and context.

#### Representations

For low-latency use cases, Honcho provides access to a `representation` endpoint that returns a static document with insights about a peer in the context of a particular session. Use this to quickly add context to a prompt without having to wait for an LLM response.

</details>
<!-- markdownlint-enable MD033 MD001 -->

## SDKs

- **Python** — [`honcho-ai`](https://pypi.org/project/honcho-ai/) on PyPI · source in [`sdks/python/`](./sdks/python)
- **TypeScript** — [`@honcho-ai/sdk`](https://www.npmjs.com/package/@honcho-ai/sdk) on npm · source in [`sdks/typescript/`](./sdks/typescript)

SDKs are versioned independently of the server. Current SDK versions track each other; the server badge above reflects the deployed server version.

See the [SDK Reference](https://honcho.dev/docs/v3/documentation/reference/sdk) for full API surface, the [API Reference](https://honcho.dev/docs/v3/api-reference/introduction) for the raw HTTP API, and per-SDK example folders for runnable demos.

## Learn More

- [Developer documentation](https://honcho.dev/docs/) — full API surface, guides, integrations.
- [Plastic Labs blog](https://blog.plasticlabs.ai/) — design philosophy and history of the project.

## Contributing

We welcome contributions to Honcho! Please read our [Contributing Guide](./CONTRIBUTING.md) for details on our development process, coding conventions, and how to submit pull requests.

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE).
