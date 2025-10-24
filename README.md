<!-- markdownlint-disable MD033 -->
<div align="center">
  <a href="https://app.honcho.dev" target="_blank">
    <img src="assets/honcho.svg" alt="Honcho" width="400">
  </a>
</div>
<!-- markdownlint-enable MD033 -->

---

![Static Badge](https://img.shields.io/badge/Version-2.4.1-blue)
[![PyPI version](https://img.shields.io/pypi/v/honcho-ai.svg)](https://pypi.org/project/honcho-ai/)
[![NPM version](https://img.shields.io/npm/v/@honcho-ai/sdk.svg)](https://npmjs.org/package/@honcho-ai/sdk)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/plasticlabs)
[![arXiv](https://img.shields.io/badge/arXiv-2310.06983-b31b1b.svg)](https://arxiv.org/abs/2310.06983)

Honcho is an AI-native memory library for building agents with perfect memory and
social cognition.

It provides [state-of-the-art
memory](https://blog.plasticlabs.ai/research/Introducing-Neuromancer-XR) and
then goes beyond storage by reasoning about the stored data to build
rich psychological profiles of each user in your system.

Use it to build

- Highly personalized experiences
- Agents with social cognition
- Agents with rich identity that evolve over time
- Multi-agent systems with complex social dynamics

## TL;DR - Getting Started

With Honcho you can easily setup your application's workflow, save your
interaction history, and leverage generated insights to inform the behavior of
your agents

> Typescript examples are available in our [docs](https://docs.honcho.dev)

1. Install the SDK

```bash
# Python
pip install honcho-ai
uv add honcho-ai
poetry add honcho-ai
```

2. Setup your `Workspace`, `Peers`, `Session`, and send `Messages`

```python
from honcho import Honcho

####### Storing Data in Honcho

# 1. Initialize your Honcho client, by default SDK will use the demo environment and workspace named "default"
honcho = Honcho(environment="demo", workspace_id="my-app-testing")

# 2.. Initialize Peers
alice = honcho.peer("alice")
tutor = honcho.peer("tutor")

# 3. Make a Session and send messages

session = honcho.session("session-1")

session.add_messages(
  alice.message("Hey there can you help me with my math homework"),
  tutor.message("Absolutely send me your first problem!"),
  .
  .
  .
)
```

3. Leverage insights from Honcho to inform your agent's behavior

```python

### 1. Use the Dialectic API to ask questions about your users in natural language
response = alice.chat("What learning styles does the user respond to best?")

### 2. Use Get context to get most recent messages and summaries to continue a conversation
context = session.get_context(summary=True, tokens=10000)

# Convert to a format to send to OpenAI and get the next message
openai_messages = context.to_openai_messages(assistant=tutor)

from openai import OpenAI
client = Openai()
response = client.chat.completions.create(
  model="gpt-4",
  messages=openai_messages
)

### 3. Search for similar messages
results = alice.search("Math Homework")

### 4. Get a cached working representation of a Peer for the Session
alice_representation = session.working_rep("alice")

```

This is a simple example of how you can use Honcho to build a chatbot and
leverage insights to personalize the agent's behavior.

Sign up at [app.honcho.dev](https://app.honcho.dev) to get started with a managed version of Honcho.

Learn more ways to use Honcho on our [developer docs](https://docs.honcho.dev).

Read about the design philosophy and history of the project on our [blog](https://blog.plasticlabs.ai/).

## Project Structure

- [Usage](#usage)
- [Local Development](#local-development)
  - [Prerequisites and Dependencies](#prerequisites-and-dependencies)
  - [Setup](#setup)
  - [Docker](#docker)
  - [Deploy on Fly](#deploy-on-fly)
- [Configuration](#configuration)
  - [Using config.toml](#using-configtoml)
  - [Using Environment Variables](#using-environment-variables)
  - [Configuration Priority](#configuration-priority)
  - [Example](#example)
- [Architecture](#architecture)
  - [Storage](#storage)
  - [Reasoning](#reasoning)
  - [Retrieving Data & Insights](#retrieving-data--insights)
- [Contributing](#contributing)
- [License](#license)

The Honcho project is split between several repositories with this one hosting
the core service logic. This is implemented as a FastAPI server/API to store
data about an application's state.

There are also client sdks in implemented in the `sdks/` directory with support
for Python and TypeScript. These SDKs wrap core SDKs that are generated using
[Stainless](https://www.stainlessapi.com/).

- [Python](https://pypi.org/project/honcho-ai/)
- [TypeScript](https://www.npmjs.com/package/@honcho-ai/sdk)

We recommend using the official client SDKs instead of the core ones for better
developer experience, however for any custom use cases you can still access the
core SDKs in their own repos:

- [Honcho Core Python](https://github.com/plastic-labs/honcho-python-core)
- [Honcho Core TypeScript](https://github.com/plastic-labs/honcho-node-core)

Examples on how to use the SDK are located within each SDK folder and in the
[SDK Reference](https://docs.honcho.dev/v2/documentation/tutorial/SDK)

There are also documented examples of how to use the core SDKs in the
[API Reference](https://docs.honcho.dev/api-reference/introduction) section of
the documentation.

## Usage

When you first install the SDKs they will be ready to go, pointing at
[https://demo.honcho.dev](https://demo.honcho.dev) which is a demo server of Honcho. This server has no
authentication, no SLA, and should only be used for testing and getting familiar
with Honcho.

For a production ready version of Honcho sign up for an account at
[https://app.honcho.dev](https://app.honcho.dev) and get started. When you sign up you'll be prompted to
join an organization which will have a dedicated instance of Honcho.

Provision API keys and change your base url to point to
[https://api.honcho.dev](https://api.honcho.dev)

Additionally, Honcho can be self-hosted for testing and evaluation purposes. See
the [Local Development](#local-development) section below for details on how to set up a local
version of Honcho.

## Local Development

Below is a guide on setting up a local environment for running the Honcho
Server.

> This guide was made using a M3 Macbook Pro. For any compatibility issues
> on different platforms, please raise an Issue.

### Prerequisites and Dependencies

Honcho is developed using [python](https://www.python.org/) and [uv](https://docs.astral.sh/uv/).

The minimum python version is `3.9`
The minimum uv version is `0.4.9`

### Setup

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

A `docker-compose` template is also available with a database configuration.

4. **Edit the environment variables**

Honcho uses a `.env` file for managing runtime environment variables. A
`.env.template` file is included for convenience. Several of the configurations
are not required and are only necessary for additional logging, monitoring, and
security.

Below are the required configurations:

```env
DB_CONNECTION_URI= # Connection uri for a postgres database (with postgresql+psycopg prefix)

# LLM Provider API Keys (at least one required depending on your configuration)
LLM_ANTHROPIC_API_KEY= # API Key for Anthropic (used for dialectic by default)
LLM_OPENAI_API_KEY= # API Key for OpenAI (optional, for embeddings if EMBED_MESSAGES=true)
LLM_GEMINI_API_KEY= # API Key for Google Gemini (used for summary/deriver by default)
LLM_GROQ_API_KEY= # API Key for Groq (used for query generation by default)
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

5. **Launch the API**

With the dependencies installed, a database setup and enabled with `pgvector`,
and the environment variables setup you can now launch a local instance of
Honcho. The following command will launch the storage API for Honcho:

```bash
fastapi dev src/main.py
```

This is a development server that will reload whenever code is changed. When
first launching the API with a connection to the database it will provision the
necessary tables for Honcho to operate.

### Pre-commit Hooks

Honcho uses pre-commit hooks to ensure code quality and consistency across the project. These hooks automatically run checks on your code before each commit, including linting, formatting, type checking, and security scans.

#### Installation

To set up pre-commit hooks in your development environment:

1. **Install pre-commit using uv**

```bash
uv add --dev pre-commit
```

2. **Install the pre-commit hooks**

```bash
uv run pre-commit install \
    --hook-type pre-commit \
    --hook-type commit-msg \
    --hook-type pre-push
```

This will install hooks for `pre-commit`, `commit-msg`, and `pre-push` stages.

#### What the hooks do

The pre-commit configuration includes:

- **Code Quality**: Python linting and formatting (ruff), TypeScript linting (biome)
- **Type Checking**: Static type analysis with basedpyright
- **Security**: Vulnerability scanning with bandit
- **Documentation**: Markdown linting and license header checks
- **Testing**: Automated test runs for Python and TypeScript code
- **File Hygiene**: Trailing whitespace, line endings, file size checks
- **Commit Standards**: Conventional commit message validation

#### Manual execution

You can run the hooks manually on all files without making a commit:

```bash
uv run pre-commit run --all-files
```

Or run specific hooks:

```bash
uv run pre-commit run ruff --all-files
uv run pre-commit run basedpyright --all-files
```

### Docker

As mentioned earlier a `docker-compose` template is included for running Honcho.
As an alternative to running Honcho locally it can also be run with the compose
template.

The docker-compose template is set to use an environment file called `.env`.
You can also copy the `.env.template` and fill with the appropriate values.

Copy the template and update the appropriate environment variables before
launching the service:

```bash
cd honcho
cp .env.template .env
# update the file with openai key and other wanted environment variables
cp docker-compose.yml.example docker-compose.yml
docker compose up
```

### Deploy on Fly

The API can also be deployed on fly.io. Follow the [Fly.io
Docs](https://fly.io/docs/getting-started/) to setup your environment and the
`flyctl`.

A sample `fly.toml` is included for convenience.

> Note: The fly.toml does not include launching a Postgres database. This must
> be configured separately

Once `flyctl` is set up use the following commands to launch the application:

```bash
cd honcho
flyctl launch --no-deploy # Follow the prompts and edit as you see fit
cat .env | flyctl secrets import # Load in your secrets
flyctl deploy # Deploy with appropriate environment variables
```

## Configuration

Honcho uses a flexible configuration system that supports both TOML files and environment variables. Configuration values are loaded in the following priority order (highest to lowest):

1. Environment variables
2. `.env` file (for local development)
3. `config.toml` file
4. Default values

### Using config.toml

Copy the example configuration file to get started:

```bash
cp config.toml.example config.toml
```

Then modify the values as needed. The TOML file is organized into sections:

- `[app]` - Application-level settings (log level, host, port, embedding settings)
- `[db]` - Database connection and pool settings
- `[auth]` - Authentication configuration
- `[llm]` - LLM provider API keys and general settings
- `[dialectic]` - Dialectic API configuration (provider, model, search settings)
- `[deriver]` - Background worker settings and theory of mind configuration
- `[summary]` - Session summarization settings
- `[sentry]` - Error tracking and monitoring settings

### Using Environment Variables

All configuration values can be overridden using environment variables. The environment variable names follow this pattern:

- `{SECTION}_{KEY}` for nested settings
- Just `{KEY}` for app-level settings

Examples:

- `DB_CONNECTION_URI` - Database connection string
- `AUTH_JWT_SECRET` - JWT secret key
- `DIALECTIC_MODEL` - Dialectic API model
- `SUMMARY_PROVIDER` - Summary generation provider
- `LOG_LEVEL` - Application log level

### Configuration Priority

When a configuration value is set in multiple places, Honcho uses this priority:

1. **Environment variables** - Always take precedence
2. **.env file** - Loaded for local development
3. **config.toml** - Base configuration
4. **Default values** - Built-in defaults

This allows you to:

- Use `config.toml` for base configuration
- Override specific values with environment variables in production
- Use `.env` files for local development without modifying config.toml

### Example

If you have this in `config.toml`:

```toml
[db]
CONNECTION_URI = "postgresql://localhost/honcho_dev"
POOL_SIZE = 10
```

You can override just the connection URI in production:

```bash
export DB_CONNECTION_URI="postgresql://prod-server/honcho_prod"
```

The application will use the production connection URI while keeping the pool size from config.toml.

## Architecture

The functionality of Honcho can be split into two different services: Storage
and Insights.

### Peer Paradigm

Honcho uses a peer-based model where both users and agents are represented as "peers". This unified approach enables:

- Multi-participant sessions with mixed human and AI agents
- Configurable observation settings (which peers observe which others)
- Flexible identity management for all participants
- Support for complex multi-agent interactions

#### Key Features

- **Theory-of-Mind System**: Multiple implementation methods that extract facts from interactions and build comprehensive models of peer psychology
- **Dialectic API**: Provides theory-of-mind informed responses that integrate long-term facts with current context
- **Background Processing**: Asynchronous processing pipeline for expensive operations like representation updates and session summarization
- **Multi-Provider Support**: Configurable LLM providers for different use cases

### Storage

Honcho contains several different primitives used for storing application and
peer data. This data is used for managing conversations, modeling peer
psychology, building RAG applications, and more.

The philosophy behind Honcho is to provide a platform that is peer-centric and
easily scalable from a single user to a million.

Below is a mapping of the different primitives and their relationships.

```
Workspaces
├── Peers ←──────────────────┐
│   ├── Sessions             │
│   └── Collections          │
│       └── Documents        │
│                            │
│                            │
└── Sessions ←───────────────┤ (many-to-many)
    ├── Peers ───────────────┘
    └── Messages (session-level)
```

**Relationship Details:**

- A **Workspace** contains multiple **Peers**
- **Peers** and **Sessions** have a many-to-many relationship (peers can participate in multiple sessions, sessions can have multiple peers)
- **Messages** can exist at two levels:
  - **Session-level**: Communication between peers within a session
- **Collections** belong to specific **Peers**
- **Documents** are stored within **Collections**

Users familiar with APIs such as the OpenAI Assistants API will be familiar with
much of the mapping here.

#### Workspaces

This is the top level construct of Honcho (formerly called Apps). Developers can register different
`Workspaces` for different assistants, agents, AI enabled features, etc. It is a way to
isolate data between use cases and provide multi-tenant capabilities.

#### Peers

Within a `Workspace` everything revolves around a `Peer`. The `Peer` object
represents any participant in the system - whether human users or AI agents.
This unified model enables complex multi-participant interactions.

#### Sessions

The `Session` object represents a set of interactions between `Peers` within a
`Workspace`. Other applications may refer to this as a thread or conversation.
Sessions can involve multiple peers with configurable observation settings.

#### Messages

The `Message` represents an atomic data unit that can exist at two levels:

- **Session-level Messages**: Communication between peers within a session context

All messages are labeled by their source peer and can be processed
asynchronously to update theory-of-mind models. This flexible design allows for
both conversational interactions and broader data ingestion for personality
modeling.

#### Collections

At a high level a `Collection` is a named group of `Documents`. Developers
familiar with RAG based applications will be familiar with these. `Collections`
store vector embedded data that developers and agents can retrieve against using
functions like cosine similarity.

Collections are also used internally by Honcho while creating theory-of-mind
representations of peers.

#### Documents

As stated before a `Document` is vector embedded data stored in a `Collection`.

### Reasoning

The reasoning functionality of Honcho is built on top of the Storage service. As
`Messages` and `Sessions` are created for `Peers`, Honcho will asynchronously
reason about peer psychology to derive facts about them and store them
in reserved `Collections`.

A high level summary of the pipeline is as follows:

1. Messages are created via the API
2. Derivation Tasks are enqueued for background processing including:
   - `representation`: To update theory-of-mind representations of `Peers`
   - `summary`: To create summaries of `Sessions`
3. Session-based queue processing ensures proper ordering
4. Results are stored internally

To read more about how this works read our [Research Paper](https://arxiv.org/abs/2310.06983)

### Retrieving Data & Insights

Honcho exposes several different ways to retrieve data from the system to best
serve the needs of any given application.

#### Get Context

In long-running conversations with an LLM, the context window can fill up
quickly. To address this, Honcho provides a `get_context`
endpoint that returns a combination of messages and summaries from a
session, up to a provided token limit.

Use this to keep sessions going indefinitely.

#### Search

There are several search endpoints that let developers query messages at the
`Workspace`, `Session`, or `Peer` level using a hybrid search strategy.

Requests can include advanced filters to further refine
the results.

#### Dialectic API

The flagship interface for using these insights is through
the [Dialectic Endpoint](https://blog.plasticlabs.ai/blog/Introducing-Honcho's-Dialectic-API).

This is a regular API endpoint (`/peers/{peer_id}/chat`) that takes natural language requests to get data
about the `Peer`. This robust design lets us use this single endpoint for all
cases where extra personalization or information about the `Peer` is necessary.

A developer's application can treat Honcho as an oracle to the `Peer` and
consult it when necessary. Some examples of how to leverage the Dialectic
API include:

- Asking Honcho for a theory-of-mind insight about the `Peer`
- Asking Honcho to hydrate a prompt with data about the `Peer`s behavior
- Asking Honcho for a 2nd opinion or approach about how to respond to the Peer
- Getting personalized responses that incorporate long-term facts and context

#### Working Representations

For low-latency use cases,
Honcho provides access to a `get_working_representation` endpoint that
returns a static document with insights about a `Peer` in the context of a
particular session.

Use this to quickly add context to a prompt without having to wait for an LLM
response.

## Contributing

We welcome contributions to Honcho! Please read our [Contributing Guide](./CONTRIBUTING.md) for details on our development process, coding conventions, and how to submit pull requests.

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
