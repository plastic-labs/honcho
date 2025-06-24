# 🫡 Honcho

![Static Badge](https://img.shields.io/badge/Version-2.0.1-blue)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/plasticlabs)
[![arXiv](https://img.shields.io/badge/arXiv-2310.06983-b31b1b.svg)](https://arxiv.org/abs/2310.06983)
![GitHub License](https://img.shields.io/github/license/plastic-labs/honcho)
![GitHub Repo stars](https://img.shields.io/github/stars/plastic-labs/honcho)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fplastic_labs)](https://twitter.com/plastic_labs)
[![PyPI version](https://img.shields.io/pypi/v/honcho-ai.svg)](https://pypi.org/project/honcho-ai/)
[![NPM version](https://img.shields.io/npm/v/honcho-ai.svg)](https://npmjs.org/package/@honcho-ai/sdk)

Honcho is an infrastructure layer for building AI agents with social cognition and theory-of-mind capabilities. It enables developers to create AI agents and LLM-powered applications that are personalized to their end users by leveraging the inherent theory-of-mind capabilities of LLMs to build coherent models of user psychology over time.

Read about the project [here](https://blog.plasticlabs.ai/blog/A-Simple-Honcho-Primer).

Read the user documentation [here](https://docs.honcho.dev)

## Table of Contents

- [Project Structure](#project-structure)
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
  - [Insights](#insights)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The Honcho project is split between several repositories with this one hosting
the core service logic. This is implemented as a FastAPI server/API to store
data about an application's state.

There are also client-sdks that are created using
[Stainless](https://www.stainlessapi.com/). Currently, there is a [Python](https://github.com/plastic-labs/honcho-python) and
[TypeScript/JavaScript](https://github.com/plastic-labs/honcho-node) SDK available.

Examples on how to use the SDK are located within each SDK repository. There is
also SDK example usage available in the [API Reference](https://docs.honcho.dev/api-reference/introduction)
along with various guides.

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
DB_CONNECTION_URI= # Connection uri for a postgres database
OPENAI_API_KEY= # API Key for OpenAI used for embedding documents
ANTHROPIC_API_KEY= # API Key for Anthropic used for the deriver and dialectic API
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

- `[app]` - Application-level settings (log level, host, port)
- `[db]` - Database connection and pool settings
- `[auth]` - Authentication configuration
- `[llm]` - LLM provider and model settings
- `[agent]` - Agent behavior settings
- `[deriver]` - Background worker settings
- `[history]` - Message history settings

### Using Environment Variables

All configuration values can be overridden using environment variables. The environment variable names follow this pattern:

- `{SECTION}_{KEY}` for nested settings
- Just `{KEY}` for app-level settings

Examples:

- `DB_CONNECTION_URI` - Database connection string
- `AUTH_JWT_SECRET` - JWT secret key
- `LLM_DIALECTIC_MODEL` - Dialectic LLM model
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
│   ├── Collections          │
│   │   └── Documents        │
│   └── Messages (peer-level)│
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
  - **Peer-level**: Data ingested by a peer to enhance its global representation
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
- **Peer-level Messages**: Arbitrary data ingested by a peer to enhance its global representation (independent of any session)

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

### Insights

The Insight functionality of Honcho is built on top of the Storage service. As
`Messages` and `Sessions` are created for `Peers`, Honcho will asynchronously
reason about peer psychology to derive facts about them and store them
in reserved `Collections`.

The system uses a sophisticated message processing pipeline:

1. Messages are created via API
2. Enqueued for background processing including:
   - `representation`: Update peer's theory of mind
   - `summary`: Create session summaries
3. Session-based queue processing ensures proper ordering
4. Results are stored internally in the vector database

To read more about how this works read our [Research Paper](https://arxiv.org/abs/2310.06983)

Developers can then leverage these insights in their application to better
serve peer needs. The primary interface for using these insights is through
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

## Contributing

We welcome contributions to Honcho! Please read our [Contributing Guide](./CONTRIBUTING.md) for details on our development process, coding conventions, and how to submit pull requests.

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
