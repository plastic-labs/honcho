# 🫡 Honcho

![Static Badge](https://img.shields.io/badge/Version-1.1.0-blue)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/plasticlabs)
[![arXiv](https://img.shields.io/badge/arXiv-2310.06983-b31b1b.svg)](https://arxiv.org/abs/2310.06983)
![GitHub License](https://img.shields.io/github/license/plastic-labs/honcho)
![GitHub Repo stars](https://img.shields.io/github/stars/plastic-labs/honcho)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fplastic_labs)](https://twitter.com/plastic_labs)
[![PyPI version](https://img.shields.io/pypi/v/honcho-ai.svg)](https://pypi.org/project/honcho-ai/)
[![NPM version](https://img.shields.io/npm/v/honcho-ai.svg)](https://npmjs.org/package/honcho-ai)

Honcho is a platform for making AI agents and LLM powered applications that are personalized
to their end users. It leverages the inherent theory-of-mind capabilities of
LLMs to cohere to user psychology over time.

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

Currently, there is a demo server of Honcho running at https://demo.honcho.dev.
This server is not production ready and does not have an reliability guarantees.
It is purely there for evaluation purposes.

A private beta for a tenant isolated production ready version of Honcho is
currently underway. If interested fill out this
[typeform](https://plasticlabs.typeform.com/honchobeta) and the Plastic Labs
team will reach out to onboard users.

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

### Storage

Honcho contains several different primitives used for storing application and
user data. This data is used for managing conversations, modeling user
psychology, building RAG applications, and more.

The philosophy behind Honcho is to provide a platform that is user-centric and
easily scalable from a single user to a million.

Below is a mapping of the different primitives.

```
Apps
└── Users
    ├── Sessions
    │   └── Messages
    ├── Collections
    │   └── Documents
    └── Metamessages
```

Users familiar with APIs such as the OpenAI Assistants API will be familiar with
much of the mapping here.

#### Apps

This is the top level construct of Honcho. Developers can register different
`Apps` for different assistants, agents, AI enabled features, etc. It is a way to
isolation data between use cases.

**Users**

Within an `App` everything revolves around a `User`. the `User` object
literally represent a user of an application.

#### Sessions

The `Session` object represents a set of interactions a `User` has with an
`App`. Other application may refer to this as a thread or conversation.

**Messages**

The `Message` represents an atomic interaction of a `User` in a `Session`.
`Message`s are labeled as either a `User` or AI message.

#### Collections

At a high level a `Collection` is a named group of `Documents`. Developers
familiar with RAG based applications will be familiar with these. `Collection`s
store vector embedded data that developers and agents can retrieve against using
functions like cosine similarity.

Developers can create multiple `Collection`s for a user for different purposes
such as modeling different personas, adding third-party data such as emails and
PDF files, and more.

#### Documents

As stated before a `Document` is vector embedded data stored in a `Collection`.

#### Metamessages

A `Metamessage` is similar to a `Message` with different use case. They are
meant to be used to store intermediate inference from AI assistants or other
derived information that is separate from the main `User` `App` interaction
loop. For complicated prompting architectures like [metacognitive prompting](https://arxiv.org/abs/2310.06983)
metamessages can store thought and reflection steps along with having developer
information such as logs.

Each `Metamessage` is associated with a `User` with the ability to optionally
tie to a `Session` and a `Message`.

### Insights

The Insight functionality of Honcho is built on top of the Storage service. As
`Messages` and `Sessions` are created for a `User`, Honcho will asynchronously
reason about the `User`'s psychology to derive facts about them and store them
in a reserved `Collection`.

To read more about how this works read our [Research Paper](https://arxiv.org/abs/2310.06983)

Developers can then leverage these insights in their application to better
server `User` needs. The primary interface for using these insights is through
the [Dialectic Endpoint](https://blog.plasticlabs.ai/blog/Introducing-Honcho's-Dialectic-API).

This is a regular API endpoint that takes natural language requests to get data
about the `User`. This robust design let's us use this single endpoint for all
cases where extra personalization or information about the `User` is necessary.

A developer's application can treat Honcho as an oracle to the `User` and
consult it when necessary. Some examples of how to leverage the Dialectic
API include:

- Asking Honcho for a theory-of-mind insight about the `User`
- Asking Honcho to hydrate a prompt with data about the `User`s behavior
- Asking Honcho for a 2nd opinion or approach about how to respond to the User

## Contributing

We welcome contributions to Honcho! Please read our [Contributing Guide](./CONTRIBUTING.md) for details on our development process, coding conventions, and how to submit pull requests.

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
