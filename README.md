# 🫡 Honcho

![Static Badge](https://img.shields.io/badge/Version-2.0.0-blue)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/plasticlabs)
[![arXiv](https://img.shields.io/badge/arXiv-2310.06983-b31b1b.svg)](https://arxiv.org/abs/2310.06983)
![GitHub License](https://img.shields.io/github/license/plastic-labs/honcho)
![GitHub Repo stars](https://img.shields.io/github/stars/plastic-labs/honcho)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fplastic_labs)](https://twitter.com/plastic_labs)
[![PyPI version](https://img.shields.io/pypi/v/honcho-ai.svg)](https://pypi.org/project/honcho-ai/)
[![NPM version](https://img.shields.io/npm/v/honcho-ai.svg)](https://npmjs.org/package/honcho-ai)

Honcho is an infrastructure layer for building AI agents with social cognition and theory-of-mind capabilities. It enables developers to create AI agents and LLM-powered applications that are personalized to their end users by leveraging the inherent theory-of-mind capabilities of LLMs to build coherent models of user psychology over time.

Read about the project [here](https://blog.plasticlabs.ai/blog/A-Simple-Honcho-Primer).

Read the user documentation [here](https://docs.honcho.dev)

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Architecture](#architecture)
  - [Storage](#storage)
  - [Insights](#insights)
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
[Contributing](./CONTRIBUTING.md) for more details on how to setup a local
version of Honcho.

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

All messages are labeled by their source peer and can be processed asynchronously to update theory-of-mind models. This flexible design allows for both conversational interactions and broader data ingestion for personality modeling.

#### Collections

At a high level a `Collection` is a named group of `Documents`. Developers
familiar with RAG based applications will be familiar with these. `Collections`
store vector embedded data that developers and agents can retrieve against using
functions like cosine similarity.

Developers can create multiple `Collections` for a peer for different purposes
such as modeling different personas, adding third-party data such as emails and
PDF files, and more. Collections are also used internally by Honcho to store
theory-of-mind representations.

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

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
