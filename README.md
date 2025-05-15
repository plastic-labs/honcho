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

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
