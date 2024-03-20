# 🫡 Honcho
![Static Badge](https://img.shields.io/badge/Version-0.0.5-blue)
[![Discord](https://img.shields.io/discord/1016845111637839922?style=flat&logo=discord&logoColor=23ffffff&label=Plastic%20Labs&labelColor=235865F2)](https://discord.gg/plasticlabs)
![GitHub License](https://img.shields.io/github/license/plastic-labs/honcho)
![GitHub Repo stars](https://img.shields.io/github/stars/plastic-labs/honcho)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fplastic_labs)](https://twitter.com/plastic_labs)

Honcho is a platform for making AI agents and LLM powered applications that are personalized
to their end users.

Read about the motivation of this project [here](https://blog.plasticlabs.ai).

Read the user documenation [here](https://docs.honcho.dev)

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
    - [API](#api)
        - [Docker](#docker)
        - [Manually](#manually)
        - [Deploying on Fly.io](#deploy-on-fly)
    - [Client SDK](#client-sdk)
        - [Use Locally](#use-locally) 
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The Honcho repo is a monorepo containing the server/API that manages database
interactions and storing data about an application's state along with the python
sdk for interacting with the API.

The folders are structured as follows:

- `api/` - contains a FastAPI application that provides user context management
  routes
- `sdk/` - contains the code for the python sdk and package hosted on PyPI
- `example/` - contains example code for different use cases of honcho

This project utilizes [poetry](https://python-poetry.org/) for dependency
management

A separate changelog is managed for the sdk and api in their respective
directories.

## Usage

### API

#### Docker

The API can be run using docker-compose. The `docker-compose.yml.example` file can be copied to `docker-compose.yml` and the environment variables can be set in the `.env` file.

```bash
cd honcho/api
cp docker-compose.yml.example docker-compose.yml
[ update the file with openai key and other wanted environment variables ]
docker compose up -d
```

#### Manually

The API can be run either by installing the necessary dependencies and then
specifying the appropriate environment variables.

1. Create a virtualenv and install the API's dependencies

```bash
cd honcho/api/ # change to the api directory
poetry shell # Activate virutal environment
poetry install # install dependencies
```

2. Copy the `.env.template` file and specify the type of database and
   connection_uri. For testing sqlite is fine. The below example uses an
   in-memory sqlite database.

> Honcho has been tested with Postgresql and PGVector

```env
DATABASE_TYPE=postgres
CONNECTION_URI=postgresql://testuser:testpwd@localhost:5432/honcho
```

3. launch a postgresd with pgvector enabled with docker-compose

```bash
cd honcho/api/local
docker-compose up -d
```

4. Run the API via uvicorn

```bash
cd honcho/api # change to the api directory
poetry shell # Activate virtual environment if not already enabled
python -m uvicorn src.main:app --reload
```

#### Deploy on Fly

The API can also be deployed on fly.io. Follow the [Fly.io
Docs](https://fly.io/docs/getting-started/) to setup your environment and the
`flyctl`.

Once `flyctl` is set up use the following commands to launch the application:

```bash
cd honcho/api
flyctl launch --no-deploy # Follow the prompts and edit as you see fit
cat .env | flyctl secrets import # Load in your secrets
flyctl deploy # Deploy with appropriate environment variables
```

### Client SDK

Install the honcho client sdk from a python project with the following command:

```bash
pip install honcho-ai
```

alternatively if you are using poetry run:

```bash
poetry add honcho-ai
```

checkout the [SDK Reference](https://api.python.honcho.dev) for a detailed
look at the different methods and how to use them. 

Also, check out the[example folder](./example/) for examples of how to use the sdk 

#### Use Locally

For local development of the sdk you can add the local directory as a package
using poetry with the following commands.

```bash
poetry add --editable ./{path_to_honcho}/honcho/sdk
```

See more information [here](https://python-poetry.org/docs/cli/#add)

## Contributing

This project is completely open source and welcomes any and all open source
contributions. The workflow for contributing is to make a fork of the
repository. You can claim an issue in the issues tab or start a new thread to
indicate a feature or bug fix you are working on.

Once you have finished your contribution make a PR pointed at the `staging`
branch, and it will be reviewed by a project manager. Feel free to join us in
our [discord](http://discord.gg/plasticlabs) to discuss your changes or get
help.

Once your changes are accepted and merged into staging they will undergo a
period of live testing before entering the upstream into `main`

## License

Honcho is licensed under the AGPL-3.0 License. Learn more at the [License file](./LICENSE)
