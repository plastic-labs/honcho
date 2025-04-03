# Contributing

This project is completely open source and welcomes any and all open source
contributions. The workflow for contributing is to make a fork of the
repository. You can claim an issue in the issues tab or start a new thread to
indicate a feature or bug fix you are working on.

Once you have finished your contribution make a PR , and it will be reviewed by
a project manager. Feel free to join us in our
[discord](http://discord.gg/plasticlabs) to discuss your changes or get help.

Your changes will undergo a period of testing and discussion before finally
being entered into the `main` branch and being staged for release

## Local Development

Below is a guide on setting up a local environment for running the Honcho
Server.

> This guide was made using a M1 Macbook Pro. For any compatibility issues
> on different platforms please raise an Issue.

### Prerequisites and Dependencies

Honcho is developed using [python](https://www.python.org/) and [uv](https://docs.astral.sh/uv/).

The minimum python version is `3.9`
The minimum poetry version is `0.4.9`

### Setup

Once the dependencies are installed on the system run the following steps to get
the local project setup.

1. Clone the repository

```bash
git clone https://github.com/plastic-labs/honcho.git
```

2. Enter the repository and install the python dependencies

We recommend using a virtual environment to isolate the dependencies for Honcho
from other projects on the same system. `uv` will create a virtual environment
when you sync your dependencies in the project.

Putting this together:

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

3. Set up a database

Honcho utilized [Postgres](https://www.postgresql.org/) for its database with
pgvector. An easy way to get started with a postgresdb is to create a project
with [Supabase](https://supabase.com/)

A `docker-compose` template is also available with a database configuration
available.

4. Edit the environment variables.

Honcho uses a `.env` file for managing runtime environment variables. A
`.env.template` file is included for convenience. Several of the configurations
are not required and are only necessary for additional logging, monitoring, and
security.

Below are the required configurations

```env
CONNECTION_URI= # Connection uri for a postgres database
OPENAI_API_KEY= # API Key for OpenAI used for embedding documents
ANTHROPIC_API_KEY= # API Key for Anthropic used for the deriver and dialectic API
```

> Note that the `CONNECTION_URI` must have the prefix `postgresql+psycopg` to
> function properly. This is a requirement brought by `sqlalchemy`

The template has the additional functionality disabled by default. To ensure
that they are disabled you can verify the following environment variables are
set to false.

```env
USE_AUTH=false
SENTRY_ENABLED=false
```

If you set `USE_AUTH` to true you will need to generate a JWT secret. You can
do this with the following command:

```bash
python scripts/generate_jwt_secret.py
```

This will generate a JWT secret and print it to the console. You can then set
the `AUTH_JWT_SECRET` environment variable. This is required for `USE_AUTH`.

```env
AUTH_JWT_SECRET=<generated_secret>
```

5. Launch the API

With the dependencies installed, a database setup and enabled with `pgvector`,
and the environment variables setup you can now launch a local instance of
Honcho. The following command will launch the storage API for Honcho

```bash
fastapi dev src/main.py
```

This is a development server that will reload whenever code is changed. When
first launching the API with a connection the database it will provision the
necessary tables for Honcho to operate.

### Docker

As mentioned earlier a `docker-compose` template is included for running Honcho.
As an alternative to running Honcho locally it can also be run with the compose
template.

The docker-compose template is set to use an environment file called `.env`.
You can also copy the `.env.template` and fill with the appropriate values.

Copy the template and update the appropriate environment variables before
launching the service.

```bash
cd honcho/api
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

> Note. The fly.toml does not include launching a Postgres database. This must
> be configured separately

Once `flyctl` is set up use the following commands to launch the application:

```bash
cd honcho/api
flyctl launch --no-deploy # Follow the prompts and edit as you see fit
cat .env | flyctl secrets import # Load in your secrets
flyctl deploy # Deploy with appropriate environment variables
```
