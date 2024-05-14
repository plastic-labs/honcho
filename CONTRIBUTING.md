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

### Docker

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
`flyctl`.

Once `flyctl` is set up use the following commands to launch the application:

```bash
cd honcho/api
flyctl launch --no-deploy # Follow the prompts and edit as you see fit
cat .env | flyctl secrets import # Load in your secrets
flyctl deploy # Deploy with appropriate environment variables
```


## Self-Hosting 

