import logging
import sys
from logging.config import fileConfig
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from alembic import context
from sqlalchemy import engine_from_config, text

from src.config import settings

# Import your models
from src.db import Base

# Set up logging more verbosely
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
logging.getLogger("alembic").setLevel(logging.DEBUG)

# Add project root to Python path
sys.path.append(str(Path(__file__).parents[1]))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url() -> str:
    url = settings.DB.CONNECTION_URI
    if url is None:
        raise ValueError("DB_CONNECTION_URI not set")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=target_metadata.schema,  # This sets schema for version table
    )

    with context.begin_transaction():
        context.execute(f"SET search_path TO {target_metadata.schema}")
        context.run_migrations()


def ensure_session_pooler(connection_uri: str) -> str:
    """
    Ensure a PostgreSQL connection URI uses the session pooler port (5432).

    Converts transaction pooler port (6543) to session pooler port (5432).
    Leaves other ports unchanged.

    Args:
        connection_uri: PostgreSQL connection URI

    Returns:
        Connection URI with session pooler port (5432)

    Examples:
        >>> ensure_session_pooler("postgresql://user:pass@host:6543/db")
        'postgresql://user:pass@host:5432/db'

        >>> ensure_session_pooler("postgresql://user:pass@host:5432/db")
        'postgresql://user:pass@host:5432/db'

        >>> ensure_session_pooler("postgresql+psycopg://user:pass@host.supabase.co:6543/postgres")
        'postgresql+psycopg://user:pass@host.supabase.co:5432/postgres'
    """
    parsed = urlparse(connection_uri)

    # Get current port, default to 5432 if not specified
    current_port = parsed.port or 5432

    # If using transaction pooler port (6543), switch to session pooler (5432)
    if current_port == 6543:
        # Replace the port in the netloc
        if parsed.port:
            # If port is explicitly in the URL, replace it
            new_netloc = parsed.netloc.replace(f":{current_port}", ":5432")
        else:
            # If port not in URL but somehow detected, add it
            new_netloc = f"{parsed.netloc}:5432"

        # Reconstruct the URL with new port
        new_parsed = parsed._replace(netloc=new_netloc)
        return urlunparse(new_parsed)

    return connection_uri


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}

    url = get_url()
    validated_url = ensure_session_pooler(url)
    configuration["sqlalchemy.url"] = validated_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        echo=False,
        connect_args={
            "prepare_threshold": None,
            "options": "-c statement_timeout=300000",  # 5 minutes in milliseconds
        },
    )

    with connectable.connect() as connection:
        # Create schema and commit it outside the main migration transaction
        connection.execute(
            text(f"CREATE SCHEMA IF NOT EXISTS {target_metadata.schema};")
        )
        connection.execute(
            text(f"GRANT ALL ON SCHEMA {target_metadata.schema} TO current_user")
        )
        # Install pgvector extension if it doesn't exist
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Set and verify search_path
        connection.execute(
            text(f"SET search_path TO {target_metadata.schema}, public, extensions")
        )
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=target_metadata.schema,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
