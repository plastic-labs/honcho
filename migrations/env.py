import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool, text

# Set up logging more verbosely
logging.basicConfig()
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
logging.getLogger("alembic").setLevel(logging.DEBUG)

# Import your models
from src.db import Base

# Add project root to Python path
sys.path.append(str(Path(__file__).parents[1]))

# Load environment variables
load_dotenv()

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


def include_name(name, type_, parent_names):
    if type_ == "schema":
        return name == target_metadata.schema
    else:
        return True


def get_url() -> str:
    url = os.getenv("CONNECTION_URI")
    if url is None:
        raise ValueError("CONNECTION_URI environment variable is not set")
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

    print(target_metadata.schema)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=target_metadata.schema,  # This sets schema for version table
        include_schemas=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.execute(f"create schema if not exists {target_metadata.schema};")
        context.execute(f"SET search_path TO {target_metadata.schema}")
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    if configuration is None:
        configuration = {}

    url = get_url()
    configuration["sqlalchemy.url"] = url

    print(f"Debug - Target metadata schema: {target_metadata.schema}")

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        echo=True,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Set and verify search_path
        connection.execute(text(f"SET search_path TO {target_metadata.schema}, public"))

        # make use of non-supported SQLAlchemy attribute to ensure
        # the dialect reflects tables in terms of the current tenant name
        # connection.dialect.default_schema_name = target_metadata.schema

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=target_metadata.schema,
            include_schemas=True,
            include_name=include_name,
            transaction_per_migration=True,
            transactional_ddl=True,
        )

        with context.begin_transaction():
            context.run_migrations()
            connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
