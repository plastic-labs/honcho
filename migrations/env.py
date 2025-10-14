import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from migrations.verification import set_skip_verification
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


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _compute_skip_verification() -> bool:
    overrides = context.get_x_argument(as_dictionary=True)
    skip_arg = overrides.get("skip_verification")
    env_skip = os.getenv("ALEMBIC_SKIP_VERIFICATION")

    return _as_bool(skip_arg) or _as_bool(env_skip)


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
    set_skip_verification(True)

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

    skip_verification = _compute_skip_verification()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        echo=False,
        poolclass=pool.NullPool,
        connect_args={"prepare_threshold": None},
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

        set_skip_verification(skip_verification)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
