import os

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

load_dotenv()

connect_args = {
    "prepare_threshold": None,
}

# if (
#     os.environ["DATABASE_TYPE"] == "sqlite"
# ):  # https://fastapi.tiangolo.com/tutorial/sql-databases/#note
#     connect_args = {"check_same_thread": False}

engine = create_async_engine(
    os.environ["CONNECTION_URI"],
    connect_args=connect_args,
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true",  # Only enable in debug mode
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_use_lifo=True,  # Use last-in-first-out (LIFO) to prevent connection spread
)

SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

table_schema = os.getenv("DATABASE_SCHEMA")
meta = MetaData()
if table_schema:
    meta.schema = table_schema
Base = declarative_base(metadata=meta)


def scaffold_db():
    """use a sync engine for scaffolding the database. ddl operations are unavailable
    with async engines
    """
    # Create engine
    engine = create_engine(
        os.environ["CONNECTION_URI"],
        pool_pre_ping=True,
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true",
    )

    # Create inspector to check if database exists
    inspector = inspect(engine)

    if table_schema:
        with engine.connect() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{table_schema}"'))
            # Check if pgvector extension exists in the schema
            result = connection.execute(
                text(
                    f"SELECT 1 FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid "
                    f"WHERE e.extname = 'vector' AND n.nspname = '{table_schema}'"
                )
            ).scalar()
            if not result:
                print(f"Installing pgvector extension in schema {table_schema}...")
                connection.execute(
                    text(
                        f'CREATE EXTENSION IF NOT EXISTS vector SCHEMA "{table_schema}"'
                    )
                )
            connection.commit()

    print(inspector.get_table_names(Base.metadata.schema))

    # If no tables exist, create them with SQLAlchemy
    if not inspector.get_table_names(Base.metadata.schema):
        print("No tables found. Creating database schema...")
        Base.metadata.create_all(bind=engine)

    # Clean up
    engine.dispose()

    # Run Alembic migrations regardless
    print("Running database migrations...")
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
