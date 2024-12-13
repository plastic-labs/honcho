import os

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import MetaData
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
    echo=True,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=50,
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
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
