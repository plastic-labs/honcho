import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()

connect_args = {}

if (
    os.environ["DATABASE_TYPE"] == "sqlite"
):  # https://fastapi.tiangolo.com/tutorial/sql-databases/#note
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    os.environ["CONNECTION_URI"],
    connect_args=connect_args,
    echo=True,
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def scaffold_db():
    """use a sync engine for scaffolding the database. ddl operations are unavailable
    with async engines
    """
    print(os.environ["CONNECTION_URI"])
    engine = create_engine(
        os.environ["CONNECTION_URI"],
        pool_pre_ping=True,
        echo=True,
    )
    Base.metadata.create_all(bind=engine)
    engine.dispose()
