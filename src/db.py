import contextvars
import logging
from urllib.parse import SplitResult, quote_plus, urlsplit, urlunsplit

from sqlalchemy import MetaData, event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from src.config import settings

logger = logging.getLogger(__name__)

connect_args: dict = {"prepare_threshold": None}

# Context variable to store request context
request_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_context", default=None
)

engine_kwargs: dict = {}


def _parse_base_uri(connection_uri: str) -> SplitResult:
    """Parse a PostgreSQL connection URI into components."""
    return urlsplit(connection_uri)


# Determine connection URI and auth-specific settings
if settings.DB.AUTH_METHOD == "iam":
    _parsed = _parse_base_uri(settings.DB.CONNECTION_URI)
    _db_path = _parsed.path or "/postgres"

    # Construct base URI from RDS settings, preserving original path/query
    connection_uri = urlunsplit((
        _parsed.scheme or "postgresql+psycopg",
        f"{settings.DB.RDS_USERNAME}@{settings.DB.RDS_HOSTNAME}:{settings.DB.RDS_PORT}",
        _db_path,
        _parsed.query,
        "",
    ))

    # SSL connect args required for IAM auth
    connect_args["sslmode"] = "require"
    if settings.DB.RDS_SSL_CA_BUNDLE:
        connect_args["sslrootcert"] = settings.DB.RDS_SSL_CA_BUNDLE
else:
    # Password mode: use CONNECTION_URI as-is
    connection_uri = settings.DB.CONNECTION_URI

if settings.DB.POOL_CLASS == "null":
    engine_kwargs["poolclass"] = NullPool
else:
    # Determine pool settings, with IAM overrides
    pool_pre_ping = settings.DB.POOL_PRE_PING
    pool_recycle = settings.DB.POOL_RECYCLE

    if settings.DB.AUTH_METHOD == "iam":
        pool_pre_ping = True
        pool_recycle = min(pool_recycle, 900)

    engine_kwargs.update(  # pyright: ignore
        {
            "pool_pre_ping": pool_pre_ping,
            "pool_size": settings.DB.POOL_SIZE,
            "max_overflow": settings.DB.MAX_OVERFLOW,
            "pool_timeout": settings.DB.POOL_TIMEOUT,
            "pool_recycle": pool_recycle,
            "pool_use_lifo": settings.DB.POOL_USE_LIFO,
        }
    )

engine = create_async_engine(
    connection_uri,
    connect_args=connect_args,
    echo=settings.DB.SQL_DEBUG,
    **engine_kwargs,
)

# Register do_connect event listener for IAM token injection
if settings.DB.AUTH_METHOD == "iam":
    from src.aws_auth import generate_rds_auth_token

    @event.listens_for(engine.sync_engine, "do_connect")
    def _inject_iam_token(dialect, conn_rec, cargs, cparams):
        """Generate a fresh IAM auth token for each new physical connection."""
        token = generate_rds_auth_token(
            region=settings.DB.AWS_REGION,
            hostname=settings.DB.RDS_HOSTNAME,
            port=settings.DB.RDS_PORT,
            username=settings.DB.RDS_USERNAME,
            profile=settings.DB.AWS_PROFILE,
        )
        cparams["password"] = token

SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

# Define your naming convention
convention = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",  # Index - supports multi-column
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",  # Unique constraint - supports multi-column
    "ck": "ck_%(table_name)s_%(constraint_name)s",  # Check constraint
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",  # Foreign key - supports composite keys
    "pk": "pk_%(table_name)s",  # Primary key
}

table_schema = settings.DB.SCHEMA
# Note: column_0_N_name expands to include all columns in multi-column constraints
# e.g., "workspace_id_tenant_id" for a composite constraint on both columns
meta = MetaData(naming_convention=convention)
meta.schema = table_schema
Base = declarative_base(metadata=meta)


async def init_db():
    """Initialize the database using Alembic migrations"""
    from alembic import command
    from alembic.config import Config

    async with engine.connect() as connection:
        # Create schema if it doesn't exist
        await connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{table_schema}"'))
        # Install pgvector extension if it doesn't exist
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.commit()

    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")

    if settings.DB.AUTH_METHOD == "iam":
        from src.aws_auth import generate_rds_auth_token

        try:
            token = generate_rds_auth_token(
                region=settings.DB.AWS_REGION,
                hostname=settings.DB.RDS_HOSTNAME,
                port=settings.DB.RDS_PORT,
                username=settings.DB.RDS_USERNAME,
                profile=settings.DB.AWS_PROFILE,
            )
        except RuntimeError:
            logger.error(
                "Failed to generate IAM auth token for Alembic migration "
                "(region=%s, hostname=%s, username=%s)",
                settings.DB.AWS_REGION,
                settings.DB.RDS_HOSTNAME,
                settings.DB.RDS_USERNAME,
            )
            raise

        # URL-encode the token since it contains special characters
        encoded_token = quote_plus(token)

        # Parse original URI and overlay IAM credentials, preserving path/query
        _alembic_parsed = _parse_base_uri(settings.DB.CONNECTION_URI)
        _alembic_path = _alembic_parsed.path or "/postgres"

        # Build SSL query params, merging with any existing query
        ssl_params = "sslmode=require"
        if settings.DB.RDS_SSL_CA_BUNDLE:
            ssl_params += f"&sslrootcert={quote_plus(settings.DB.RDS_SSL_CA_BUNDLE)}"
        existing_query = _alembic_parsed.query
        merged_query = f"{existing_query}&{ssl_params}" if existing_query else ssl_params

        iam_uri = urlunsplit((
            _alembic_parsed.scheme or "postgresql+psycopg",
            f"{settings.DB.RDS_USERNAME}:{encoded_token}@{settings.DB.RDS_HOSTNAME}:{settings.DB.RDS_PORT}",
            _alembic_path,
            merged_query,
            "",
        ))

        alembic_cfg.set_main_option("sqlalchemy.url", iam_uri)

    command.upgrade(alembic_cfg, "head")
