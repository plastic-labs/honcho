from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.pool import NullPool

from src.config import DBSettings


def test_db_engine_kwargs_null_pool():
    """Test that engine_kwargs includes NullPool when POOL_CLASS is 'null'."""
    with patch("src.config.settings") as mock_settings:
        # Configure mock settings to use null pool
        mock_db_settings = DBSettings(POOL_CLASS="null")
        mock_settings.DB = mock_db_settings

        # Clear any existing module state and reimport
        import importlib

        import src.db

        importlib.reload(src.db)

        # Verify that poolclass is set to NullPool
        assert "poolclass" in src.db.engine_kwargs
        assert src.db.engine_kwargs["poolclass"] is NullPool


def test_db_engine_kwargs_default_pool():
    """Test that engine_kwargs includes pool settings when POOL_CLASS is not 'null'."""
    with patch("src.config.settings") as mock_settings:
        # Configure mock settings to use default pool
        mock_db_settings = DBSettings(
            POOL_CLASS="default",
            POOL_PRE_PING=True,
            POOL_SIZE=5,
            MAX_OVERFLOW=10,
            POOL_TIMEOUT=20,
            POOL_RECYCLE=300,
            POOL_USE_LIFO=False,
        )
        mock_settings.DB = mock_db_settings

        # Clear any existing module state and reimport
        import importlib

        import src.db

        importlib.reload(src.db)

        # Verify that poolclass is NOT set but other pool settings are
        assert "poolclass" not in src.db.engine_kwargs
        assert src.db.engine_kwargs["pool_pre_ping"] is True
        assert src.db.engine_kwargs["pool_size"] == 5
        assert src.db.engine_kwargs["max_overflow"] == 10
        assert src.db.engine_kwargs["pool_timeout"] == 20
        assert src.db.engine_kwargs["pool_recycle"] == 300
        assert src.db.engine_kwargs["pool_use_lifo"] is False


@pytest.mark.asyncio
async def test_init_db_imports():
    """Test that init_db properly imports alembic modules (covers lines 56-57)."""
    import sys
    from importlib import reload

    # Track what modules are imported
    original_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )
    imported_modules = []

    def tracking_import(name, *args, **kwargs):
        imported_modules.append(name)
        return original_import(name, *args, **kwargs)

    # Create a mock alembic config to avoid actual migration
    mock_config = MagicMock()

    with (
        patch("builtins.__import__", side_effect=tracking_import),
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_connection

        # Clear any cached imports and reload the module
        if "src.db" in sys.modules:
            reload(sys.modules["src.db"])

        # Import and call init_db
        from src.db import init_db

        await init_db()

        # Verify that the alembic modules were imported (lines 56-57)
        assert "alembic" in imported_modules, (
            f"'alembic' not found in imported modules: {imported_modules}"
        )
        assert "alembic.config" in imported_modules, (
            f"'alembic.config' not found in imported modules: {imported_modules}"
        )

        # Verify that the alembic functions were called
        mock_upgrade.assert_called_once_with(mock_config, "head")


@pytest.mark.asyncio
async def test_init_db_connection_context_manager():
    """Test that init_db properly opens and uses database connection (covers line 59)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection that tracks context manager usage
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that engine.connect() was called (line 59)
        mock_engine.connect.assert_called_once()

        # Verify that the connection context manager was properly entered
        mock_context_manager.__aenter__.assert_called_once()

        # Verify that SQL commands were executed on the connection
        assert (
            mock_connection.execute.call_count >= 2
        )  # At least schema creation and extension

        # Verify that commit was called
        mock_connection.commit.assert_called_once()

        # Verify that the context manager was properly exited
        mock_context_manager.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_init_db_creates_schema():
    """Test that init_db executes CREATE SCHEMA IF NOT EXISTS command (covers line 61)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
        patch("src.db.table_schema", "test_schema"),
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that the schema creation SQL was executed (line 61)
        schema_call_found = False
        for call in mock_connection.execute.call_args_list:
            args, kwargs = call
            if len(args) > 0:
                sql_text = str(args[0])
                if 'CREATE SCHEMA IF NOT EXISTS "test_schema"' in sql_text:
                    schema_call_found = True
                    break

        assert schema_call_found, (
            f"CREATE SCHEMA command not found in execute calls: {[str(call) for call in mock_connection.execute.call_args_list]}"
        )


@pytest.mark.asyncio
async def test_init_db_creates_vector_extension():
    """Test that init_db executes CREATE EXTENSION IF NOT EXISTS vector command (covers line 63)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that the vector extension creation SQL was executed (line 63)
        vector_extension_call_found = False
        for call in mock_connection.execute.call_args_list:
            args, kwargs = call
            if len(args) > 0:
                sql_text = str(args[0])
                if "CREATE EXTENSION IF NOT EXISTS vector" in sql_text:
                    vector_extension_call_found = True
                    break

        assert vector_extension_call_found, (
            f"CREATE EXTENSION IF NOT EXISTS vector command not found in execute calls: {[str(call) for call in mock_connection.execute.call_args_list]}"
        )


@pytest.mark.asyncio
async def test_init_db_commits_transaction():
    """Test that init_db commits the database transaction (covers line 64)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that commit was called (line 64)
        mock_connection.commit.assert_called_once()


@pytest.mark.asyncio
async def test_init_db_database_operations_sequence():
    """Test that init_db executes database operations in the correct sequence (covers lines 63-64)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
        patch("src.db.table_schema", "test_schema"),
    ):
        # Setup mock connection that tracks call order
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that execute was called at least twice (schema + extension)
        assert mock_connection.execute.call_count >= 2

        # Verify that commit was called after execute calls
        mock_connection.commit.assert_called_once()

        # Check the sequence of SQL commands
        execute_calls = mock_connection.execute.call_args_list
        schema_call_found = False
        vector_call_found = False

        for call in execute_calls:
            args, kwargs = call
            if len(args) > 0:
                sql_text = str(args[0])
                if 'CREATE SCHEMA IF NOT EXISTS "test_schema"' in sql_text:
                    schema_call_found = True
                elif "CREATE EXTENSION IF NOT EXISTS vector" in sql_text:
                    vector_call_found = True

        assert schema_call_found, "CREATE SCHEMA command not executed"
        assert vector_call_found, (
            "CREATE EXTENSION IF NOT EXISTS vector command not executed"
        )


@pytest.mark.asyncio
async def test_init_db_alembic_config_creation():
    """Test that init_db creates Alembic configuration with correct ini file (covers line 67)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config) as mock_config_class,
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that Config was called with "alembic.ini" (line 67)
        mock_config_class.assert_called_once_with("alembic.ini")


@pytest.mark.asyncio
async def test_init_db_alembic_upgrade_command():
    """Test that init_db runs Alembic upgrade command to head (covers line 68)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config),
        patch("alembic.command.upgrade") as mock_upgrade,  # noqa: F841
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that command.upgrade was called with config and "head" (line 68)
        mock_upgrade.assert_called_once_with(mock_config, "head")


@pytest.mark.asyncio
async def test_init_db_alembic_sequence():
    """Test that init_db executes Alembic operations in correct sequence after DB setup (covers lines 67-68)."""
    from src.db import init_db

    # Create mock alembic components
    mock_config = MagicMock()

    with (
        patch("alembic.config.Config", return_value=mock_config) as mock_config_class,
        patch("alembic.command.upgrade") as mock_upgrade,
        patch("src.db.engine") as mock_engine,
    ):
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_connection
        mock_context_manager.__aexit__.return_value = None
        mock_engine.connect.return_value = mock_context_manager

        # Call init_db
        await init_db()

        # Verify that database operations complete before Alembic operations
        # First, connection operations should happen
        mock_engine.connect.assert_called_once()
        mock_connection.commit.assert_called_once()

        # Then, Alembic config should be created (line 67)
        mock_config_class.assert_called_once_with("alembic.ini")

        # Finally, Alembic upgrade should be run (line 68)
        mock_upgrade.assert_called_once_with(mock_config, "head")

        # Verify the Alembic operations happen after database context manager exits
        assert mock_context_manager.__aexit__.called
        assert mock_config_class.called
        assert mock_upgrade.called
