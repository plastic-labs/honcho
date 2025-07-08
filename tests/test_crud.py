from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, schemas


@pytest.mark.asyncio
async def test_get_or_create_workspace_creates_new_workspace(db_session: AsyncSession):
    """Test that get_or_create_workspace creates a new workspace when it doesn't exist."""
    workspace_name = "test_workspace_new"
    metadata = {"key": "value"}
    configuration = {"feature": True}

    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata=metadata, configuration=configuration
    )

    # Call the function that contains line 65
    result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Verify workspace was created with correct values
    assert result.name == workspace_name
    assert result.h_metadata == metadata
    assert result.configuration == configuration
    assert result.id is not None
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_get_or_create_workspace_creates_new_workspace_with_minimal_data(
    db_session: AsyncSession,
):
    """Test creating a workspace with minimal required data."""
    workspace_name = "test_workspace_minimal"

    workspace_create = schemas.WorkspaceCreate(name=workspace_name)

    # Call the function that contains line 65
    result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Verify workspace was created with default values
    assert result.name == workspace_name
    assert result.h_metadata == {}  # default empty dict
    assert result.configuration == {}  # default empty dict
    assert result.id is not None
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_get_or_create_workspace_creates_new_workspace_with_empty_metadata(
    db_session: AsyncSession,
):
    """Test creating a workspace with explicitly empty metadata and configuration."""
    workspace_name = "test_workspace_empty"

    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata={}, configuration={}
    )

    # Call the function that contains line 65
    result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Verify workspace was created
    assert result.name == workspace_name
    assert result.h_metadata == {}
    assert result.configuration == {}
    assert result.id is not None
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_get_or_create_workspace_creates_new_workspace_with_complex_data(
    db_session: AsyncSession,
):
    """Test creating a workspace with complex metadata and configuration."""
    workspace_name = "test_workspace_complex"
    metadata = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "number": 42,
        "boolean": True,
    }
    configuration = {
        "api_version": "v2",
        "features": {"experimental": True, "beta_features": ["feature1", "feature2"]},
    }

    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata=metadata, configuration=configuration
    )

    # Call the function that contains line 65
    result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Verify workspace was created with complex data
    assert result.name == workspace_name
    assert result.h_metadata == metadata
    assert result.configuration == configuration
    assert result.id is not None
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_get_or_create_workspace_returns_existing_workspace(
    db_session: AsyncSession,
):
    """Test that get_or_create_workspace returns existing workspace when it already exists."""
    workspace_name = "test_workspace_existing"
    metadata = {"existing": "data"}
    configuration = {"existing": "config"}

    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata=metadata, configuration=configuration
    )

    # Create workspace first time
    first_result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Create workspace second time - should return existing
    second_result = await crud.get_or_create_workspace(db_session, workspace_create)

    # Verify same workspace is returned
    assert first_result.id == second_result.id
    assert first_result.name == second_result.name
    assert first_result.created_at == second_result.created_at
    assert first_result.h_metadata == second_result.h_metadata
    assert first_result.configuration == second_result.configuration


@pytest.mark.asyncio
async def test_get_or_create_workspace_database_persistence(db_session: AsyncSession):
    """Test that get_or_create_workspace properly persists to database - covers lines 70-73."""
    workspace_name = "test_workspace_persistence"
    metadata = {"test": "data"}
    configuration = {"test": "config"}

    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata=metadata, configuration=configuration
    )

    # Mock the database session to verify db.add and db.commit are called
    with (
        patch.object(db_session, "add") as mock_add,
        patch.object(db_session, "commit") as mock_commit,
    ):
        # Make commit async
        mock_commit.return_value = AsyncMock()

        # Call the function
        result = await crud.get_or_create_workspace(db_session, workspace_create)

        # Verify db.add was called (line 70)
        mock_add.assert_called_once()

        # Verify db.commit was called (line 71)
        mock_commit.assert_called_once()

        # Verify the workspace was returned (line 73)
        assert result.name == workspace_name
        assert result.h_metadata == metadata
        assert result.configuration == configuration


@pytest.mark.asyncio
async def test_get_or_create_workspace_logging_success(db_session: AsyncSession):
    """Test that get_or_create_workspace logs success message - covers line 72."""
    workspace_name = "test_workspace_logging"

    workspace_create = schemas.WorkspaceCreate(name=workspace_name)

    # Mock the logger to verify the success message is logged
    with patch("src.crud.logger") as mock_logger:
        # Call the function
        result = await crud.get_or_create_workspace(db_session, workspace_create)

        # Verify the success log message was called (line 72)
        mock_logger.info.assert_called_with(
            f"Workspace created successfully: {workspace_name}"
        )

        # Verify the workspace was returned
        assert result.name == workspace_name


@pytest.mark.asyncio
async def test_get_or_create_workspace_commit_and_return_sequence(
    db_session: AsyncSession,
):
    """Test the complete sequence: add -> commit -> log -> return - covers lines 70-73."""
    workspace_name = "test_workspace_sequence"
    metadata = {"sequence": "test"}

    workspace_create = schemas.WorkspaceCreate(name=workspace_name, metadata=metadata)

    # Mock the database session and logger to track call sequence
    with (
        patch.object(db_session, "add") as mock_add,
        patch.object(db_session, "commit") as mock_commit,
        patch("src.crud.logger") as mock_logger,
    ):
        # Make commit async
        mock_commit.return_value = AsyncMock()

        # Call the function
        result = await crud.get_or_create_workspace(db_session, workspace_create)

        # Verify all operations were called in sequence
        mock_add.assert_called_once()
        mock_commit.assert_called_once()
        mock_logger.info.assert_called_once_with(
            f"Workspace created successfully: {workspace_name}"
        )

        # Verify the workspace object was returned (line 73)
        assert result is not None
        assert result.name == workspace_name
        assert result.h_metadata == metadata


@pytest.mark.asyncio
async def test_update_workspace_with_metadata_not_none(db_session: AsyncSession):
    """Test update_workspace when metadata is not None - covers lines 114-115."""
    workspace_name = "test_workspace_update_metadata"
    metadata = {"key": "value", "updated": True}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with metadata not None
    workspace_update = schemas.WorkspaceUpdate(metadata=metadata)

    # Call update_workspace function that contains lines 114-115
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify metadata was set (line 115)
    assert result.h_metadata == metadata
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_metadata_none(db_session: AsyncSession):
    """Test update_workspace when metadata is None - ensures line 114 condition is false."""
    workspace_name = "test_workspace_update_no_metadata"

    # First create a workspace with initial metadata
    initial_metadata = {"initial": "data"}
    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, metadata=initial_metadata
    )
    _original_workspace = await crud.get_or_create_workspace(
        db_session, workspace_create
    )

    # Update with metadata as None (should not trigger lines 114-115)
    workspace_update = schemas.WorkspaceUpdate(metadata=None)

    # Call update_workspace function
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify original metadata is preserved since line 115 was not executed
    # get_or_create_workspace returns existing workspace, so metadata should be unchanged
    assert result.h_metadata == initial_metadata
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_complex_metadata(db_session: AsyncSession):
    """Test update_workspace with complex metadata structure - covers lines 114-115."""
    workspace_name = "test_workspace_update_complex"
    complex_metadata = {
        "nested": {"deep": {"value": "test"}},
        "array": [1, 2, 3],
        "boolean": True,
        "null_value": None,
        "number": 42.5,
    }

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with complex metadata
    workspace_update = schemas.WorkspaceUpdate(metadata=complex_metadata)

    # Call update_workspace function that contains lines 114-115
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify complex metadata was set correctly (line 115)
    assert result.h_metadata == complex_metadata
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_configuration_not_none(db_session: AsyncSession):
    """Test update_workspace when configuration is not None - covers lines 117-118."""
    workspace_name = "test_workspace_update_configuration"
    configuration = {"api_version": "v2", "feature_flag": True, "timeout": 30}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with configuration not None
    workspace_update = schemas.WorkspaceUpdate(configuration=configuration)

    # Call update_workspace function that contains lines 117-118
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify configuration was set (line 118)
    assert result.configuration == configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_configuration_none(db_session: AsyncSession):
    """Test update_workspace when configuration is None - ensures line 117 condition is false."""
    workspace_name = "test_workspace_update_no_configuration"

    # First create a workspace with initial configuration
    initial_configuration = {"initial": "config"}
    workspace_create = schemas.WorkspaceCreate(
        name=workspace_name, configuration=initial_configuration
    )
    _original_workspace = await crud.get_or_create_workspace(
        db_session, workspace_create
    )

    # Update with configuration as None (should not trigger lines 117-118)
    workspace_update = schemas.WorkspaceUpdate(configuration=None)

    # Call update_workspace function
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify original configuration is preserved since line 118 was not executed
    # get_or_create_workspace returns existing workspace, so configuration should be unchanged
    assert result.configuration == initial_configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_complex_configuration(db_session: AsyncSession):
    """Test update_workspace with complex configuration structure - covers lines 117-118."""
    workspace_name = "test_workspace_update_complex_config"
    complex_configuration = {
        "api_settings": {"version": "v2", "timeout": 5000},
        "features": {"experimental": True, "beta_access": ["feature1", "feature2"]},
        "limits": {"max_requests": 1000, "rate_limit": 100.5},
        "debug": False,
        "nested_config": {"deep": {"level": {"setting": "value"}}},
    }

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with complex configuration
    workspace_update = schemas.WorkspaceUpdate(configuration=complex_configuration)

    # Call update_workspace function that contains lines 117-118
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify complex configuration was set correctly (line 118)
    assert result.configuration == complex_configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_both_metadata_and_configuration(
    db_session: AsyncSession,
):
    """Test update_workspace with both metadata and configuration not None - covers lines 114-115 and 117-118."""
    workspace_name = "test_workspace_update_both"
    metadata = {"user": "test", "environment": "staging"}
    configuration = {"debug": True, "log_level": "INFO"}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with both metadata and configuration not None
    workspace_update = schemas.WorkspaceUpdate(
        metadata=metadata, configuration=configuration
    )

    # Call update_workspace function that contains both line sets
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify both metadata and configuration were set (lines 115 and 118)
    assert result.h_metadata == metadata
    assert result.configuration == configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_with_empty_configuration(db_session: AsyncSession):
    """Test update_workspace with empty configuration dict - covers lines 117-118."""
    workspace_name = "test_workspace_update_empty_config"
    empty_configuration = {}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with empty configuration (still not None, so should trigger lines 117-118)
    workspace_update = schemas.WorkspaceUpdate(configuration=empty_configuration)

    # Call update_workspace function that contains lines 117-118
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify empty configuration was set (line 118)
    assert result.configuration == empty_configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_update_workspace_commit_log_return_sequence(db_session: AsyncSession):
    """Test update_workspace database commit, logging, and return sequence - covers lines 120-122."""
    workspace_name = "test_workspace_commit_log_return"
    metadata = {"test": "commit_sequence"}
    configuration = {"test": "config_sequence"}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Mock the database session commit and logger to verify they're called
    with (
        patch.object(db_session, "commit") as mock_commit,
        patch("src.crud.logger") as mock_logger,
    ):
        # Make commit async
        mock_commit.return_value = AsyncMock()

        # Update workspace with both metadata and configuration
        workspace_update = schemas.WorkspaceUpdate(
            metadata=metadata, configuration=configuration
        )

        # Call update_workspace function that contains lines 120-122
        result = await crud.update_workspace(
            db_session, workspace_name, workspace_update
        )

        # Verify db.commit was called (line 120)
        mock_commit.assert_called_once()

        # Verify the success log message was called (line 121)
        mock_logger.info.assert_called_once_with(
            f"Workspace with id {result.id} updated successfully"
        )

        # Verify the workspace object was returned (line 122)
        assert result is not None
        assert result.name == workspace_name
        assert result.h_metadata == metadata
        assert result.configuration == configuration
        assert result.id is not None


@pytest.mark.asyncio
async def test_get_or_create_peers_updates_existing_peer_metadata(
    db_session: AsyncSession,
):
    """Test that get_or_create_peers updates existing peer metadata when metadata is not None - covers line 165."""
    workspace_name = "test_workspace"
    peer_name = "test_peer"
    initial_metadata = {"initial": "data"}
    updated_metadata = {"updated": "data", "key": "value"}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create initial peer with metadata
    initial_peer_create = schemas.PeerCreate(name=peer_name, metadata=initial_metadata)
    initial_peers = await crud.get_or_create_peers(
        db_session, workspace_name, [initial_peer_create]
    )

    # Verify initial peer was created
    assert len(initial_peers) == 1
    assert initial_peers[0].name == peer_name
    assert initial_peers[0].h_metadata == initial_metadata

    # Update existing peer with new metadata (this should trigger line 165)
    updated_peer_create = schemas.PeerCreate(name=peer_name, metadata=updated_metadata)
    updated_peers = await crud.get_or_create_peers(
        db_session, workspace_name, [updated_peer_create]
    )

    # Verify the existing peer's metadata was updated (line 165: existing_peer.h_metadata = peer_schema.metadata)
    assert len(updated_peers) == 1
    assert updated_peers[0].name == peer_name
    assert updated_peers[0].h_metadata == updated_metadata
    assert updated_peers[0].id == initial_peers[0].id  # Same peer, just updated


@pytest.mark.asyncio
async def test_get_all_workspaces_no_filters():
    """Test get_all_workspaces with no filters - covers lines 86-89."""
    # Call the function with no filters
    stmt = await crud.get_all_workspaces()

    # Verify the statement is built correctly
    # Line 86: stmt = select(models.Workspace)
    # Line 87: stmt = apply_filter(stmt, models.Workspace, filters) with filters=None
    # Line 88: stmt = stmt.order_by(models.Workspace.created_at)
    # Line 89: return stmt

    # Check the statement structure
    assert stmt is not None
    # Verify it returns a Select statement
    from sqlalchemy import Select

    assert isinstance(stmt, Select)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_filters():
    """Test get_all_workspaces with filters - covers lines 86-89."""
    # Test with metadata filters
    filters = {"metadata": {"type": "test"}}

    # Call the function with filters
    stmt = await crud.get_all_workspaces(filters=filters)

    # Verify the statement is built correctly
    # Line 86: stmt = select(models.Workspace)
    # Line 87: stmt = apply_filter(stmt, models.Workspace, filters)
    # Line 88: stmt = stmt.order_by(models.Workspace.created_at)
    # Line 89: return stmt

    # Check the statement structure
    assert stmt is not None
    from sqlalchemy import Select

    assert isinstance(stmt, Select)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_empty_filters():
    """Test get_all_workspaces with empty filters dictionary - covers lines 86-89."""
    # Test with empty filters dict
    filters = {}

    # Call the function with empty filters
    stmt = await crud.get_all_workspaces(filters=filters)

    # Verify the statement is built correctly
    # Line 86: stmt = select(models.Workspace)
    # Line 87: stmt = apply_filter(stmt, models.Workspace, filters) with empty dict
    # Line 88: stmt = stmt.order_by(models.Workspace.created_at)
    # Line 89: return stmt

    # Check the statement structure
    assert stmt is not None
    from sqlalchemy import Select

    assert isinstance(stmt, Select)


@pytest.mark.asyncio
async def test_get_all_workspaces_with_complex_filters():
    """Test get_all_workspaces with complex filters - covers lines 86-89."""
    # Test with complex filters including logical operators
    filters = {
        "AND": [
            {"metadata": {"environment": "production"}},
            {"created_at": {"gte": "2024-01-01"}},
        ]
    }

    # Call the function with complex filters
    stmt = await crud.get_all_workspaces(filters=filters)

    # Verify the statement is built correctly
    # Line 86: stmt = select(models.Workspace)
    # Line 87: stmt = apply_filter(stmt, models.Workspace, filters)
    # Line 88: stmt = stmt.order_by(models.Workspace.created_at)
    # Line 89: return stmt

    # Check the statement structure
    assert stmt is not None
    from sqlalchemy import Select

    assert isinstance(stmt, Select)


@pytest.mark.asyncio
async def test_update_workspace_metadata_assignment_line_118(db_session: AsyncSession):
    """Test update_workspace metadata assignment - covers line 118 specifically."""
    workspace_name = "test_workspace_line_118"
    test_metadata = {"specific": "line_118_test", "coverage": True}
    test_configuration = {"api_version": "v3", "line_118_coverage": True}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Update with both metadata and configuration to ensure both lines 115 and 118 are hit
    workspace_update = schemas.WorkspaceUpdate(
        metadata=test_metadata, configuration=test_configuration
    )

    # Call update_workspace function - this should cover line 118
    result = await crud.update_workspace(db_session, workspace_name, workspace_update)

    # Verify both metadata (line 115) and configuration (line 118) were set
    assert result.h_metadata == test_metadata
    assert result.configuration == test_configuration
    assert result.name == workspace_name


@pytest.mark.asyncio
async def test_get_or_create_peers_extract_names_and_query_construction(
    db_session: AsyncSession,
):
    """Test get_or_create_peers extracts peer names and constructs query - covers lines 147-148."""
    workspace_name = "test_workspace_peer_names"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create multiple peers with different names to test name extraction (line 147)
    peer_names = ["peer1", "peer2", "peer3"]
    peers_to_create = [
        schemas.PeerCreate(name=name, metadata={"test": f"data_{name}"})
        for name in peer_names
    ]

    # Call get_or_create_peers - this should execute lines 147-148
    # Line 147: peer_names = [p.name for p in peers]
    # Line 148: stmt = (select(models.Peer)...
    result_peers = await crud.get_or_create_peers(
        db_session, workspace_name, peers_to_create
    )

    # Verify the function executed correctly and extracted names properly
    assert len(result_peers) == 3
    result_names = {peer.name for peer in result_peers}
    expected_names = set(peer_names)
    assert result_names == expected_names

    # Verify all peers were created in the correct workspace
    for peer in result_peers:
        assert peer.workspace_name == workspace_name
        assert peer.name in peer_names
        assert peer.h_metadata is not None


@pytest.mark.asyncio
async def test_get_or_create_peers_single_peer_name_extraction(
    db_session: AsyncSession,
):
    """Test get_or_create_peers with single peer for name extraction - covers line 147."""
    workspace_name = "test_workspace_single_peer"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create a single peer to test name extraction from single-item list
    single_peer = [schemas.PeerCreate(name="single_peer", metadata={"type": "test"})]

    # Call get_or_create_peers - this should execute line 147 with single item
    # Line 147: peer_names = [p.name for p in peers] -> ["single_peer"]
    result_peers = await crud.get_or_create_peers(
        db_session, workspace_name, single_peer
    )

    # Verify the function handled single peer name extraction correctly
    assert len(result_peers) == 1
    assert result_peers[0].name == "single_peer"
    assert result_peers[0].workspace_name == workspace_name


@pytest.mark.asyncio
async def test_get_or_create_peers_empty_list_name_extraction(
    db_session: AsyncSession,
):
    """Test get_or_create_peers with empty peer list for name extraction - covers line 147."""
    workspace_name = "test_workspace_empty_peers"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Call get_or_create_peers with empty list
    # Line 147: peer_names = [p.name for p in peers] -> []
    result_peers = await crud.get_or_create_peers(db_session, workspace_name, [])

    # Verify the function handled empty list correctly
    assert len(result_peers) == 0


@pytest.mark.asyncio
async def test_get_or_create_peers_duplicate_names_extraction(
    db_session: AsyncSession,
):
    """Test get_or_create_peers with duplicate peer names for name extraction - covers line 147."""
    workspace_name = "test_workspace_duplicate_names"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # First create one peer
    first_peer = [
        schemas.PeerCreate(name="duplicate_peer", metadata={"instance": "first"})
    ]
    await crud.get_or_create_peers(db_session, workspace_name, first_peer)

    # Now try to create duplicate peers - the function should handle this by updating existing
    duplicate_peers = [
        schemas.PeerCreate(name="duplicate_peer", metadata={"instance": "updated"}),
        schemas.PeerCreate(name="unique_peer", metadata={"instance": "unique"}),
    ]

    # Call get_or_create_peers - this should execute line 147 with duplicates
    # Line 147: peer_names = [p.name for p in peers] -> ["duplicate_peer", "unique_peer"]
    result_peers = await crud.get_or_create_peers(
        db_session, workspace_name, duplicate_peers
    )

    # Verify the function handled duplicate names correctly
    # Should return exactly 2 peers (one updated duplicate_peer, one new unique_peer)
    assert len(result_peers) == 2
    result_names = {peer.name for peer in result_peers}
    assert result_names == {"duplicate_peer", "unique_peer"}

    # Verify the duplicate peer was updated with new metadata
    duplicate_peer = next(p for p in result_peers if p.name == "duplicate_peer")
    assert duplicate_peer.h_metadata == {"instance": "updated"}


@pytest.mark.asyncio
async def test_get_or_create_peers_complex_names_extraction(
    db_session: AsyncSession,
):
    """Test get_or_create_peers with valid complex peer names for extraction - covers line 147."""
    workspace_name = "test_workspace_complex_names"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create peers with valid complex names (alphanumeric, dashes, underscores only)
    complex_names = [
        "peer-with-dashes",
        "peer_with_underscores",
        "UPPERCASE_PEER",
        "peer123numbers",
        "MiXeD_CaSe-PeEr123",
        "peer-123_TEST",
        "a1b2c3d4e5f6",
    ]

    complex_peers = [
        schemas.PeerCreate(name=name, metadata={"name_type": "complex"})
        for name in complex_names
    ]

    # Call get_or_create_peers - this should execute line 147 with complex names
    # Line 147: peer_names = [p.name for p in peers] -> list of complex names
    result_peers = await crud.get_or_create_peers(
        db_session, workspace_name, complex_peers
    )

    # Verify the function extracted and handled complex names correctly
    assert len(result_peers) == len(complex_names)
    result_names = {peer.name for peer in result_peers}
    expected_names = set(complex_names)
    assert result_names == expected_names

    # Verify all peers were created with correct names and workspace
    for peer in result_peers:
        assert peer.name in complex_names
        assert peer.workspace_name == workspace_name


@pytest.mark.asyncio
async def test_peer_repr_method(db_session: AsyncSession):
    """Test Peer model __repr__ method - covers line 134 in src/models.py."""
    workspace_name = "test_workspace_peer_repr"
    peer_name = "test_peer_repr"
    metadata = {"test": "repr_method", "number": 42}
    configuration = {"feature": True, "debug": False}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create a peer with specific metadata and configuration to test __repr__
    peer_create = schemas.PeerCreate(
        name=peer_name, metadata=metadata, configuration=configuration
    )
    peers = await crud.get_or_create_peers(db_session, workspace_name, [peer_create])

    # Get the created peer
    assert len(peers) == 1
    peer = peers[0]

    # Call the __repr__ method (line 134) and verify the output format
    repr_string = repr(peer)

    # Verify the __repr__ output contains all expected components
    assert repr_string.startswith("Peer(")
    assert repr_string.endswith(")")
    assert f"id={peer.id}" in repr_string
    assert f"name={peer.name}" in repr_string
    assert f"workspace_name={peer.workspace_name}" in repr_string
    assert f"created_at={peer.created_at}" in repr_string
    assert f"h_metadata={peer.h_metadata}" in repr_string
    assert f"configuration={peer.configuration}" in repr_string

    # Verify the specific values are correctly represented
    assert peer_name in repr_string
    assert workspace_name in repr_string
    assert str(metadata) in repr_string
    assert str(configuration) in repr_string


@pytest.mark.asyncio
async def test_session_repr_method(db_session: AsyncSession):
    """Test Session model __repr__ method - covers line 168 in src/models.py."""
    workspace_name = "test_workspace_session_repr"
    session_name = "test_session_repr"
    metadata = {"test": "session_repr_method", "session_type": "test"}

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create a session with specific metadata to test __repr__
    session_create = schemas.SessionCreate(name=session_name, metadata=metadata)
    session = await crud.get_or_create_session(db_session, session_create, workspace_name)

    # Call the __repr__ method (line 168) and verify the output format
    repr_string = repr(session)

    # Verify the __repr__ output contains all expected components
    assert repr_string.startswith("Session(")
    assert repr_string.endswith(")")
    assert f"id={session.id}" in repr_string
    assert f"name={session.name}" in repr_string
    assert f"workspace_name={session.workspace_name}" in repr_string
    assert f"is_active={session.is_active}" in repr_string
    assert f"created_at={session.created_at}" in repr_string
    assert f"h_metadata={session.h_metadata}" in repr_string

    # Verify the specific values are correctly represented
    assert session_name in repr_string
    assert workspace_name in repr_string
    assert str(metadata) in repr_string
    assert str(session.is_active) in repr_string


@pytest.mark.asyncio
async def test_message_repr_method(db_session: AsyncSession):
    """Test Message model __repr__ method - covers line 227 in src/models.py."""
    workspace_name = "test_workspace_message_repr"
    session_name = "test_session_message_repr"
    peer_name = "test_peer_message_repr"
    message_content = "Test message content for __repr__ method testing"

    # First create a workspace
    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    await crud.get_or_create_workspace(db_session, workspace_create)

    # Create a session
    session_create = schemas.SessionCreate(name=session_name)
    await crud.get_or_create_session(db_session, session_create, workspace_name)

    # Create a peer
    peer_create = schemas.PeerCreate(name=peer_name)
    peers = await crud.get_or_create_peers(db_session, workspace_name, [peer_create])
    peer = peers[0]

    # Create a message with specific content to test __repr__
    message_create = schemas.MessageCreate(
        content=message_content,
        peer_id=peer.name,
        metadata={"test": "message_repr_method"}
    )
    messages = await crud.create_messages(
        db_session, [message_create], workspace_name, session_name
    )

    # Get the created message
    assert len(messages) == 1
    message = messages[0]

    # Call the __repr__ method (line 227) and verify the output format
    repr_string = repr(message)

    # Verify the __repr__ output contains all expected components based on line 227:
    # return f"Message(id={self.id}, session_name={self.session_name}, peer_name={self.peer_name}, content={self.content})"
    assert repr_string.startswith("Message(")
    assert repr_string.endswith(")")
    assert f"id={message.id}" in repr_string
    assert f"session_name={message.session_name}" in repr_string
    assert f"peer_name={message.peer_name}" in repr_string
    assert f"content={message.content}" in repr_string

    # Verify the specific values are correctly represented
    assert str(message.id) in repr_string
    assert session_name in repr_string
    assert peer_name in repr_string
    assert message_content in repr_string
