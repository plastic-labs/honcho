# File upload tests for session endpoints
import io
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks, UploadFile
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import Headers

from src import models, schemas
from src.config import settings
from src.exceptions import ValidationException
from src.models import Peer, Workspace
from src.routers.messages import create_messages_with_file
from src.utils.files import ExtractedFileText


async def _create_test_session(
    db_session: AsyncSession, test_workspace: Workspace
) -> models.Session:
    """Helper function to create a test session"""
    test_session = models.Session(
        workspace_name=test_workspace.name, name=str(generate_nanoid())
    )
    db_session.add(test_session)
    await db_session.commit()
    return test_session


def _get_upload_url(workspace_name: str, session_name: str) -> str:
    """Helper function to get the session upload URL"""
    return f"/v3/workspaces/{workspace_name}/sessions/{session_name}/messages/upload"


@pytest.mark.asyncio
async def test_create_messages_with_text_file(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test creating messages with a text file upload"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock text file
    file_content = (
        "This is a test text file.\nIt has multiple lines.\nFor testing purposes."
    )
    file_data = io.BytesIO(file_content.encode("utf-8"))

    # Multipart form data - API accepts single file
    files = {"file": ("test.txt", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1  # Should be 1 message since text is short

    message = data[0]
    assert file_content in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name


@pytest.mark.asyncio
async def test_create_messages_with_large_file_chunking(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test that large files get split into multiple messages"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a large text file that will require chunking
    large_content = "This is a test line.\n" * 3000  # Should exceed 49500 chars
    file_data = io.BytesIO(large_content.encode("utf-8"))

    files = {"file": ("large_test.txt", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) > 1  # Should be multiple messages due to chunking

    # All messages should have the same peer_id and session_id
    for message in data:
        assert message["peer_id"] == test_peer.name
        assert message["session_id"] == session_name


@pytest.mark.asyncio
async def test_create_messages_with_json_file(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test creating messages with a JSON file upload"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock JSON file
    json_data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
    file_content = json.dumps(json_data, indent=2)
    file_data = io.BytesIO(file_content.encode("utf-8"))

    files = {"file": ("test.json", file_data, "application/json")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    message = data[0]
    assert '"name": "test"' in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name


@pytest.mark.asyncio
async def test_create_messages_with_empty_json_file(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test that empty JSON uploads do not crash and create empty content."""
    test_workspace, test_peer = sample_data

    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"")
    files = {"file": ("empty.json", file_data, "application/json")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == ""
    assert data[0]["peer_id"] == test_peer.name
    assert data[0]["session_id"] == session_name


@pytest.mark.asyncio
async def test_create_messages_with_unsupported_file_type(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test error handling for unsupported file types"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a file with unsupported type
    file_data = io.BytesIO(b"some binary data")
    files = {"file": ("test.exe", file_data, "application/x-executable")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 415


@pytest.mark.asyncio
async def test_create_message_missing_peer_id_session(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test error when peer_id is missing for session endpoint"""
    test_workspace, _test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Upload file but missing peer_id (required form field for session endpoint)
    file_data = io.BytesIO(b"test content")
    files = {"file": ("test.txt", file_data, "text/plain")}
    form_data: dict[str, Any] = {}  # Missing peer_id

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    # Session endpoint requires peer_id as form field
    assert (
        response.status_code == 422
    )  # FastAPI validation error for missing required field


@pytest.mark.asyncio
async def test_create_messages_with_empty_file(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test handling of empty files"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Empty file
    file_data = io.BytesIO(b"")
    files = {"file": ("empty.txt", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    # Should create one message with empty content
    assert len(data) == 1
    assert data[0]["content"] == ""


@pytest.mark.asyncio
async def test_file_metadata_stored_in_internal_metadata(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test that file metadata is stored in internal_metadata (database check)"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"test file content for internal metadata")
    files = {"file": ("internal_test.txt", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    message_id = data[0]["id"]

    # Check the database directly for internal_metadata
    stmt = select(models.Message).where(models.Message.public_id == message_id)
    result = await db_session.execute(stmt)
    db_message = result.scalar_one()

    # File metadata should be in internal_metadata
    assert "file_id" in db_message.internal_metadata
    assert "filename" in db_message.internal_metadata
    assert db_message.internal_metadata["filename"] == "internal_test.txt"
    assert db_message.internal_metadata["content_type"] == "text/plain"
    assert "chunk_index" in db_message.internal_metadata
    assert "total_chunks" in db_message.internal_metadata


@pytest.mark.asyncio
async def test_missing_file_parameter(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test error when no file is provided"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # No file provided
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, data=form_data)

    # Should return 422 for missing file
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_pdf_file_processing(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test creating messages with a PDF file upload"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a simple PDF file (this is a minimal PDF structure)
    # This minimal PDF contains: catalog, pages tree, single page, and content stream with "Test PDF content" text
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
    file_data = io.BytesIO(pdf_content)

    files = {"file": ("test.pdf", file_data, "application/pdf")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) >= 1  # PDF should create at least one message

    message = data[0]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name


@pytest.mark.asyncio
async def test_file_too_large_rejected(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test that files larger than MAX_FILE_SIZE are rejected"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a file larger than the configured max size
    max_size = settings.MAX_FILE_SIZE
    large_content = b"x" * (max_size + 1)  # 1 byte over the limit
    file_data = io.BytesIO(large_content)

    files = {"file": ("too_large.txt", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    # Should reject the file with 413 (Request Entity Too Large)
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_file_upload_with_metadata(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test file upload with metadata parameter"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock text file
    file_content = "Test file with metadata"
    file_data = io.BytesIO(file_content.encode("utf-8"))

    # Prepare metadata
    metadata = {"source": "test", "category": "upload", "priority": 1}

    files = {"file": ("test_metadata.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "metadata": json.dumps(metadata),
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    message = data[0]
    assert file_content in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name
    # Check that metadata was applied
    assert message["metadata"] == metadata


@pytest.mark.asyncio
async def test_file_upload_with_configuration(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test file upload with configuration parameter"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock text file
    file_content = "Test file with configuration"
    file_data = io.BytesIO(file_content.encode("utf-8"))

    # Prepare configuration
    configuration = {"skip_deriver": True, "custom_flag": "test"}

    files = {"file": ("test_config.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "configuration": json.dumps(configuration),
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    message = data[0]
    assert file_content in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name
    # Note: Configuration is used during processing, may not be directly stored
    # This test confirms the endpoint accepts it without error


@pytest.mark.asyncio
async def test_file_upload_with_created_at(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test file upload with created_at parameter"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock text file
    file_content = "Test file with created_at"
    file_data = io.BytesIO(file_content.encode("utf-8"))

    # Prepare created_at timestamp (ISO 8601 format)
    from datetime import datetime, timezone

    test_timestamp = datetime(2023, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
    created_at_str = test_timestamp.isoformat()

    files = {"file": ("test_timestamp.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "created_at": created_at_str,
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    message = data[0]
    assert file_content in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name
    # Check that created_at was applied (compare timestamps, allowing for small differences)
    message_timestamp = datetime.fromisoformat(
        message["created_at"].replace("Z", "+00:00")
    )
    assert abs((message_timestamp - test_timestamp).total_seconds()) < 1


@pytest.mark.asyncio
async def test_file_upload_with_all_parameters(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test file upload with metadata, configuration, and created_at all together"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a mock text file
    file_content = "Test file with all parameters"
    file_data = io.BytesIO(file_content.encode("utf-8"))

    # Prepare all parameters
    metadata = {"source": "comprehensive_test", "version": "1.0"}
    configuration = {"skip_deriver": False, "test_mode": True}
    from datetime import datetime, timezone

    test_timestamp = datetime(2023, 6, 20, 14, 15, 30, tzinfo=timezone.utc)
    created_at_str = test_timestamp.isoformat()

    files = {"file": ("test_all_params.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "metadata": json.dumps(metadata),
        "configuration": json.dumps(configuration),
        "created_at": created_at_str,
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1

    message = data[0]
    assert file_content in message["content"]
    assert message["peer_id"] == test_peer.name
    assert message["session_id"] == session_name
    # Check metadata
    assert message["metadata"] == metadata
    # Check created_at
    message_timestamp = datetime.fromisoformat(
        message["created_at"].replace("Z", "+00:00")
    )
    assert abs((message_timestamp - test_timestamp).total_seconds()) < 1


@pytest.mark.asyncio
async def test_file_upload_with_invalid_metadata_json(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test file upload with invalid JSON in metadata parameter"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_content = "Test file"
    file_data = io.BytesIO(file_content.encode("utf-8"))

    files = {"file": ("test.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "metadata": "invalid json {",  # Invalid JSON
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    # Should still succeed but metadata will be None (backend handles gracefully)
    assert response.status_code == 201
    data = response.json()
    # Metadata parsing failure is logged but doesn't fail the request
    assert len(data) == 1


@pytest.mark.asyncio
async def test_large_file_upload_with_metadata(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test that large files with metadata get split correctly and metadata is applied to all chunks"""
    test_workspace, test_peer = sample_data

    # Create session for session endpoint
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    # Create a large text file that will require chunking
    large_content = "This is a test line.\n" * 3000  # Should exceed 49500 chars
    file_data = io.BytesIO(large_content.encode("utf-8"))

    metadata = {"source": "chunked_test", "chunked": True}

    files = {"file": ("large_metadata.txt", file_data, "text/plain")}
    form_data = {
        "peer_id": test_peer.name,
        "metadata": json.dumps(metadata),
    }

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) > 1  # Should be multiple messages due to chunking

    # All messages should have the same metadata, peer_id and session_id
    for message in data:
        assert message["peer_id"] == test_peer.name
        assert message["session_id"] == session_name
        assert message["metadata"] == metadata


@pytest.mark.asyncio
async def test_create_messages_with_mp3_file(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Test creating messages with an MP3 upload."""
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"fake mp3 bytes")
    files = {"file": ("call.mp3", file_data, "audio/mpeg")}
    form_data = {"peer_id": test_peer.name}

    extracted = ExtractedFileText(
        text="First sentence.\nSecond sentence.",
        metadata={
            "processing_type": "audio_transcription",
            "audio_segment_count": 1,
            "transcription_provider": "openai",
        },
    )

    with (
        patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
        patch(
            "src.utils.files.AudioProcessor.extract_text",
            new=AsyncMock(return_value=extracted),
        ),
    ):
        url = _get_upload_url(test_workspace.name, session_name)
        response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == extracted.text
    assert data[0]["peer_id"] == test_peer.name
    assert data[0]["session_id"] == session_name

    stmt = select(models.Message).where(models.Message.public_id == data[0]["id"])
    result = await db_session.execute(stmt)
    db_message = result.scalar_one()
    assert db_message.internal_metadata["processing_type"] == "audio_transcription"
    assert db_message.internal_metadata["audio_segment_count"] == 1
    assert db_message.internal_metadata["transcription_provider"] == "openai"
    assert "transcription_fallback_used" not in db_message.internal_metadata


@pytest.mark.asyncio
async def test_audio_upload_over_generic_limit_uses_audio_size_limit_when_validated(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    original_generic_max = settings.MAX_FILE_SIZE
    original_audio_max = settings.AUDIO.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE = 5
    settings.AUDIO.MAX_FILE_SIZE_BYTES = 10
    extracted = ExtractedFileText(
        text="Transcribed audio",
        metadata={
            "processing_type": "audio_transcription",
            "audio_segment_count": 1,
            "transcription_provider": "openai",
        },
    )
    try:
        file_data = io.BytesIO(b"123456")
        files = {"file": ("call.mp3", file_data, "audio/mpeg")}
        form_data = {"peer_id": test_peer.name}

        with (
            patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
            patch(
                "src.routers.messages.is_validated_audio_upload",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "src.utils.files.AudioProcessor.extract_text",
                new=AsyncMock(return_value=extracted),
            ),
        ):
            url = _get_upload_url(test_workspace.name, session_name)
            response = client.post(url, files=files, data=form_data)
    finally:
        settings.MAX_FILE_SIZE = original_generic_max
        settings.AUDIO.MAX_FILE_SIZE_BYTES = original_audio_max

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == extracted.text


@pytest.mark.asyncio
async def test_extension_only_audio_upload_over_generic_limit_uses_audio_size_limit_when_validated(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    original_generic_max = settings.MAX_FILE_SIZE
    original_audio_max = settings.AUDIO.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE = 5
    settings.AUDIO.MAX_FILE_SIZE_BYTES = 10
    extracted = ExtractedFileText(
        text="Transcribed extension-only audio",
        metadata={
            "processing_type": "audio_transcription",
            "audio_segment_count": 1,
            "transcription_provider": "openai",
        },
    )
    try:
        file_data = io.BytesIO(b"123456")
        files = {"file": ("renamed.mp3", file_data, "application/octet-stream")}
        form_data = {"peer_id": test_peer.name}

        with (
            patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
            patch(
                "src.routers.messages.is_validated_audio_upload",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "src.utils.files.AudioProcessor.extract_text",
                new=AsyncMock(return_value=extracted),
            ),
        ):
            url = _get_upload_url(test_workspace.name, session_name)
            response = client.post(url, files=files, data=form_data)
    finally:
        settings.MAX_FILE_SIZE = original_generic_max
        settings.AUDIO.MAX_FILE_SIZE_BYTES = original_audio_max

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == extracted.text


@pytest.mark.asyncio
async def test_audio_upload_over_generic_limit_keeps_generic_limit_without_transcription_credentials(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    original_generic_max = settings.MAX_FILE_SIZE
    original_audio_max = settings.AUDIO.MAX_FILE_SIZE_BYTES
    settings.MAX_FILE_SIZE = 5
    settings.AUDIO.MAX_FILE_SIZE_BYTES = 10
    try:
        file_data = io.BytesIO(b"123456")
        files = {"file": ("call.mp3", file_data, "audio/mpeg")}
        form_data = {"peer_id": test_peer.name}

        with patch.dict("src.utils.files.CLIENTS", {}, clear=True):
            url = _get_upload_url(test_workspace.name, session_name)
            response = client.post(url, files=files, data=form_data)
    finally:
        settings.MAX_FILE_SIZE = original_generic_max
        settings.AUDIO.MAX_FILE_SIZE_BYTES = original_audio_max

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_create_messages_with_file_opens_tracked_db_after_file_processing():
    background_tasks = BackgroundTasks()
    form_data = schemas.MessageUploadCreate(peer_id="peer")
    file = UploadFile(
        file=io.BytesIO(b"ID3\x03\x00\x00"),
        filename="call.mp3",
        headers=Headers({"content-type": "audio/mpeg"}),
    )

    tracked_db_entered = False
    db_session = AsyncMock()
    created_message = models.Message(
        session_name="session",
        peer_name="peer",
        workspace_name="workspace",
        content="transcribed text",
        public_id=generate_nanoid(),
        token_count=2,
        seq_in_session=1,
        created_at=datetime.now(UTC),
        h_metadata={},
        internal_metadata={},
    )

    @asynccontextmanager
    async def fake_tracked_db(_operation_name: str | None = None):
        nonlocal tracked_db_entered
        tracked_db_entered = True
        yield db_session

    async def fake_process_file_uploads_for_messages(*_args: Any, **_kwargs: Any):
        assert not tracked_db_entered
        return [
            {
                "message_create": schemas.MessageCreate(
                    content="transcribed text",
                    peer_id="peer",
                ),
                "file_metadata": {"processing_type": "audio_transcription"},
            }
        ]

    with (
        patch(
            "src.routers.messages.process_file_uploads_for_messages",
            side_effect=fake_process_file_uploads_for_messages,
        ),
        patch("src.routers.messages.tracked_db", fake_tracked_db),
        patch(
            "src.routers.messages.crud.create_messages",
            new=AsyncMock(return_value=[created_message]),
        ),
        patch("src.routers.messages.flag_modified"),
    ):
        result = await create_messages_with_file(
            background_tasks=background_tasks,
            workspace_id="workspace",
            session_id="session",
            form_data=form_data,
            file=file,
        )

    assert tracked_db_entered
    db_session.commit.assert_awaited_once()
    assert result == [created_message]


@pytest.mark.asyncio
async def test_filename_audio_extension_with_text_plain_mime_uses_text_processor(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"plain text, not audio")
    files = {"file": ("notes.mp3", file_data, "text/plain")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "plain text, not audio"


@pytest.mark.asyncio
async def test_create_messages_with_wav_file_accepts_audio_wave_mime(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"fake wav bytes")
    files = {"file": ("call.wav", file_data, "audio/wave")}
    form_data = {"peer_id": test_peer.name}

    extracted = ExtractedFileText(
        text="WAV transcript",
        metadata={
            "processing_type": "audio_transcription",
            "audio_segment_count": 1,
            "transcription_provider": "openai",
        },
    )

    with (
        patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
        patch(
            "src.utils.files.AudioProcessor.extract_text",
            new=AsyncMock(return_value=extracted),
        ),
    ):
        url = _get_upload_url(test_workspace.name, session_name)
        response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "WAV transcript"


@pytest.mark.asyncio
async def test_malformed_audio_upload_returns_validation_error(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"not-valid-audio")
    files = {"file": ("broken.mp3", file_data, "audio/mpeg")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    with (
        patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
        patch(
            "src.utils.files.AudioProcessor._probe_audio_duration_seconds",
            side_effect=ValidationException("Uploaded audio is invalid or unreadable"),
        ),
    ):
        response = client.post(url, files=files, data=form_data)

    assert response.status_code == 422
    assert "Uploaded audio is invalid or unreadable" in response.json()["detail"]


@pytest.mark.asyncio
async def test_empty_audio_upload_returns_validation_error(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"")
    files = {"file": ("empty.mp3", file_data, "audio/mpeg")}
    form_data = {"peer_id": test_peer.name}

    url = _get_upload_url(test_workspace.name, session_name)
    with patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False):
        response = client.post(url, files=files, data=form_data)

    assert response.status_code == 422
    assert "Audio upload is empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_large_audio_upload_applies_audio_metadata_to_all_message_chunks(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
):
    """Large audio transcripts should chunk into multiple messages with shared audio metadata."""
    test_workspace, test_peer = sample_data
    test_session = await _create_test_session(db_session, test_workspace)
    session_name = test_session.name

    file_data = io.BytesIO(b"fake long mp3 bytes")
    files = {"file": ("lecture.mp3", file_data, "audio/mpeg")}
    form_data = {"peer_id": test_peer.name}

    long_text = "Segment line. " * 4000
    extracted = ExtractedFileText(
        text=long_text,
        metadata={
            "processing_type": "audio_transcription",
            "audio_segment_count": 3,
            "transcription_provider": "openai",
        },
    )

    with (
        patch.dict("src.utils.files.CLIENTS", {"openai": object()}, clear=False),
        patch(
            "src.utils.files.AudioProcessor.extract_text",
            new=AsyncMock(return_value=extracted),
        ),
    ):
        url = _get_upload_url(test_workspace.name, session_name)
        response = client.post(url, files=files, data=form_data)

    assert response.status_code == 201
    data = response.json()
    assert len(data) > 1

    stmt = select(models.Message).where(
        models.Message.session_name == session_name,
        models.Message.peer_name == test_peer.name,
    )
    result = await db_session.execute(stmt)
    db_messages = list(result.scalars().all())

    assert len(db_messages) == len(data)
    for db_message in db_messages:
        assert db_message.internal_metadata["processing_type"] == "audio_transcription"
        assert db_message.internal_metadata["audio_segment_count"] == 3
        assert db_message.internal_metadata["transcription_provider"] == "openai"
        assert "transcription_fallback_used" not in db_message.internal_metadata
        assert "chunk_index" in db_message.internal_metadata
        assert "total_chunks" in db_message.internal_metadata
