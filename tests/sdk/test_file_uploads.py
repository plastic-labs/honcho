import json
from io import BytesIO

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.utils.file_upload import normalize_file_input


def test_normalize_file_input_with_iobase_tuple():
    """
    Test normalize_file_input with tuple containing IOBase object.
    This specifically tests lines 37-38 in file_upload.py.
    """
    # Create a BytesIO object (which is an IOBase)
    test_content = b"This is test content"
    file_obj = BytesIO(test_content)
    file_obj.name = "test.txt"

    # Test tuple with IOBase object
    filename = "test_file.txt"
    content_type = "text/plain"
    input_tuple = (filename, file_obj, content_type)

    # Call normalize_file_input - this should hit lines 37-38
    result = normalize_file_input(input_tuple)

    # Verify the result
    assert result == (filename, file_obj, content_type)
    assert result[0] == filename
    assert result[1] is file_obj  # Should be the same object
    assert result[2] == content_type


def test_normalize_file_input_with_invalid_file_content():
    """
    Test normalize_file_input with invalid file content in tuple.
    This specifically tests line 40 in file_upload.py.
    """
    # Test tuple with invalid file content (string instead of bytes or IOBase)
    filename = "test_file.txt"
    invalid_content = "This is a string, not bytes or IOBase"  # Invalid type
    content_type = "text/plain"
    input_tuple = (filename, invalid_content, content_type)

    # Call normalize_file_input - this should hit line 40 and raise ValueError
    with pytest.raises(
        ValueError, match="File content must be bytes or a file-like object."
    ):
        normalize_file_input(input_tuple)  # pyright: ignore


def test_normalize_file_input_with_file_object_without_name():
    """
    Test normalize_file_input with file object that lacks .name attribute.
    This specifically tests line 46 in file_upload.py.
    """
    # Create a BytesIO object without setting the .name attribute
    test_content = b"This is test content"
    file_obj = BytesIO(test_content)
    # Explicitly do NOT set file_obj.name to trigger the error condition

    # Call normalize_file_input - this should hit line 46 and raise ValueError
    with pytest.raises(
        ValueError, match="File object must have a \\.name attribute\\."
    ):
        normalize_file_input(file_obj)


@pytest.mark.asyncio
async def test_session_upload_file(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests uploading a single file to a session.
    """
    honcho_client, _client_type = client_fixture

    # Create test file
    text_content = (
        "This is a test text file.\nIt has multiple lines.\nFor testing purposes."
    )

    # Create file object for testing the flexible interface
    from io import BytesIO

    text_file = BytesIO(text_content.encode("utf-8"))
    text_file.name = "test.txt"

    # Handle sync and async clients separately
    if isinstance(honcho_client, Honcho):
        # Sync client
        session = honcho_client.session(id="test-session-upload")
        user = honcho_client.peer(id="user-upload")
        messages = session.upload_file(
            file=text_file,
            peer_id=user.id,
        )
    else:
        # Async client
        session = await honcho_client.session(id="test-session-upload")
        user = await honcho_client.peer(id="user-upload")
        messages = await session.upload_file(
            file=text_file,
            peer_id=user.id,
        )

    # Verify messages were created
    assert len(messages) >= 1

    # Check first message (text file)
    assert text_content in messages[0].content
    assert messages[0].peer_id == user.id
    assert messages[0].session_id == session.id


@pytest.mark.asyncio
async def test_large_file_chunking(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that large files get split into multiple messages automatically.
    """
    honcho_client, _client_type = client_fixture

    # Create a large text file that will require chunking
    large_content = "This is a test line.\n" * 3000  # Should exceed 49500 chars

    # Create file object for testing the flexible interface
    from io import BytesIO

    large_file = BytesIO(large_content.encode("utf-8"))
    large_file.name = "large_test.txt"

    # Handle sync and async clients separately
    if isinstance(honcho_client, Honcho):
        # Sync client
        session = honcho_client.session(id="test-session-chunking")
        user = honcho_client.peer(id="user-chunking")
        messages = session.upload_file(
            file=large_file,
            peer_id=user.id,
        )
    else:
        # Async client
        session = await honcho_client.session(id="test-session-chunking")
        user = await honcho_client.peer(id="user-chunking")
        messages = await session.upload_file(
            file=large_file,
            peer_id=user.id,
        )

    # Should be multiple messages due to chunking
    assert len(messages) > 1

    # All messages should have the same peer_id and session_id
    for message in messages:
        assert message.peer_id == user.id
        assert message.session_id == session.id


@pytest.mark.asyncio
async def test_multiple_files_upload(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests uploading multiple files one by one.
    """
    honcho_client, _client_type = client_fixture

    # Create multiple files
    file1_content = "Content of first file"
    file2_content = "Content of second file"
    file3_content = "Content of third file"

    # Create file objects for testing the flexible interface
    from io import BytesIO

    file1 = BytesIO(file1_content.encode("utf-8"))
    file1.name = "file1.txt"

    file2 = BytesIO(file2_content.encode("utf-8"))
    file2.name = "file2.txt"

    file3 = BytesIO(file3_content.encode("utf-8"))
    file3.name = "file3.txt"

    # Handle sync and async clients separately
    if isinstance(honcho_client, Honcho):
        # Sync client
        session = honcho_client.session(id="test-session-multiple")
        user = honcho_client.peer(id="user-multiple")
        messages1 = session.upload_file(file=file1, peer_id=user.id)
        messages2 = session.upload_file(file=file2, peer_id=user.id)
        messages3 = session.upload_file(file=file3, peer_id=user.id)
    else:
        # Async client
        session = await honcho_client.session(id="test-session-multiple")
        user = await honcho_client.peer(id="user-multiple")
        messages1 = await session.upload_file(file=file1, peer_id=user.id)
        messages2 = await session.upload_file(file=file2, peer_id=user.id)
        messages3 = await session.upload_file(file=file3, peer_id=user.id)

    # Should be at least one message per file
    assert len(messages1) >= 1
    assert len(messages2) >= 1
    assert len(messages3) >= 1

    # Check that all files were processed
    assert file1_content in messages1[0].content
    assert file2_content in messages2[0].content
    assert file3_content in messages3[0].content

    # All messages should have correct peer_id and session_id
    for messages in [messages1, messages2, messages3]:
        for message in messages:
            assert message.peer_id == user.id
            assert message.session_id == session.id


@pytest.mark.asyncio
async def test_json_file_upload(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests uploading JSON files specifically.
    """
    honcho_client, _client_type = client_fixture

    # Create JSON file
    json_data = {
        "name": "test_json",
        "values": [1, 2, 3, 4, 5],
        "nested": {"key": "value", "array": ["a", "b", "c"]},
        "boolean": True,
        "null_value": None,
    }
    json_content = json.dumps(json_data)

    # Create file object for testing the flexible interface
    from io import BytesIO

    json_file = BytesIO(json_content.encode("utf-8"))
    json_file.name = "test.json"

    # Handle sync and async clients separately
    if isinstance(honcho_client, Honcho):
        # Sync client
        session = honcho_client.session(id="test-session-json")
        user = honcho_client.peer(id="user-json")
        messages = session.upload_file(
            file=json_file,
            peer_id=user.id,
        )
    else:
        # Async client
        session = await honcho_client.session(id="test-session-json")
        user = await honcho_client.peer(id="user-json")
        messages = await session.upload_file(
            file=json_file,
            peer_id=user.id,
        )

    # Should create at least one message
    assert len(messages) >= 1

    # Check that JSON content is properly formatted in the message
    message_content = messages[0].content
    assert '"name": "test_json"' in message_content
    assert '"values": [1, 2, 3, 4, 5]' in message_content
    assert '"nested": {' in message_content
    assert '"key": "value"' in message_content
    assert '"array": ["a", "b", "c"]' in message_content
    assert '"boolean": true' in message_content
    assert '"null_value": null' in message_content

    assert messages[0].peer_id == user.id
    assert messages[0].session_id == session.id


@pytest.mark.asyncio
async def test_file_upload_with_tuple_input(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests uploading files using tuple input format.
    """
    honcho_client, _client_type = client_fixture

    # Create test content
    content = "This is test content for tuple input"
    filename = "tuple_test.txt"
    content_type = "text/plain"

    # Handle sync and async clients separately
    if isinstance(honcho_client, Honcho):
        # Sync client
        session = honcho_client.session(id="test-session-tuple")
        user = honcho_client.peer(id="user-tuple")
        messages = session.upload_file(
            file=(filename, content.encode("utf-8"), content_type),
            peer_id=user.id,
        )
    else:
        # Async client
        session = await honcho_client.session(id="test-session-tuple")
        user = await honcho_client.peer(id="user-tuple")
        messages = await session.upload_file(
            file=(filename, content.encode("utf-8"), content_type),
            peer_id=user.id,
        )

    # Should create at least one message
    assert len(messages) >= 1
    assert content in messages[0].content
    assert messages[0].peer_id == user.id
    assert messages[0].session_id == session.id
